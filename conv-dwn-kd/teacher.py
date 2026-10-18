"""Train the CNN teacher (same as the pilot's) and cache its logits, plus, with --save-features, its
block-2 feature maps pooled to 4x4 for feature-level distillation.

Teacher for CIFAR-10: three blocks of [conv-bn-relu x2, maxpool] with 32, 64, 128 channels. Block-2
output is 64 x 8 x 8; it is average-pooled to 64 x 4 x 4 and stored as float16 (50k x 1024 values).
Trained on 90% of the training set so the held-out 10% stays clean for tuning.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import common


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CNN(nn.Module):
    def __init__(self, channels, widths, feature_block=2):
        super().__init__()
        stages, c = [], channels
        for w in widths:
            stages.append(nn.Sequential(block(c, w), block(w, w), nn.MaxPool2d(2)))
            c = w
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c, 10))
        self.feature_block = feature_block

    def forward(self, x, return_features=False):
        feat = None
        for i, s in enumerate(self.stages):
            x = s(x)
            if i + 1 == self.feature_block:
                feat = x
        out = self.head(x)
        return (out, feat) if return_features else out


def augment(x, pad, flip):
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    ar = torch.arange
    idx_h = i.view(n, 1) + ar(h, device=x.device).view(1, h)
    idx_w = j.view(n, 1) + ar(w, device=x.device).view(1, w)
    x = xp[ar(n, device=x.device).view(n, 1, 1, 1), ar(x.shape[1], device=x.device).view(1, -1, 1, 1),
           idx_h.view(n, 1, h, 1), idx_w.view(n, 1, 1, w)]
    if flip:
        m = torch.rand(n, device=x.device) < 0.5
        x[m] = x[m].flip(-1)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--feat-size", type=int, default=4, help="spatial size the saved feature map is pooled to")
    args = p.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    xtr_all, ytr_all, xte, yte, _ = common.split(args.dataset, val=False)
    xtr, ytr, _, _, train_idx = common.split(args.dataset, val=True)
    mean = (xtr.float().mean(dim=(0, 2, 3), keepdim=True) / 255).to(dev)
    std = (xtr.float().std(dim=(0, 2, 3), keepdim=True) / 255).to(dev)
    norm = lambda x: (x.to(dev).float() / 255 - mean) / std

    model = CNN(xtr.shape[1], (32, 64, 128)).to(dev)
    pad, flip = 4, True
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = args.epochs * ((len(ytr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.05, total_steps=steps)

    def predict(x):
        model.eval()
        logits, feats = [], []
        with torch.no_grad():
            for i in range(0, len(x), 1000):
                lg, ft = model(norm(x[i:i + 1000]), return_features=True)
                logits.append(lg.cpu())
                feats.append(F.adaptive_avg_pool2d(ft, args.feat_size).half().cpu())
        return torch.cat(logits), torch.cat(feats)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for s in range(0, len(ytr), args.bs):
            b = perm[s:s + args.bs]
            loss = F.cross_entropy(model(augment(norm(xtr[b]), pad, flip)), ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = (predict(xte)[0].argmax(1) == yte).float().mean().item()
        print(f"epoch {ep + 1} loss {loss.item():.3f} test_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    train_logits, train_feats = predict(xtr_all)
    test_logits, _ = predict(xte)
    res = {
        "dataset": args.dataset,
        "test_acc": (test_logits.argmax(1) == yte).float().mean().item(),
        "train_acc_full": (train_logits.argmax(1) == ytr_all).float().mean().item(),
        "train_acc_seen": (train_logits[train_idx].argmax(1) == ytr_all[train_idx]).float().mean().item(),
        "epochs": args.epochs, "params": sum(p.numel() for p in model.parameters()),
        "feature_shape": list(train_feats.shape[1:]) if args.save_features else None,
        "seconds": round(time.time() - t0),
    }
    out = {"train_logits": train_logits, "test_logits": test_logits}
    if args.save_features:
        out["train_features"] = train_feats
    torch.save(out, os.path.join(common.CACHE, f"teacher_{args.dataset}.pt"))
    os.makedirs("results", exist_ok=True)
    with open(f"results/teacher_{args.dataset}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
