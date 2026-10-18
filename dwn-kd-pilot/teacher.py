"""Train a small CNN teacher and save its logits on the train and test sets.

The teacher is trained on 90% of the training set (the same split the student uses for
hyperparameter tuning), so it never sees the validation images.
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


def make_cnn(channels, widths):
    layers, c = [], channels
    for w in widths:
        layers += [block(c, w), block(w, w), nn.MaxPool2d(2)]
        c = w
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c, 10))


def make_cnn_fc(channels, widths, size):
    """Same conv stack with a fully connected head. A first Fashion-MNIST teacher built with
    make_cnn (global average pooling) underfit at 90.45% test accuracy, so 28x28 data uses this."""
    layers, c = [], channels
    for w in widths:
        layers += [block(c, w), block(w, w), nn.MaxPool2d(2)]
        c, size = w, size // 2
    return nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.3), nn.Linear(c * size * size, 256),
                         nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 10))


def augment(x, pad, flip):
    if pad:
        n, _, h, w = x.shape
        xp = F.pad(x, (pad, pad, pad, pad))
        i = torch.randint(0, 2 * pad + 1, (n,))
        j = torch.randint(0, 2 * pad + 1, (n,))
        idx_h = (i.view(n, 1) + torch.arange(h).view(1, h))
        idx_w = (j.view(n, 1) + torch.arange(w).view(1, w))
        xp = xp[torch.arange(n).view(n, 1, 1, 1), torch.arange(x.shape[1]).view(1, -1, 1, 1),
                idx_h.view(n, 1, h, 1), idx_w.view(n, 1, 1, w)]
        x = xp
    if flip:
        m = torch.rand(x.shape[0]) < 0.5
        x[m] = x[m].flip(-1)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    xtr_all, ytr_all, xte, yte, _ = common.split(args.dataset, val=False)
    xtr, ytr, _, _, train_idx = common.split(args.dataset, val=True)
    mean = xtr.float().mean(dim=(0, 2, 3), keepdim=True) / 255
    std = xtr.float().std(dim=(0, 2, 3), keepdim=True) / 255
    norm = lambda x: (x.float() / 255 - mean) / std

    if args.dataset == "cifar10":
        model, pad, flip = make_cnn(3, (32, 64, 128)), 4, True
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        model, pad, flip = make_cnn_fc(1, (32, 64), 28), 2, args.dataset == "fmnist"
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = args.epochs * ((len(ytr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps)

    def predict(x):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(norm(x[i:i + 1000])) for i in range(0, len(x), 1000)])

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for s in range(0, len(ytr), args.bs):
            b = perm[s:s + args.bs]
            loss = F.cross_entropy(model(augment(norm(xtr[b]), pad, flip)), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = (predict(xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {ep + 1} loss {loss.item():.3f} test_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    train_logits = predict(xtr_all)
    test_logits = predict(xte)
    res = {
        "dataset": args.dataset,
        "test_acc": (test_logits.argmax(1) == yte).float().mean().item(),
        "train_acc_full": (train_logits.argmax(1) == ytr_all).float().mean().item(),
        "train_acc_seen": (train_logits[train_idx].argmax(1) == ytr_all[train_idx]).float().mean().item(),
        "epochs": args.epochs,
        "params": sum(p.numel() for p in model.parameters()),
        "seconds": time.time() - t0,
    }
    torch.save({"train_logits": train_logits, "test_logits": test_logits},
               os.path.join(common.CACHE, f"teacher_{args.dataset}.pt"))
    os.makedirs("results", exist_ok=True)
    with open(f"results/teacher_{args.dataset}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
