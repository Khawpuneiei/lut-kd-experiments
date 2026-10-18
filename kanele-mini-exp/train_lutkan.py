"""Train the KANELÉ-style LUT-KAN student with or without distillation from the pilot's CNN teacher.

Reuses ../dwn-kd-pilot for data loading (common.py) and the cached teacher logits
(teacher_<dataset>.pt in $CACHE_DIR), so run the pilot's teacher.py first.

Modes: ce (labels only), kd (Hinton KD, same loss as the pilot's `kd` mode).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dwn-kd-pilot"))
import common  # noqa: E402

from lutkan import LUTKAN, prune_threshold  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["mnist", "fmnist", "cifar10"])
    p.add_argument("--mode", default="ce", choices=["ce", "kd"])
    p.add_argument("--hidden", default="64", help="comma-separated hidden widths, e.g. 64 or 128,32")
    p.add_argument("--in_bits", type=int, default=2, help="bits per input pixel (KANELÉ MNIST used 1)")
    p.add_argument("--act_bits", type=int, default=4, help="bits between layers")
    p.add_argument("--prune", type=float, default=0.0, help="target edge-norm threshold T; 0 = no pruning")
    p.add_argument("--t0", type=int, default=3, help="epoch pruning starts")
    p.add_argument("--tf", type=int, default=8, help="epoch the threshold reaches 95% of T")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val", action="store_true", help="evaluate on the held-out 10% instead of test")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/runs.jsonl")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    xtr, ytr, xev, yev, train_idx = common.split(args.dataset, val=args.val)
    # Pixels to [-1, 1]; the input quantiser's learnable scale does the rest.
    dev = torch.device(args.device)
    xtr, xev = xtr.flatten(1).float() / 127.5 - 1, xev.flatten(1).float() / 127.5 - 1
    xtr, ytr, xev, yev = xtr.to(dev), ytr.to(dev), xev.to(dev), yev.to(dev)

    t_logits = None
    if args.mode == "kd":
        teacher = torch.load(os.path.join(common.CACHE, f"teacher_{args.dataset}.pt"))
        t_logits = teacher["train_logits"][train_idx].to(dev)

    hidden = [int(h) for h in args.hidden.split(",") if h]
    model = LUTKAN(xtr.shape[1], hidden, 10, args.in_bits, args.act_bits).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def evaluate(x, y):
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x[i:i + 1000]).argmax(1) for i in range(0, len(y), 1000)])
        model.train()
        return (pred == y).float().mean().item()

    t0, curve, edges = time.time(), [], []
    for ep in range(args.epochs):
        perm, tot = torch.randperm(len(ytr), device=dev), 0.0
        for s in range(0, len(ytr), args.bs):
            b = perm[s:s + args.bs]
            logits = model(xtr[b])
            loss = F.cross_entropy(logits, ytr[b])
            if args.mode == "kd":
                T = args.T
                kl = F.kl_div(F.log_softmax(logits / T, 1), F.log_softmax(t_logits[b] / T, 1),
                              reduction="batchmean", log_target=True) * T * T
                loss = (1 - args.alpha) * loss + args.alpha * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if args.prune > 0:
            model.prune(prune_threshold(ep, args.prune, args.t0, args.tf))
        acc = evaluate(xev, yev)
        curve.append(round(acc, 4))
        edges.append(model.active_edges())
        print(f"[{args.mode} s{args.seed}] epoch {ep + 1} loss {tot / len(ytr):.4f} "
              f"{'val' if args.val else 'test'}_acc {acc:.4f} edges {edges[-1]} ({time.time() - t0:.0f}s)", flush=True)

    res = dict(vars(args), final_acc=curve[-1], best_acc=max(curve), curve=curve, edges=edges[-1],
               table_bits=model.table_bits(), train_acc=evaluate(xtr, ytr), seconds=round(time.time() - t0))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: res[k] for k in ("dataset", "mode", "seed", "final_acc", "train_acc", "edges", "table_bits", "seconds")}))


if __name__ == "__main__":
    main()
