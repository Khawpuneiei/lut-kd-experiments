"""Train a DWN (LUT-based) student with or without distillation from the CNN teacher.

Modes (identical architecture, so identical HeLutNet/FHE inference cost):
  ce        plain cross-entropy on labels (how HeLutNet's models are trained)
  ce_ls     cross-entropy with label smoothing: a control, since part of KD's gain is often
            just softer targets rather than anything the teacher knows
  kd       Hinton KD: (1-a)*CE + a*T^2*KL(teacher_T || student_T)
  kd_gated  KD only on samples the teacher classifies correctly; label CE elsewhere
  kd_conf   per-sample mix weighted by the teacher's probability on the true label
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import common
from dwn import DistributiveThermometer, make_dwn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", default="ce", choices=["ce", "ce_ls", "kd", "kd_gated", "kd_conf"])
    p.add_argument("--ls", type=float, default=0.1, help="label smoothing for ce_ls (control for KD)")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--n1", type=int, default=2000)
    p.add_argument("--n2", type=int, default=2000)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--step", type=int, default=0, help="StepLR step; default = 2/3 of epochs")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val", action="store_true", help="evaluate on the held-out 10% instead of test")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/runs.jsonl")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    xtr, ytr, xev, yev, train_idx = common.split(args.dataset, val=args.val)
    xtr, xev = xtr.flatten(1).float(), xev.flatten(1).float()
    therm = DistributiveThermometer(args.bits).fit(xtr)
    btr, bev = therm.binarize(xtr), therm.binarize(xev)
    del xtr, xev

    teacher = torch.load(os.path.join(common.CACHE, f"teacher_{args.dataset}.pt"))
    t_logits = teacher["train_logits"][train_idx]
    t_prob1 = torch.softmax(t_logits, 1)
    t_correct = (t_logits.argmax(1) == ytr).float()
    t_conf = t_prob1.gather(1, ytr[:, None]).squeeze(1)

    model = make_dwn(btr.shape[1], args.n1, args.n2, args.n, 10, args.tau)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)  # fused: ~5x faster on CPU
    step = args.step or max(1, round(args.epochs * 2 / 3))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.1)

    def evaluate(b, y):
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(b[i:i + 1000].float()).argmax(1) for i in range(0, len(y), 1000)])
        model.train()
        return (pred == y).float().mean().item()

    t0, curve = time.time(), []
    for ep in range(args.epochs):
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for s in range(0, len(ytr), args.bs):
            b = perm[s:s + args.bs]
            logits = model(btr[b].float())
            ce = F.cross_entropy(logits, ytr[b], reduction="none")
            if args.mode == "ce":
                loss = ce.mean()
            elif args.mode == "ce_ls":
                loss = F.cross_entropy(logits, ytr[b], label_smoothing=args.ls)
            else:
                T = args.T
                kl = F.kl_div(F.log_softmax(logits / T, 1), F.log_softmax(t_logits[b] / T, 1),
                              reduction="none", log_target=True).sum(1) * T * T
                mixed = (1 - args.alpha) * ce + args.alpha * kl
                if args.mode == "kd":
                    w = torch.ones_like(ce)
                elif args.mode == "kd_gated":
                    w = t_correct[b]
                else:
                    w = t_conf[b]
                loss = (w * mixed + (1 - w) * ce).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        acc = evaluate(bev, yev)
        curve.append(round(acc, 4))
        print(f"[{args.mode} s{args.seed}] epoch {ep + 1} loss {tot / len(ytr):.4f} "
              f"{'val' if args.val else 'test'}_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    res = dict(vars(args), final_acc=curve[-1], best_acc=max(curve), curve=curve,
               train_acc=evaluate(btr, ytr), seconds=round(time.time() - t0),
               teacher_train_acc=t_correct.mean().item())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: res[k] for k in ("dataset", "mode", "seed", "final_acc", "train_acc", "seconds")}))


if __name__ == "__main__":
    main()
