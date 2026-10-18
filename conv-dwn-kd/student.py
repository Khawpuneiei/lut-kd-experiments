"""Train a flat or convolutional DWN student, with or without distillation from the CNN teacher.

  --arch flat     the pilot's model: LUTLayer(learnable) -> LUTLayer(random) -> GroupSum
  --arch conv     ConvDWN: [ConvLUTTree -> OrPool] x blocks -> LUTLayer -> GroupSum
  --init          residual (Petersen's pass-through init) or gaussian (uniform random tables); conv only
  --mode ce       cross-entropy on labels
  --mode kd       Hinton KD on logits: (1-a)*CE + a*T^2*KL(teacher_T || student_T)
  --mode kd_feat  kd + beta * MSE(proj(student block-2 bits), teacher block-2 features); conv only.
                  The 1x1 projection is a training-only module and is not part of the deployed circuit.
  --tau-scale     multiplies GroupSum's tau (Petersen: raise it by sqrt(2) when using a teacher)

Every run appends one JSON row to --out with the full config, the per-epoch test curve, final/last-3
accuracy, train accuracy, and the deployed-circuit size and depth from ConvDWN.circuit_stats().
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import common
from conv_dwn import DistributiveThermometer, make_conv_dwn, make_flat_dwn, to_bitplanes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch", default="conv", choices=["flat", "conv"])
    p.add_argument("--mode", default="ce", choices=["ce", "kd", "kd_feat"])
    p.add_argument("--init", default="residual", choices=["residual", "gaussian"])
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--channels", default="32,64,64", help="conv: tree channels per block")
    p.add_argument("--depth", type=int, default=3, help="conv: tree depth (2**depth leaf bits)")
    p.add_argument("--head", type=int, default=4000, help="conv: LUTs in the flat head layer")
    p.add_argument("--head-mapping", default="random", choices=["random", "learnable"],
                   help="conv: how the head LUTs pick their 2 inputs from the last feature map")
    p.add_argument("--n1", type=int, default=8000, help="flat: first-layer LUTs")
    p.add_argument("--n2", type=int, default=8000, help="flat: second-layer LUTs")
    p.add_argument("--tau", type=float, default=0, help="GroupSum tau; default = LUTs per class / 40")
    p.add_argument("--tau-scale", type=float, default=1.0)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=1.0, help="kd_feat: weight of the feature hint loss")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--step", type=int, default=0, help="StepLR step; default = 2/3 of epochs")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val", action="store_true", help="evaluate on the held-out 10% instead of test")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/runs.jsonl")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.mode == "kd_feat":
        assert args.arch == "conv", "feature KD needs the convolutional student"

    # ---- data: thermometer bits, as (N, C*H*W*bits) for flat and (N, C*bits, H, W) for conv
    xtr, ytr, xev, yev, train_idx = common.split(args.dataset, val=args.val)
    n_tr, (c, h, w) = len(ytr), tuple(xtr.shape[1:])
    therm = DistributiveThermometer(args.bits).fit(xtr.flatten(1).float())
    btr, bev = therm.binarize(xtr.flatten(1).float()), therm.binarize(xev.flatten(1).float())
    del xtr, xev
    if args.arch == "conv":
        btr, bev = to_bitplanes(btr, c, h, w), to_bitplanes(bev, c, h, w)
    else:
        btr, bev = btr.flatten(1), bev.flatten(1)
    btr, bev, ytr_d, yev_d = btr.to(dev), bev.to(dev), ytr.to(dev), yev.to(dev)

    # ---- teacher outputs
    t_logits = t_feats = None
    if args.mode != "ce":
        teacher = torch.load(os.path.join(common.CACHE, f"teacher_{args.dataset}.pt"))
        t_logits = teacher["train_logits"][train_idx].to(dev)
        if args.mode == "kd_feat":
            assert "train_features" in teacher, "rerun teacher.py with --save-features"
            t_feats = teacher["train_features"][train_idx].float()
            mu, sd = t_feats.mean(dim=(0, 2, 3), keepdim=True), t_feats.std(dim=(0, 2, 3), keepdim=True) + 1e-6
            t_feats = ((t_feats - mu) / sd).to(dev)                  # per-channel standardised

    # ---- student
    if args.arch == "conv":
        channels = [int(x) for x in args.channels.split(",")]
        tau = args.tau or args.head / 10 / 40
        model = make_conv_dwn(c * args.bits, h, channels, args.head, 10, tau * args.tau_scale,
                              depth=args.depth, init=args.init, head_mapping=args.head_mapping)
        stats = model.circuit_stats()
    else:
        tau = args.tau or args.n2 / 10 / 40
        model = make_flat_dwn(btr.shape[1], args.n1, args.n2, 10, tau * args.tau_scale)
        stats = {"gates_total": args.n1 + args.n2, "depth_levels": 2,
                 "trainable_tables": 4 * (args.n1 + args.n2)}
    model = model.to(dev)
    params = list(model.parameters())
    proj = None
    if args.mode == "kd_feat":
        assert len(channels) >= 2, "feature KD reads the block-2 output"
        proj = torch.nn.Conv2d(channels[1], t_feats.shape[1], 1).to(dev)   # training-only hint projection
        params += list(proj.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    step = args.step or max(1, round(args.epochs * 2 / 3))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.1)
    print(f"arch={args.arch} mode={args.mode} init={args.init} tau={tau * args.tau_scale:.2f} "
          f"device={dev} circuit={stats}", flush=True)

    def evaluate(b, y):
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(b[i:i + 1000].float()).argmax(1) for i in range(0, len(y), 1000)])
        model.train()
        return (pred == y).float().mean().item()

    t0, curve = time.time(), []
    for ep in range(args.epochs):
        perm = torch.randperm(n_tr, device=dev)
        tot = 0.0
        for s in range(0, n_tr, args.bs):
            b = perm[s:s + args.bs]
            x = btr[b].float()
            if args.mode == "kd_feat":
                logits, feats = model(x, return_features=True)
            else:
                logits = model(x)
            ce = F.cross_entropy(logits, ytr_d[b])
            if args.mode == "ce":
                loss = ce
            else:
                T = args.T
                kl = F.kl_div(F.log_softmax(logits / T, 1), F.log_softmax(t_logits[b] / T, 1),
                              reduction="batchmean", log_target=True) * T * T
                loss = (1 - args.alpha) * ce + args.alpha * kl
                if args.mode == "kd_feat":
                    sf = F.adaptive_avg_pool2d(feats[1], t_feats.shape[-1])   # block-2 bits -> 4x4 rates
                    loss = loss + args.beta * F.mse_loss(proj(sf), t_feats[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        acc = evaluate(bev, yev_d)
        curve.append(round(acc, 4))
        print(f"[{args.arch} {args.mode} s{args.seed}] epoch {ep + 1} loss {tot / n_tr:.4f} "
              f"{'val' if args.val else 'test'}_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    res = dict(vars(args), tau_used=tau * args.tau_scale, final_acc=curve[-1], best_acc=max(curve),
               last3_acc=round(sum(curve[-3:]) / len(curve[-3:]), 4), curve=curve,
               train_acc=evaluate(btr, ytr_d), seconds=round(time.time() - t0), device=dev, **stats)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: res[k] for k in ("arch", "mode", "init", "seed", "final_acc", "last3_acc",
                                          "train_acc", "gates_total", "depth_levels", "seconds")}))


if __name__ == "__main__":
    main()
