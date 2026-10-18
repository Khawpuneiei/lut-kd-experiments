"""Prints the results table from results/runs.jsonl (one row per condition, mean ± sd over seeds,
gain over the conv `ce` baseline on matching seeds). Paste the output into RESULTS.md."""
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = [  # (arch, init, mode, tau_scale≈√2?) -> label, README run number
    (("flat", "ce", False), "1  Flat DWN 8k+8k, no KD"),
    (("conv", "ce", False), "2  Conv DWN, no KD (baseline)"),
    (("conv-gaussian", "ce", False), "3  Conv DWN, Gaussian init, no KD"),
    (("conv", "kd", False), "4  Conv DWN, logit KD"),
    (("conv", "kd", True), "5  Conv DWN, logit KD, tau x sqrt2"),
    (("conv", "kd_feat", False), "6  Conv DWN, logit + feature KD"),
]


def key(r):
    arch = r["arch"] + ("-gaussian" if r["arch"] == "conv" and r.get("init") == "gaussian" else "")
    return (arch, r["mode"], abs(r.get("tau_scale", 1.0) - 1.41421356) < 1e-3)


def fmt(xs):
    if not xs:
        return "–"
    m = 100 * st.mean(xs)
    return f"{m:.2f} ± {100 * st.stdev(xs):.2f}" if len(xs) > 1 else f"{m:.2f}"


def main(path=os.path.join(HERE, "results", "runs.jsonl")):
    if not os.path.exists(path):
        print("no results yet")
        return
    by = defaultdict(dict)
    for r in map(json.loads, open(path)):
        if r.get("val"):
            continue
        by[key(r)][r["seed"]] = r          # last row per (condition, seed) wins
    base = by.get(("conv", "ce", False), {})
    print("| # Condition | Test acc % (last epoch) | Last-3-epoch mean % | Gain vs conv no-KD, same seeds | "
          "Train acc % | Gates | Depth | Seeds | min/run |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, label in ORDER:
        rs = by.get(k)
        if not rs:
            continue
        vals = list(rs.values())
        common = sorted(set(rs) & set(base))
        gain = "–"
        if k != ("conv", "ce", False) and common:
            d = [100 * (rs[s]["final_acc"] - base[s]["final_acc"]) for s in common]
            gain = f"{st.mean(d):+.2f} ({', '.join(f'{x:+.2f}' for x in d)})"
        g = vals[0].get("gates_total", "–")
        print(f"| {label} | {fmt([r['final_acc'] for r in vals])} | {fmt([r['last3_acc'] for r in vals])} | {gain} | "
              f"{fmt([r['train_acc'] for r in vals])} | {g:,} | {vals[0].get('depth_levels', '–')} | "
              f"{', '.join(str(s) for s in sorted(rs))} | {st.mean(r['seconds'] for r in vals) / 60:.0f} |")
    print("\nPer-epoch curves (test acc %, seed 0):")
    for k, label in ORDER:
        r = by.get(k, {}).get(0)
        if r:
            print(f"  {label}: {' '.join(f'{100 * a:.1f}' for a in r['curve'])}")


if __name__ == "__main__":
    main()
