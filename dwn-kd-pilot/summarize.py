"""Builds the results tables in RESULTS.md from results/*.jsonl.

Primary metric, fixed before the runs: test accuracy after the last epoch. Secondary metric: the mean
test accuracy over the last three epochs, added after seeing that test accuracy moves by about half a
point from one epoch to the next. It selects nothing using the test set; it only averages out that noise.
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["ce", "ce_ls", "kd", "kd_gated", "kd_conf"]
NAMES = {"ce": "No KD (how HeLutNet trains)", "ce_ls": "Label smoothing 0.1 (control)", "kd": "Standard KD",
         "kd_gated": "KD only where teacher is right", "kd_conf": "KD weighted by teacher confidence"}


def rows(name, tag):
    path = os.path.join(HERE, "results", name)
    if not os.path.exists(path):
        return []
    out = [r for r in map(json.loads, open(path)) if r.get("tag") == tag]
    keys = [(r["mode"], r["seed"]) for r in out]
    if len(keys) != len(set(keys)):
        print(f"WARNING: {name} has repeated (mode, seed) rows; the last one is used")
    return out


def last3(r):
    return st.mean(r["curve"][-3:])


def fmt(xs):
    if not xs:
        return "–"
    m = 100 * st.mean(xs)
    return f"{m:.2f} ± {100 * st.stdev(xs):.2f}" if len(xs) > 1 else f"{m:.2f}"


def gains(by, m, key):
    common = sorted(set(by[m]) & set(by["ce"]))
    return [100 * (key(by[m][s]) - key(by["ce"][s])) for s in common]


def table(runs):
    by = {m: {r["seed"]: r for r in runs if r["mode"] == m} for m in MODES}
    out = ["| Training | Test accuracy % | Gain over no KD, same seeds (points) | Same, last-3-epoch mean | "
           "Train accuracy % | Seeds |", "|---|---|---|---|---|---|"]
    for m in MODES:
        if not by[m]:
            continue
        rs = list(by[m].values())
        d, d3 = gains(by, m, lambda r: r["final_acc"]), gains(by, m, last3)
        g = "–" if m == "ce" or not d else f"{st.mean(d):+.2f} ({', '.join(f'{x:+.2f}' for x in d)})"
        g3 = "–" if m == "ce" or not d3 else f"{st.mean(d3):+.2f} ({', '.join(f'{x:+.2f}' for x in d3)})"
        out.append(f"| {NAMES[m]} | {fmt([r['final_acc'] for r in rs])} | {g} | {g3} | "
                   f"{fmt([r['train_acc'] for r in rs])} | {', '.join(str(s) for s in sorted(by[m]))} |")
    return "\n".join(out)


if __name__ == "__main__":
    for ds in ("fmnist", "cifar10"):
        t = json.load(open(os.path.join(HERE, "results", f"teacher_{ds}.json")))
        print(f"\n## {ds} (teacher CNN: {100 * t['test_acc']:.2f}% test)\n")
        print(table(rows(f"runs_{ds}.jsonl", "final")))
    n2 = {(r["mode"], r["seed"]): r for r in rows("runs_cifar10.jsonl", "final")}
    print("\n## cifar10 with 3-input LUTs (n = 3), no KD\n")
    n3 = rows("runs_cifar10_n3.jsonl", "n3")
    seeds = [r["seed"] for r in n3 if ("ce", r["seed"]) in n2 and ("kd", r["seed"]) in n2]
    if seeds:
        avg = lambda rs: f"{100 * st.mean(r['final_acc'] for r in rs):.2f}"
        print(f"mean over seeds {seeds}: n=3 no KD {avg([r for r in n3 if r['seed'] in seeds])}, "
              f"n=2 no KD {avg([n2[('ce', s)] for s in seeds])}, n=2 KD {avg([n2[('kd', s)] for s in seeds])}")
    for r in n3:
        ce, kd = n2.get(("ce", r["seed"])), n2.get(("kd", r["seed"]))
        print(f"seed {r['seed']}: n=3 {100 * r['final_acc']:.2f} (last-3 {100 * last3(r):.2f}, train "
              f"{100 * r['train_acc']:.2f}, {r['seconds']} s)"
              + (f"; n=2 no KD {100 * ce['final_acc']:.2f}" if ce else "")
              + (f"; n=2 KD {100 * kd['final_acc']:.2f}" if kd else ""))
    long = {r["mode"]: r for r in rows("runs_cifar10_long.jsonl", "long")}
    print("\n## cifar10, twice the training (20 epochs, learning rate drop after epoch 13), seed 0\n")
    for m, r in long.items():
        print(f"{m}: {100 * r['final_acc']:.2f} (last-3 {100 * last3(r):.2f}, train {100 * r['train_acc']:.2f}, "
              f"{r['seconds']} s); curve " + " ".join(f"{100 * a:.1f}" for a in r["curve"]))
    if "ce" in long and "kd" in long:
        d = [100 * (k - c) for k, c in zip(long["kd"]["curve"], long["ce"]["curve"])]
        print(f"KD minus no KD, per epoch: " + " ".join(f"{x:+.1f}" for x in d))
    wide = {(r["mode"], r["seed"]): r for r in rows("runs_cifar10_wide.jsonl", "wide")}
    print("\n## cifar10, twice as wide (8000 + 8000 LUTs, n = 2, tau 20)\n")
    for (m, s), r in sorted(wide.items()):
        narrow = n2.get((m, s))
        print(f"{m} seed {s}: {100 * r['final_acc']:.2f} (last-3 {100 * last3(r):.2f}, train "
              f"{100 * r['train_acc']:.2f}, {r['seconds']} s); curve " + " ".join(f"{100 * a:.1f}" for a in r["curve"])
              + (f"; 4000 + 4000 same mode and seed: {100 * narrow['final_acc']:.2f}" if narrow else ""))
    for s in sorted({s for _, s in wide}):
        if ("ce", s) in wide and ("kd", s) in wide:
            c, k = wide[("ce", s)], wide[("kd", s)]
            print(f"seed {s}: KD minus no KD {100 * (k['final_acc'] - c['final_acc']):+.2f} "
                  f"(last-3 {100 * (last3(k) - last3(c)):+.2f}); per epoch "
                  + " ".join(f"{100 * (a - b):+.1f}" for a, b in zip(k["curve"], c["curve"])))
    print("\n## fmnist, HeLutNet's exact model (7 bits, 8000 + 8000 LUTs, n = 2; paper reports 89.75%)\n")
    for r in rows("runs_fmnist_exact.jsonl", "exact"):
        print(f"{r['mode']} seed {r['seed']}: {100 * r['final_acc']:.2f} (best epoch {100 * r['best_acc']:.2f}, "
              f"last-3 {100 * last3(r):.2f}, train {100 * r['train_acc']:.2f}, {r['seconds']} s)")
