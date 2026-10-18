"""Runs the README's table of runs in priority order, one at a time (meant for a single GPU).

Resumable: a finished run's name goes into results/completed_runs.txt and is skipped on rerun.
Every command and exit code is appended to results/pipeline_log.txt.

    python3 pipeline.py              # teacher (if missing), then all runs
    EPOCHS=20 python3 pipeline.py    # different epoch count
    SEEDS=0 python3 pipeline.py      # one seed, for a quick first look
"""
import os
import subprocess
import sys
import time

import common

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOGS = os.path.join(RES, "logs")
DONE = os.path.join(RES, "completed_runs.txt")
os.makedirs(LOGS, exist_ok=True)

EPOCHS = os.environ.get("EPOCHS", "15")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
THREADS = os.environ.get("THREADS", "4")


def log(msg):
    with open(os.path.join(RES, "pipeline_log.txt"), "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def run(script, args, name):
    if os.path.exists(DONE) and name in open(DONE).read().split():
        log(f"skip {name} (done)")
        return 0
    cmd = [sys.executable, script] + [str(a) for a in args]
    log(f"start {name}: {' '.join(cmd[1:])}")
    with open(os.path.join(LOGS, f"{name}.log"), "w") as f:
        rc = subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log(f"end {name} rc={rc}")
    if rc == 0:
        with open(DONE, "a") as f:
            f.write(name + "\n")
    return rc


COMMON = ["--dataset", "cifar10", "--epochs", EPOCHS, "--threads", THREADS, "--out", "results/runs.jsonl"]
CONV = COMMON + ["--arch", "conv", "--channels", "32,64,64", "--head", "4000"]
FLAT = COMMON + ["--arch", "flat", "--n1", "8000", "--n2", "8000"]
SQRT2 = "1.41421356"


def jobs():
    """README run numbers in priority order 2, 4, 1, 5, 6, 3; each seed in turn so partial results are
    balanced across conditions."""
    out = []
    for s in SEEDS:
        seed = ["--seed", s]
        out += [
            ("student.py", CONV + seed + ["--mode", "ce"], f"conv_ce_s{s}"),                              # 2
            ("student.py", CONV + seed + ["--mode", "kd"], f"conv_kd_s{s}"),                              # 4
            ("student.py", FLAT + seed + ["--mode", "ce"], f"flat_ce_s{s}"),                              # 1
            ("student.py", CONV + seed + ["--mode", "kd", "--tau-scale", SQRT2], f"conv_kd_tau_s{s}"),    # 5
            ("student.py", CONV + seed + ["--mode", "kd_feat"], f"conv_kdfeat_s{s}"),                     # 6
        ]
        if s == SEEDS[0]:
            out.append(("student.py", CONV + seed + ["--mode", "ce", "--init", "gaussian"], "conv_ce_gauss_s0"))  # 3
    return out


if __name__ == "__main__":
    log(f"pipeline start (epochs {EPOCHS}, seeds {SEEDS})")
    tpath = os.path.join(common.CACHE, "teacher_cifar10.pt")
    if not os.path.exists(tpath) or "train_features" not in __import__("torch").load(tpath, map_location="cpu"):
        run("teacher.py", ["--dataset", "cifar10", "--epochs", 15, "--save-features", "--threads", THREADS], "teacher")
    for script, args, name in jobs():
        run(script, args, name)
    log("pipeline done")
