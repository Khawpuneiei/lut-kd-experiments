"""Overnight driver: tuning on a held-out split, then final runs on the test set.

Stream A (Fashion-MNIST): tune tau for plain CE, tune KD (T, alpha), then all modes x 3 seeds.
Stream B (CIFAR-10): reuse the Fashion-MNIST settings, then all modes x seeds as time allows.
Every command and result is logged, so the whole night can be audited from results/.
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.environ.get("LOG_DIR", "/home/claude/work/logs")
CACHE = os.environ.get("CACHE_DIR", "/home/claude/work/cache")
os.makedirs(LOGS, exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
lock = threading.Lock()


def log(msg):
    with lock:
        with open(os.path.join(HERE, "results", "pipeline_log.txt"), "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}\n")


DONE = os.path.join(HERE, "results", "completed_runs.txt")


def run(args, name):
    """Runs one training job, skipping it if it already finished (so the pipeline can resume
    after the container restarts; a job killed mid-run wrote nothing and simply reruns)."""
    if os.path.exists(DONE) and name in open(DONE).read().split():
        log(f"skip {name} (already completed)")
        return 0
    cmd = [sys.executable, "student.py"] + [str(a) for a in args]
    log(f"start {name}: {' '.join(cmd[1:])}")
    with open(os.path.join(LOGS, f"{name}.log"), "w") as f:
        rc = subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log(f"end {name} rc={rc}")
    if rc == 0:
        with lock:
            with open(DONE, "a") as f:
                f.write(name + "\n")
    return rc


def run_parallel(jobs):
    threads = [threading.Thread(target=run, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def best(path, tag, key):
    rows = [json.loads(l) for l in open(os.path.join(HERE, path))]
    rows = [r for r in rows if r.get("tag") == tag]
    top = max(rows, key=lambda r: r["final_acc"])
    log(f"best {tag}: {[ (key(r), r['final_acc']) for r in rows ]} -> {key(top)}")
    return top


def wait_for(path):
    while not os.path.exists(path):
        time.sleep(20)


FM = ["--dataset", "fmnist", "--bits", 3, "--n1", 4000, "--n2", 4000, "--n", 2, "--bs", 256]
CF = ["--dataset", "cifar10", "--bits", 3, "--n1", 4000, "--n2", 4000, "--n", 2, "--bs", 256]
MODES = ["ce", "ce_ls", "kd", "kd_gated", "kd_conf"]
settings = {}
ready = threading.Event()


def prepare():
    """Teachers and tuning. The machine has 4 cores and PyTorch threads spin-wait, so
    oversubscribing them slows everything several-fold: every phase keeps to 4 threads in total."""
    if not os.path.exists(os.path.join(CACHE, "teacher_fmnist.pt")):
        log("start teacher fmnist")
        with open(os.path.join(LOGS, "teacher_fmnist.log"), "w") as f:
            subprocess.call([sys.executable, "teacher.py", "--dataset", "fmnist", "--epochs", "12", "--threads", "2"],
                            cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        log("end teacher fmnist")
    wait_for(os.path.join(CACHE, "teacher_fmnist.pt"))
    for pid in os.environ.get("RESUME_PIDS", "").split():
        os.kill(int(pid), 18)  # SIGCONT: resume the CIFAR-10 teacher paused while this teacher ran
        log(f"resumed pid {pid}")
    # Tuning runs alone on the machine (stream B waits for its result), so 4 one-thread jobs at once.
    tune = ["--val", "--epochs", 5, "--threads", 1, "--out", "results/tune_fmnist.jsonl"]
    run_parallel([(FM + tune + ["--mode", "ce", "--tau", t, "--tag", "tau"], f"tune_tau{t}") for t in (3.33, 10, 30, 100)])
    tau = best("results/tune_fmnist.jsonl", "tau", lambda r: r["tau"])["tau"]
    grid = [(T, a) for T in (2, 4) for a in (0.5, 0.9)]
    run_parallel([(FM + tune + ["--mode", "kd", "--tau", tau, "--T", T, "--alpha", a, "--tag", "kd"], f"tune_kd_T{T}_a{a}")
                  for T, a in grid])
    top = best("results/tune_fmnist.jsonl", "kd", lambda r: (r["T"], r["alpha"]))
    settings.update(tau=tau, T=top["T"], alpha=top["alpha"])
    log(f"settings {settings}")
    wait_for(os.path.join(CACHE, "teacher_cifar10.pt"))


def final_jobs():
    """All test-set runs in priority order, so whatever finishes first is the most useful: the CIFAR-10
    KD-versus-no-KD pairs come first, since CIFAR-10 is where seed 0 showed a gain. Finished runs are skipped."""
    s = ["--tau", settings["tau"], "--T", settings["T"], "--alpha", settings["alpha"], "--threads", 2]
    fm = FM + s + ["--epochs", EPOCHS_FM, "--out", "results/runs_fmnist.jsonl", "--tag", "final"]
    cf = CF + s + ["--epochs", EPOCHS_CF, "--out", "results/runs_cifar10.jsonl", "--tag", "final"]
    # Reference: the same CIFAR-10 model with 3-input LUTs, which HeLutNet measures as 7-10x slower under FHE.
    cf3 = ["--dataset", "cifar10", "--bits", 3, "--n1", 4000, "--n2", 4000, "--n", 3, "--bs", 256] + s + [
        "--epochs", EPOCHS_CF, "--out", "results/runs_cifar10_n3.jsonl", "--tag", "n3"]
    # HeLutNet's exact Fashion-MNIST model (7-bit thermometer, 8000 + 8000 LUTs, n = 2). Twice the LUTs per
    # class doubles the popcount range, so tau is doubled to keep the softmax equally sharp.
    ex = ["--dataset", "fmnist", "--bits", 7, "--n1", 8000, "--n2", 8000, "--n", 2, "--bs", 256,
          "--tau", 2 * settings["tau"], "--T", settings["T"], "--alpha", settings["alpha"], "--threads", 2,
          "--epochs", EPOCHS_FM, "--out", "results/runs_fmnist_exact.jsonl", "--tag", "exact"]
    # Twice the training. On Fashion-MNIST, KD led early and plain training caught up by the last epochs,
    # so this checks whether the CIFAR-10 gain is only faster convergence.
    cfl = CF + s + ["--epochs", 2 * EPOCHS_CF, "--out", "results/runs_cifar10_long.jsonl", "--tag", "long"]

    def fmj(m, seed):
        return fm + ["--mode", m, "--seed", seed], f"fmnist_{m}_s{seed}"

    def cfj(m, seed):
        return cf + ["--mode", m, "--seed", seed], f"cifar10_{m}_s{seed}"

    def n3j(seed):
        return cf3 + ["--mode", "ce", "--seed", seed], f"cifar10_n3_ce_s{seed}"

    def exj(m):
        return ex + ["--mode", m, "--seed", 0], f"fmnist_exact_{m}_s0"

    def lgj(m):
        return cfl + ["--mode", m, "--seed", 0], f"cifar10_long_{m}_s0"

    jobs = [fmj(m, seed) for seed in (0, 1) for m in MODES] + [cfj(m, 0) for m in MODES]
    jobs += [cfj(m, seed) for m, seed in (("ce", 1), ("kd", 1), ("ce", 2), ("kd", 2), ("ce_ls", 1), ("kd_gated", 1))]
    jobs += [n3j(0), lgj("ce"), lgj("kd"), n3j(1), cfj("kd_conf", 1), exj("ce")]
    jobs += [fmj(m, 2) for m in MODES]
    jobs += [exj("kd")]
    jobs += [cfj(m, 2) for m in ("ce_ls", "kd_gated", "kd_conf")]
    return jobs


def alive(pid):
    """True while the process runs. A finished child of a killed pipeline may linger as a zombie
    if nothing reaps it, so the zombie state counts as finished."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def worker(jobs, wait_pid):
    if wait_pid:  # a job left running by the previous pipeline process: let it finish first
        while alive(wait_pid):
            time.sleep(10)
    while True:
        with lock:
            if not jobs:
                return
            args, name = jobs.pop(0)
        run(args, name)


if __name__ == "__main__":
    EPOCHS_FM = int(os.environ.get("EPOCHS_FM", 12))
    EPOCHS_CF = int(os.environ.get("EPOCHS_CF", 10))
    log(f"pipeline start (fmnist epochs {EPOCHS_FM}, cifar10 epochs {EPOCHS_CF})")
    prepare()
    jobs = final_jobs()
    pids = (os.environ.get("WAIT_PIDS", "").split() + [None, None])[:2]
    workers = [threading.Thread(target=worker, args=(jobs, p)) for p in pids]  # 2 workers x 2 threads = 4 cores
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    log("pipeline done")
