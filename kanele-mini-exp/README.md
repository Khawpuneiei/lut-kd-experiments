# Mini experiment: a KANELÉ-style student, with and without a CNN teacher

**Idea under test.** KANELÉ (Hoang, Gupta, Harris, MIT, FPGA '26; [arXiv 2512.12850](https://arxiv.org/abs/2512.12850))
turns a KAN into lookup tables: every connection carries its own learned curve, the input to the curve is
squeezed to *b* bits, so the curve becomes a table with 2^b rows. It beats DWN on tabular data but trails it on
images (96.3% vs 97.8% on MNIST) and has no CIFAR-10 result; its authors blame missing "spatial priors".
P' Palm's suggestion is to distil a CNN into a lookup-table model. This experiment asks:

1. **Does a CNN teacher help a KANELÉ-style student on images?** (same question the DWN pilot asked, new student)
2. **Which LUT student takes the teacher better, DWN or KAN?** A DWN table outputs one bit; a KAN table outputs a
   number. Soft teacher scores might survive better in the second.
3. **Does KANELÉ's pruning recipe keep accuracy while shrinking the model?** Your pilot found width is the
   cheapest accuracy lever for DWN; pruning is the matching cost lever.

**Results of the first full run (16 Oct 2026, CPU): [RESULTS.md](RESULTS.md).** Rerun everything with
`bash run_all.sh` (about 70 minutes on 4 CPU cores).

Notes on the paper itself: [../kanele-paper-notes.md](../kanele-paper-notes.md). Background numbers from the
DWN pilot: [../dwn-kd-pilot/RESULTS.md](../dwn-kd-pilot/RESULTS.md).

## What the student is

`lutkan.py` builds a KAN whose edges are tables rather than splines:

```
pixels -> b-bit quantiser (learnable scale) -> one table per (input, hidden) edge, summed per hidden unit
       -> a-bit quantiser -> one table per (hidden, class) edge, summed per class -> class scores
```

KANELÉ trains a B-spline on each edge and *then* enumerates it at the 2^b quantised inputs to make the table.
Since the spline is only ever read at those 2^b points, this script learns the 2^b entries directly. That is
the exact object KANELÉ synthesises ("logical LUT"), minus the spline's smoothness prior. If results look odd,
that prior is the first thing to add back (their code is MIT-licensed: https://github.com/Duchstf/KANELE).

Other things copied from the paper: learnable quantiser scales with straight-through gradients, AdamW, and
edge pruning by the L2 norm of each table (minus its mean) with a threshold that ramps up from epoch `t0` to
`tf`. Things not copied: adder-tree pipelining and the VHDL export, which do not affect accuracy.

Reference sizes: KANELÉ's MNIST model is 784 inputs -> 62 hidden -> 10, with **1-bit** inputs, after heavy
pruning. Default here is 2-bit inputs, 64 hidden, 4-bit between layers.

## Files

| File | What it does |
|---|---|
| `lutkan.py` | The student: quantisers, table-edge layer, pruning ramp. |
| `train_lutkan.py` | Trains it in `ce` or `kd` mode. Reuses `../dwn-kd-pilot/common.py` for data and the pilot's cached teacher logits. Appends one JSON line per run to `--out`. |
| `results/` | Where runs land. Keep the `.jsonl` files; they are the record. |

## Setup

Same as the pilot: Python 3 with `torch`, `numpy`, `pillow`; data under `$DATA_DIR` (default `/home/claude/data`),
teacher logits under `$CACHE_DIR`. If you already ran the pilot, nothing new is needed. Otherwise:

```bash
cd ../dwn-kd-pilot
python3 teacher.py --dataset fmnist --epochs 12 --threads 2      # ~93% CNN teacher, caches logits
python3 teacher.py --dataset cifar10 --epochs 15 --threads 2     # ~87%
cd ../kanele-mini-exp
```

## The runs

On a machine with a GPU, `train_lutkan.py` uses it automatically (override with `--device cpu` or
`--device cuda:1`). The pilot's `teacher.py` must still have cached `teacher_<dataset>.pt` first.

Everything below is on a CPU. A 10-epoch Fashion-MNIST run takes about 70 s on 2 threads; CIFAR-10 has
3072 inputs so expect roughly 4x that at the same width.

### Step 1: sanity, does the student train at all (Fashion-MNIST, about a minute)

```bash
python3 train_lutkan.py --dataset fmnist --mode ce --epochs 10 --seed 0 --out results/fmnist.jsonl
```

Smoke test on 15 Oct 2026 with the defaults (2-bit pixels, 64 hidden, 4-bit between layers, 50,816 edges):
**87.15%** test, 91.8% train, 67 s. The pilot's DWN (4,000 + 4,000, n = 2) gets 88.2%, so the student is in the
right range. `--prune 0.2` on the same run kept **86.9%** with half the first-layer edges gone (25,525 of
50,176 left). `kd` mode also runs end to end, but its smoke number (83.8%) used a throwaway 1-epoch teacher
at 88.4%, barely above the student, so it says nothing about distillation; use the pilot's real 93% teacher.

If plain training is far below 88%, raise `--hidden` or `--in_bits` before going on.

### Step 2: the main question, teacher vs no teacher (CIFAR-10, 3 seeds)

```bash
for s in 0 1 2; do for m in ce kd; do
  python3 train_lutkan.py --dataset cifar10 --mode $m --hidden 64 --in_bits 2 --act_bits 4 \
    --epochs 10 --T 2 --alpha 0.9 --seed $s --out results/cifar10.jsonl
done; done
```

`--T 2 --alpha 0.9` are the values the pilot tuned for the DWN student, so the comparison to the pilot is fair;
tune them for this student later if the gap looks real. Read the result the same way as the pilot: mean of
3 seeds, gain per seed, and whether `kd` is ahead at every epoch (the `curve` field).

### Step 3: which student takes the teacher better

Put this experiment's CIFAR-10 rows next to the pilot's:

| Student | No teacher | With teacher | Gain |
|---|---|---|---|
| DWN 4,000 + 4,000, n = 2 (pilot) | 49.3 | 50.9 | +1.6 |
| DWN 8,000 + 8,000, n = 2 (pilot) | 52.9 | 52.4 | -0.5 |
| LUT-KAN, 64 hidden | ? | ? | ? |

Match on something fair before comparing: either accuracy without a teacher (pick `--hidden` so the `ce` rows
are close) or model size (`table_bits` in the JSON: surviving edges x 2^bits x 8).

### Step 4: pruning (does the KANELÉ recipe cost accuracy?)

```bash
for T in 0 0.1 0.2 0.3 0.5; do
  python3 train_lutkan.py --dataset cifar10 --mode kd --prune $T --t0 3 --tf 8 \
    --epochs 10 --seed 0 --tag prune --out results/cifar10_prune.jsonl
done
```

Plot `final_acc` against `edges` (or `table_bits`). The interesting point is the smallest model that keeps the
teacher's gain. On Fashion-MNIST after 3 epochs, first-layer edge norms had a median of 0.18 (10th to 90th
percentile 0.07 to 0.46) and second-layer norms a median of 0.88, so one threshold prunes almost only the first
layer; `0.2` removed about half of it. If a threshold prunes nothing or everything on CIFAR-10, shift the range.

### Optional: match KANELÉ's own MNIST setting

```bash
python3 train_lutkan.py --dataset mnist --mode ce --hidden 62 --in_bits 1 --act_bits 4 --epochs 20 --prune 0.05
```

KANELÉ reports 96.3% here (Table 2/3). Landing within a point or two says the re-implementation is close
enough to trust; MNIST needs the four `.gz` files in `$DATA_DIR/mnist/`.

## What to write down

One table per step, with seeds and the exact command, in a `RESULTS.md` like the pilot's. The claim you can
make from this, if it holds: *a CNN teacher helps (or does not help) a KANELÉ-style LUT student on CIFAR-10,
and the KAN student takes the teacher better (or worse) than DWN at matched size.* Either answer is useful
to P' Palm, because neither paper tried it.

## Caveats before anyone quotes a number

- This is a CPU re-implementation of the learned object, not KANELÉ's code; the spline prior is dropped.
- No FHE angle: whether a multi-bit-output table is cheap in HeLutNet is an open question for P' Palm, not
  something this experiment answers.
- Small models, 10 epochs. Same honesty rule as the pilot: a reason to test properly, not a finding.
