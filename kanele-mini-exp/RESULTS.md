# Results: KANELÉ-style student, with and without a CNN teacher

Run on 16 October 2026 on a 4-core CPU (two jobs at a time, 2 threads each), with `run_all.sh`. The whole
thing took 68 minutes. Raw results are in `results/*.jsonl` (one line per run, with the per-epoch `curve`),
and the logs are in `results/logs/`. Setup, commands and caveats are in [README.md](README.md).

## Short answer

- **CIFAR-10: the teacher helped the KAN student too, by about as much as it helped the small DWN.** Test
  accuracy rose from **46.8% to 47.9%** (+1.1 points, mean of 3 seeds; +1.2 over the last three epochs).
  The distilled model was ahead at every epoch of every seed. The pilot's small DWN gained +1.6.
- **Fashion-MNIST: no gain** (-0.2 points over 3 seeds), the same as the DWN pilot.
- **The KAN student is worse than the DWN and far bigger at this setting.** It is 2.5 points below the small
  DWN on CIFAR-10 with no teacher (46.8 vs 49.3), and its tables hold about 200 times more bits (see
  "Size"). So on this evidence the answer to "which LUT student takes the teacher better?" is "neither
  clearly". The gains are similar, and the DWN is the better model.
- **KANELÉ's pruning cut a third of the first layer cheaply, and then accuracy fell off fast.** Threshold 0.1
  removed 41% of the first-layer edges for -0.8 points. Threshold 0.2 removed 81% for -4.5.
- **The teachers are the pilot's, retrained bit for bit** (93.27% Fashion-MNIST, 86.99% CIFAR-10, same
  seed and code), so this run's teacher gain is directly comparable to the pilot's.

## Setup

| | |
|---|---|
| Student | 784 or 3072 inputs → 64 hidden → 10 classes. One table per edge: 2-bit pixels (4 entries) into the hidden layer, then 4-bit activations (16 entries) into the class layer. 197k edges on CIFAR-10, 51k on Fashion-MNIST. |
| Training | AdamW, lr 3e-3, weight decay 1e-4, cosine schedule, batch 256, 10 epochs. About 85 s per Fashion-MNIST run and 270 s per CIFAR-10 run. |
| Distillation | Same loss and values as the pilot: `(1 − α)·CE + α·T²·KL`, T = 2, α = 0.9, not re-tuned for this student. |
| Metric | Test accuracy after the last epoch, plus the mean of the last three epochs. Each seed changes the student's init and batch order; the teacher is fixed. |

## CIFAR-10 (3 seeds)

| Training | Seed 0 | Seed 1 | Seed 2 | Mean | Last-3 mean | Train acc |
|---|---|---|---|---|---|---|
| No teacher | 46.47 | 47.26 | 46.61 | 46.78 ± 0.42 | 46.65 | 54.64 |
| Teacher (KD) | 48.04 | 47.79 | 47.75 | 47.86 ± 0.16 | 47.81 | 55.25 |
| Gain | +1.57 | +0.53 | +1.14 | **+1.08** | +1.16 | +0.61 |

KD minus no-teacher at each epoch:
- Seed 0: +2.7, +3.0, +1.9, +1.9, +2.1, +1.3, +1.3, +1.4, +1.4, +1.6
- Seed 1: +2.6, +0.9, +2.0, +1.7, +0.6, +0.6, +1.1, +1.2, +0.7, +0.5
- Seed 2: +3.4, +0.8, +1.6, +1.5, +1.8, +1.3, +1.2, +1.4, +1.3, +1.1

One difference from the DWN pilot: there, distillation *lowered* training accuracy in every setting. Here
it raised it slightly (+0.6). The KAN student underfits (55% train), so the teacher may be making it easier to
fit rather than regularising it. That is a guess from one number.

## Fashion-MNIST (3 seeds)

| Training | Seed 0 | Seed 1 | Seed 2 | Mean | Last-3 mean | Train acc |
|---|---|---|---|---|---|---|
| No teacher | 87.15 | 87.49 | 87.21 | 87.28 ± 0.18 | 87.16 | 91.66 |
| Teacher (KD) | 87.11 | 87.08 | 87.05 | 87.08 ± 0.03 | 87.02 | 90.17 |
| Gain | -0.04 | -0.41 | -0.16 | **-0.20** | -0.14 | -1.49 |

## Next to the DWN pilot (CIFAR-10, mean of 3 seeds, same teacher)

| Student | No teacher | Teacher | Gain | Table bits |
|---|---|---|---|---|
| DWN 4,000 + 4,000, n = 2 (pilot) | 49.28 | 50.92 | +1.63 | about 32 k |
| DWN 8,000 + 8,000, n = 2 (pilot, 2 seeds) | 52.89 | 52.38 | -0.51 | about 64 k |
| LUT-KAN, 64 hidden (this run) | 46.78 | 47.86 | +1.08 | about 6.4 M |

**Size.** A DWN table with n = 2 holds 4 one-bit entries, so 8,000 tables is about 32 k bits. A KAN edge
table holds 4 or 16 entries, each counted here as 8 bits (`table_bits` in the JSON), and there are 197k
edges, so about 6.4 M bits. The 8-bit entry width is my assumption, so treat the ratio as rough, but it is not
close. KANELÉ gets its small designs by pruning hard and using 1-bit inputs; this run did neither by default.

## Pruning sweep (CIFAR-10, with teacher, seed 0)

The threshold ramps up from epoch 3 to epoch 8, leaving two epochs to recover.

| Threshold | First-layer edges left | Class-layer edges left | Table bits | Test acc | Change |
|---|---|---|---|---|---|
| 0 (no pruning) | 196,608 (100%) | 640 | 6.37 M | 48.04 | – |
| 0.1 | 115,391 (59%) | 640 | 3.77 M | 47.20 | -0.84 |
| 0.2 | 37,602 (19%) | 639 | 1.29 M | 43.52 | -4.52 |
| 0.3 | 10,381 (5%) | 622 | 0.41 M | 33.12 | -14.92 |
| 0.5 | 752 (0.4%) | 429 | 0.08 M | 14.73 | -33.31 |

Even the most aggressive point that still works (threshold 0.2, 1.3 M bits) is about 40 times bigger than the
small DWN and 6 points less accurate. One seed only.

## What it suggests

- **The CNN teacher's CIFAR-10 gain is not specific to DWN.** It showed up on a second, quite different LUT
  student, at a similar size (+1.1 vs +1.6), and again did nothing on Fashion-MNIST. That makes "a CNN teacher
  helps small LUT models on harder image data" a slightly stronger statement than the pilot alone.
- **But a KAN student is not the better vehicle for it here.** At this setting it is less accurate and much
  larger than the DWN. For HeLutNet the DWN stays the natural student.
- **The pilot's main caveat still applies.** On DWN, the gain vanished once the model was made wider. This run
  did not test a wider KAN, so the same could happen here.

## Caveats

- This is my CPU re-implementation of the learned object, not KANELÉ's code. It learns the tables directly
  and drops their B-spline smoothness prior, which may be part of why it trails.
- Hyperparameters are a first guess (64 hidden, 2/4 bits, lr 3e-3), and the KD settings were tuned for the
  DWN, not this student.
- Small models, 10 epochs, 3 seeds (1 for pruning). As with the pilot: a reason to test properly, not a
  finding to claim.

## Possible next runs

- A wider or deeper KAN (`--hidden 256` or `128,64`), to see if the teacher's gain survives like it did not
  for the wider DWN.
- KANELÉ's own MNIST setting (`--in_bits 1 --hidden 62`), to check this re-implementation lands near their
  96.3%.
- Tune T and α for this student on the held-out split (`--val`).
