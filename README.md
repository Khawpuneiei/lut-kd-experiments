# LUT-network students with knowledge distillation

Small CPU experiments on one question: **does a CNN teacher help a lookup-table (LUT) network learn?** LUT
networks such as DWN are cheap to run in hardware and under homomorphic encryption (HeLutNet), but they
trail CNNs on accuracy. Distillation only changes training, so any gain it gives is free at inference.

Each experiment has its own folder with code, a README (setup, exact commands, caveats), a RESULTS file, raw
per-run results (`*.jsonl`) and training logs. Every run was on a 4-core CPU, so the models are small and most
comparisons use 1–3 seeds: treat the numbers as first looks, not findings.

| # | Experiment | Folder | Date |
|---|---|---|---|
| 1 | Distilling a CNN into a flat DWN (pilot) | [`dwn-kd-pilot/`](dwn-kd-pilot/) | 4–5 Oct 2026 |
| 2 | KANELÉ-style LUT-KAN student with a CNN teacher | [`kanele-mini-exp/`](kanele-mini-exp/) | 16 Oct 2026 |
| 3 | Convolutional DWN with a CNN teacher | [`conv-dwn-kd/`](conv-dwn-kd/) | 16 Oct 2026 |

---

## 1. Distilling a CNN into a flat DWN (pilot)

**Question:** can logit distillation from a small CNN teacher make a 2-input-LUT DWN more accurate on
Fashion-MNIST and CIFAR-10?

**Result** ([full results](dwn-kd-pilot/RESULTS.md)):
- CIFAR-10, 4,000 + 4,000 LUTs: distillation raised test accuracy from **49.3% to 50.9%** (+1.6 points, 3
  seeds), ahead at every epoch of every seed. Label smoothing alone lowered accuracy, so the gain comes from
  the teacher.
- A model twice as wide (8,000 + 8,000 LUTs) reached **52.9% without a teacher**, and distillation did not
  help it (52.4%). Width was the stronger lever.
- Fashion-MNIST: no gain (−0.2 points).

**Code:** a CPU re-implementation of DWN (`dwn.py`, checked against the official CUDA kernel by
`check_efd.py`), the CNN teacher (`teacher.py`), the student with five training modes (`student.py`) and the
overnight driver (`pipeline.py`). [README](dwn-kd-pilot/README.md)

---

## 2. KANELÉ-style LUT-KAN student with a CNN teacher

**Question:** does the same teacher help a KANELÉ-style student (a KAN whose every edge is a small lookup
table) as much as it helps the DWN, and how much of it can KANELÉ's edge pruning remove?

**Result** ([full results](kanele-mini-exp/RESULTS.md)):
- CIFAR-10: distillation helped by about as much as on the DWN, **46.8% → 47.9%** (+1.1 points, 3 seeds),
  ahead at every epoch of every seed.
- Fashion-MNIST: no gain (−0.2 points), the same as the DWN.
- The KAN student is 2.5 points below the small DWN without a teacher and its tables hold about 200× more
  bits, so the DWN is the better model at this size.
- Pruning: threshold 0.1 removed 41% of first-layer edges for −0.8 points; threshold 0.2 removed 81% for −4.5.

**Code:** `lutkan.py` (the LUT-KAN model and pruning), `train_lutkan.py`, `run_all.sh` (every run in the
results). It reuses the pilot's data loading and teacher from `../dwn-kd-pilot`.
[README](kanele-mini-exp/README.md)

---

## 3. Convolutional DWN with a CNN teacher

**Question:** following Petersen et al.'s convolutional logic gate networks (NeurIPS 2024), does giving the
DWN convolution-style structure (shared trees of 2-input LUTs over 3×3 patches, OR pooling) close the gap
to CNNs on CIFAR-10, and does distillation help it?

**Result so far** ([full results](conv-dwn-kd/results/RESULTS.md), one seed):
- The small conv DWN **underfits badly**: 32.7% without a teacher, against 52.9% for the flat DWN, with train
  accuracy barely above test.
- Distillation helped much more than on the flat model: **32.7% → 38.5%** (+5.8 points).
- A learnable head mapping helped too: **37.9%** without a teacher, **40.9%** with one.
- Likely cause: every LUT output is rounded to a hard bit during training, and the gradient approximation
  compounds over 13 levels. The next step is a relaxed (soft) training mode, as Petersen et al. use.

**Code:** `conv_dwn.py` (the model, with gate count and circuit depth), `student.py`, `teacher.py`,
`check_conv.py` (gradient and consistency checks, including against the pilot's `dwn.py`), `pipeline.py`.
[README](conv-dwn-kd/README.md)

---

## Running

Python 3 with `torch`, `numpy` and `pillow` (`torchvision` optional, for downloading CIFAR-10). Datasets and
cached teacher logits are not in the repo: set `DATA_DIR` and `CACHE_DIR` and follow each experiment's README.
Experiment 2 must sit next to `dwn-kd-pilot/`, as it is here.
