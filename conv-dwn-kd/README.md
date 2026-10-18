# Mini experiment: distilling a CNN into a *convolutional* DWN

Follow-up to the [flat-DWN pilot](../dwn-kd-pilot/README.md), using the idea from Petersen et al.,
*Convolutional Differentiable Logic Gate Networks* (NeurIPS 2024; notes in
[neurips24-conv-logic-gate-networks.md](../neurips24-conv-logic-gate-networks.md)).

## Question

The pilot showed that distilling a CNN into a flat (fully connected, randomly wired) 2-input-LUT DWN gives
about +1.6 points on CIFAR-10, and that simply making the flat model wider gives more. Petersen et al. show
that giving a 2-input-gate network *convolutional structure* is what lifts it from about 60% to 86% on
CIFAR-10, and they used a neural-network teacher for their big models without ever measuring what it added.

So the mini experiment asks two things, in order:

1. **Does a small convolutional DWN beat the widest flat DWN on CIFAR-10 at the same LUT size (n = 2)?**
   Petersen's smallest model (0.4 M gates, no teacher) reached 60.4%; the pilot's wide flat model reached 52.9%.
2. **Once the student is convolutional, how much does a CNN teacher add, and which kind of distillation?**
   Logit KD (as in the pilot and in Petersen), logit KD with the √2-higher softmax temperature Petersen
   recommends, and feature-map KD, which only a convolutional student can do.

Everything stays at n = 2, because HeLutNet's encrypted cost is the same for every n = 2 model it tested
(8.3 ms). The open hardware question, whether a model with more gate *layers* stays cheap under FHE, is for
P' Palm; the experiment records depth and gate count for every model so he can answer it.

## The student: a convolutional DWN built from the pilot's LUT layer

Plain-language version of Petersen's design, mapped onto DWN pieces:

| Petersen et al. | What it means | DWN version here |
|---|---|---|
| Logic gate tree as a conv kernel | A "filter" is a tiny tree of 2-input gates: 4 gates read 8 input bits, 2 gates combine those, 1 gate gives the output bit. Same tree reused at every image position, like a CNN filter. | A depth-3 tree of 7 two-input LUTs (n = 2). Build it from `LUTLayer` by treating every image patch as its own batch row (`unfold` the image into 3×3 patches, apply the LUT tree, fold back). |
| 2 input channels per tree | Each tree reads bits from only 2 of the input channels, so wiring stays simple. | Random choice of 2 input channels per output channel, fixed at init (same as DWN's random mapping). |
| OR pooling | Max-pooling with a logical OR, so the model stays pure logic. | `max` over each 2×2 window of 0/1 bits. It is exactly OR. |
| Residual initialisation | Every gate starts as "pass input A through", so a deep stack of gates behaves like the identity at first and gradients survive. Biggest single factor in their ablation (−8.8 points without it). | Initialise each LUT table so entries where input-bit A = 1 are +1 and where A = 0 are −1, plus small noise. (DWN tables are real-valued in [−1, 1] during training, so this is one line.) |
| Input encoding | 2-bit inputs, 3 thresholds, for the small models. | The pilot's 3-bit distributive thermometer, kept per channel so the result is 9 bit-planes of 32×32. |
| GroupSum head | Popcount per class, divided by a temperature τ. | The pilot's `GroupSum`, preceded by one flat `LUTLayer` with random mapping. |

**Size for a CPU run** (one block smaller than Petersen's model S, so one run fits in about an hour on 2–4 threads):

```
input   9 × 32 × 32 bits (3-bit thermometer on RGB)
block 1 conv LUT tree, 32 channels, 3×3 patch → 32 × 32 × 32   then OR-pool → 32 × 16 × 16
block 2 conv LUT tree, 64 channels, 3×3 patch → 64 × 16 × 16   then OR-pool → 64 × 8 × 8
block 3 conv LUT tree, 64 channels, 3×3 patch → 64 × 8 × 8     then OR-pool → 64 × 4 × 4   (= 1,024 bits)
head    LUTLayer 1,024 → 4,000 (random mapping) → GroupSum(10, τ)
```

LUT count: 7 × (32 + 64 + 64) = 1,120 *distinct* trees' worth of LUTs, but they are evaluated at every
position, so the deployed circuit has about 7 × (32·1024 + 64·256 + 64·64) ≈ 0.36 M LUT evaluations plus the
4,000-LUT head. That is close to Petersen's model S (0.40 M gates), which is the number to quote to P' Palm.
Gate *depth* is 3 trees × 3 levels + 3 OR gates + 1 head layer = 13 levels, against 2 in HeLutNet's models.

If this is too slow on CPU, drop block 3 and pool block 2 to 4×4; if it is too fast, add a 4th block (Petersen's
S has 4 blocks with 32 channels each).

## Teacher

Reuse the pilot's CIFAR-10 CNN (86.99% test, logits already cached by `teacher.py`). For feature KD the
teacher's intermediate feature maps are needed too, so `teacher.py` gets a `--save-features` option that
also caches the output of its second conv block (after pooling) for every training image.

## Runs

All on CIFAR-10, 3 seeds each, test accuracy at the last epoch plus mean of the last 3 epochs (the pilot's
protocol). Same optimiser as the pilot (Adam, lr 0.01, batch 256, lr × 0.1 after 2/3 of the epochs), 15 epochs.

| # | Student | Training | Why |
|---|---|---|---|
| 1 | Flat DWN 8,000 + 8,000 (pilot's wide model) | `ce` | Baseline, already run once: 52.9%. Rerun with the new epoch count. |
| 2 | Conv DWN | `ce` | **Main question 1.** Does structure beat width at the same n? Target: above 52.9%. |
| 3 | Conv DWN, Gaussian init instead of residual | `ce` | Checks the biggest ablation in Petersen transfers to DWN tables. One seed is enough. |
| 4 | Conv DWN | `kd` (T = 2, α = 0.9, τ unchanged) | Logit KD exactly as the pilot did it. |
| 5 | Conv DWN | `kd` with τ × √2 | Petersen's temperature rule. Cheap, and the pilot never tried it. |
| 6 | Conv DWN | `kd_feat`: logit KD + hint loss on block-2 features | **Main question 2.** Only possible with a convolutional student. |

`kd_feat`: the student's block-2 output is 64 × 8 × 8 bits. Average-pool it to 64 × 4 × 4, pass through a
learnable 1×1 projection to the teacher's channel count, and add `β · MSE(projection, teacher block-2
features pooled to 4 × 4)` to the KD loss (FitNets-style hint). β from {0.1, 1} on the held-out 10% split,
as the pilot tuned T and α. The projection is training-only and does not exist in the deployed model.

Priority order if time runs out: 2, 4, 1, 5, 6, 3.

## What would count as a result

- Run 2 above 52.9% (3 seeds): structure beats width for n = 2 DWNs, same conclusion as Petersen but now in
  the DWN/HeLutNet setting. Worth telling P' Palm on its own.
- Run 4 or 5 above run 2 by more than the seed spread (about ±0.6 in the pilot): the teacher helps the
  convolutional student, which Petersen asserted but never measured.
- Run 6 above run 4: feature-level distillation adds something that logit KD cannot, which is the novel bit.
- Any of these *not* happening is also a finding; write it down the same way.

## Files

| File | What it does |
|---|---|
| `README.md` | Plan (this file). |
| `conv_dwn.py` | The model. `LUTLayer`/`EFDFunction` (the pilot's DWN maths, made device-aware for CUDA), `ConvLUTTree` (3×3 patch, depth-3 tree of n = 2 LUTs, 2 input channels per tree, shared across positions), `OrPool`, residual init, `ConvDWN` with `circuit_stats()` (gate count and depth for P' Palm), `make_flat_dwn()` for the baseline, thermometer and bit-plane helpers. |
| `check_conv.py` | Sanity checks, seconds on CPU, no data: conv tree equals a brute-force per-patch evaluation; constant input gives constant output and shifting the input shifts the output; residual init passes leaf bit A through; every LUT table gets gradient (CPU and CUDA); `EFDFunction` matches the pilot's `dwn.py` bit-for-bit. |
| `student.py` | Training. `--arch flat\|conv`, `--init residual\|gaussian`, `--mode ce\|kd\|kd_feat`, `--tau-scale`, `--device auto`. One JSON row per run with the config, the test curve, final/last-3/train accuracy, gates and depth. |
| `teacher.py` | The pilot's CNN teacher (86.99% on CIFAR-10) with `--save-features`, which also caches block-2 feature maps pooled to 4×4 for `kd_feat`. |
| `common.py` | Data loading (copied from the pilot; also falls back to a torchvision download if the fast.ai mirror is not there). |
| `pipeline.py` | Runs the table above in priority order, one job at a time, resumable. `SEEDS=0 python3 pipeline.py` for a first look. |
| `summarize.py` | Prints the results table and per-epoch curves from `results/runs.jsonl`. |
| `results/` | Raw JSONL, per-run logs, pipeline log. Write RESULTS.md here once there is something to report. |

## Run it

Needs `torch`, `numpy`, `pillow` (or `torchvision` for the data download). Data under `$DATA_DIR`
(default `~/data`) in the pilot's layout, or let `common.py` download CIFAR-10 with torchvision; teacher
outputs are cached in `$CACHE_DIR` (default `~/work/cache`). Everything picks CUDA automatically.

```bash
cd outreach/conv-dwn-kd
python3 check_conv.py                                                   # seconds; should end with ALL OK
python3 teacher.py --dataset cifar10 --epochs 15 --save-features        # a few minutes on a GPU
python3 student.py --arch conv --mode ce --seed 0                       # run 2, one seed
python3 student.py --arch conv --mode kd --seed 0                       # run 4
python3 student.py --arch conv --mode kd --tau-scale 1.41421356 --seed 0  # run 5
python3 student.py --arch conv --mode kd_feat --seed 0                  # run 6
python3 student.py --arch flat --mode ce --seed 0                       # run 1 (pilot's wide model)
SEEDS=0 python3 pipeline.py                                             # or: everything for one seed
python3 pipeline.py                                                     # all 16 runs, 3 seeds
python3 summarize.py
```

Default τ is `LUTs per class / 40` (so 10 for the 4,000-LUT head and 20 for the 8,000-LUT flat model, the
pilot's values); `--tau` overrides it. If the GPU runs out of memory, lower `--bs`; the conv model's
memory goes with batch × positions × channels because every patch is a row during the tree lookup.

## State of the code (16 Oct)

- `check_conv.py` passes: the conv tree equals a brute-force per-patch evaluation, residual init passes
  bit A through, every table gets gradient, and the LUT maths matches the pilot's `dwn.py` exactly.
- Teacher, all five student conditions, `pipeline.py`'s job list and `summarize.py` were run end to end
  on a small synthetic stand-in for CIFAR-10 (no real data or GPU on the machine that wrote this), so the
  plumbing works. On that toy task the conv model learned, but slowly (loss falling every epoch, 22% after
  6 epochs where the flat model hit 47% in one), which is expected with 13 levels of straight-through
  gradient but means the first real run should be watched. If `conv ce` is not clearly above the flat
  baseline by epoch 15, the knobs to try first, in order: more epochs (Petersen trains for hundreds),
  `--head-mapping learnable`, `--lr 0.02`, and a wider first block (`--channels 64,64,64`).
- Not yet done: a real CIFAR-10 run. The ~1 h per run estimate in the plan is a guess for CPU; on a GPU
  each run should take minutes.

## Caveats to keep in mind

- CPU-only and short training. Petersen trains for hundreds of epochs on a GPU with fused kernels; this is
  15 epochs with `unfold` in plain PyTorch. Expect numbers well below the paper's, and compare only within
  this table.
- DWN's EFD gradient and Petersen's softmax-over-16-gates relaxation are different ways to train a 2-input
  LUT. Which trains better in a deep tree is itself unknown; if the conv DWN does not train at all, that is
  the first thing to suspect, and the Gaussian-vs-residual run will show whether it is an initialisation problem.
- Whether a 13-level gate circuit runs in HeLutNet's FHE budget is not tested here. Record depth and gate
  count and ask P' Palm.
