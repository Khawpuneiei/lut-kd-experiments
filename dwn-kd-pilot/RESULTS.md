# Pilot results: distilling a CNN into a DWN lookup-table model

Run overnight on 4–5 October 2026 on a 4-core CPU, plus a wider-model check on the evening of 5 October. The
code and every raw result are in this folder, and [README.md](README.md) says how to rerun it. The tables are
printed by `summarize.py`.

## Short answer

- **CIFAR-10, 4,000 + 4,000 LUTs: distillation helped.** A small CNN teacher raised a 2-input-LUT DWN's test
  accuracy from **49.3% to 50.9%** (+1.6 points, mean of 3 seeds; +1.7 averaged over the last three epochs).
  The distilled model was ahead at every epoch of every seed.
- **But a model twice as wide did better without a teacher, and distillation did not help it.** With
  8,000 + 8,000 LUTs, the size of HeLutNet's own Fashion-MNIST model, plain training reached **52.9%** and
  distillation 52.4% (mean of 2 seeds; distillation was behind on both). HeLutNet's Table 8 runs every n = 2
  model in 8.3 ms, that one included, so width should cost little or nothing under FHE. On this evidence width
  is the stronger lever, and distillation's gain did not survive the bigger baseline.
- **Distillation should cost nothing at inference.** The trained model has the same structure with or
  without it, so its encrypted-inference cost in HeLutNet should be the same.
- **On the small model, the gain comes from the teacher, not just softer targets.** Label smoothing softens
  the targets without a teacher, and it lowered CIFAR-10 accuracy by 1.1 points.
- **Fashion-MNIST: no gain** (−0.2 points over 3 seeds, and −0.4 on HeLutNet's exact model). Distillation was
  ahead only in the first few epochs.
- **Longer training:** with twice the training (one seed), the small model's gain narrowed but held: +1.6 at
  the last epoch and +1.1 over the last three epochs.
- **It is a small pilot.** Small models, short training, my own CPU re-implementation of DWN, and a baseline
  8 points below the DWN paper's own CIFAR-10 result. It is a reason to test ideas properly, not a finding
  to claim.

## Related work

- **DWN** (Bacellar et al., ICML 2024, Table 1) reports **57.51%** on CIFAR-10 with n = 2. The baselines here
  are smaller models trained for 10 epochs, so they sit below the published one.
- **Convolutional differentiable logic gate networks** (Petersen et al., NeurIPS 2024) reach **86.29%** on
  CIFAR-10 with 2-input logic gates, which are 2-input LUTs. Their Appendix A.2 says: "For the CIFAR-10 B, L,
  and G models, we use a neural network teacher, supervising the class scores." So teacher supervision for
  LUT-style networks is not new. A quick web search found no paper on distillation for DWNs.

## Setup

| | |
|---|---|
| Data | Fashion-MNIST (60k train, 10k test) and CIFAR-10 (50k, 10k). Each pixel value is encoded with a 3-bit distributive thermometer (per-pixel quantile thresholds): 2,352 input bits for Fashion-MNIST, 9,216 for CIFAR-10. No augmentation for the student. |
| Student | The DWN architecture HeLutNet uses: 4,000 LUTs with 2 inputs each and a learnable input mapping, then 4,000 more with random mapping, then a popcount over 400 LUTs per class, divided by τ = 10. The wider check uses 8,000 + 8,000 LUTs and τ = 20, like HeLutNet's Fashion-MNIST model (which also uses 7 bits). |
| Training | Adam, learning rate 0.01, batch 256, learning rate × 0.1 after two thirds of the epochs (after epoch 7 of 10). 12 epochs on Fashion-MNIST and 10 on CIFAR-10 (about 10 and 18 minutes per run on 2 CPU threads, 30 for the wider CIFAR-10 model). Every mode is identical except for the loss. |
| Teachers | Small CNNs trained on 90% of the training set with random crops and flips: 93.27% test on Fashion-MNIST (0.87 M parameters) and 86.99% on CIFAR-10 (0.29 M). |
| Distillation | Hinton KD, `(1 − α)·CE + α·T²·KL`, with T = 2 and α = 0.9. |
| Tuning | On the 10% of Fashion-MNIST's training set that the teacher never saw, with 5-epoch runs: τ from {3.33, 10, 30, 100} gave 10, and (T, α) from {2, 4} × {0.5, 0.9} gave (2, 0.9). CIFAR-10 and the wider model reuse these without their own tuning. |
| Metric | Test accuracy after the last epoch, fixed in advance. The mean over the last three epochs was added after seeing that test accuracy moves about half a point between epochs. |
| Seeds | Each seed changes the student's initialisation, LUT wiring and batch order. The teacher stays fixed. |

## Results

### CIFAR-10 (2-input LUTs, 4,000 + 4,000, 10 epochs)

| Training | Test accuracy % | Gain over no KD, same seeds (points) | Same, last-3-epoch mean | Train accuracy % | Seeds |
|---|---|---|---|---|---|
| No KD (how HeLutNet trains) | 49.28 ± 0.61 | – | – | 60.91 ± 0.59 | 0, 1, 2 |
| Label smoothing 0.1 (control) | 48.22 ± 0.54 | -1.07 (-0.31, -0.89, -2.00) | -0.96 (-0.93, -0.88, -1.08) | 59.43 ± 0.69 | 0, 1, 2 |
| Standard KD | 50.92 ± 0.19 | +1.63 (+1.91, +2.24, +0.75) | +1.71 (+1.75, +1.77, +1.60) | 59.77 ± 0.20 | 0, 1, 2 |
| KD only where teacher is right | 51.01 ± 0.17 | +1.72 (+1.99, +2.32, +0.86) | +1.68 (+1.87, +1.87, +1.30) | 60.35 ± 0.11 | 0, 1, 2 |
| KD weighted by teacher confidence | 51.07 ± 0.22 | +1.78 (+1.67, +2.50, +1.18) | +1.76 (+1.83, +1.77, +1.67) | 60.62 ± 0.40 | 0, 1, 2 |

### Twice as wide (CIFAR-10, 2-input LUTs, 10 epochs, seeds 0 and 1)

| Model and training | Seed 0 | Seed 1 | Mean | Mean of last 3 epochs | Train accuracy |
|---|---|---|---|---|---|
| 4,000 + 4,000, no distillation | 49.15 | 48.75 | 48.95 | 49.20 | 60.80 |
| 4,000 + 4,000, distillation | 51.06 | 50.99 | 51.03 | 50.96 | 59.85 |
| 8,000 + 8,000, no distillation | 53.02 | 52.75 | 52.89 | 52.80 | 68.88 |
| 8,000 + 8,000, distillation | 52.27 | 52.49 | 52.38 | 52.34 | 62.48 |

On the wider model, distillation minus plain training per epoch was +1.8, +1.5, +1.0, +1.6, +1.2, +0.1, −0.1,
−0.5, −0.9, −0.7 on seed 0 and +2.0, +0.8, +1.9, +1.2, +1.0, +1.2, +1.0, 0.0, −0.3, −0.3 on seed 1: ahead
early on both seeds, behind after the learning-rate drop. τ was doubled with the LUT count so the class scores
keep the same range; T and α stay at the values tuned for the small model, and a lower α might suit the wider
one better. The wider model without distillation was still improving near the last epoch.

### Fashion-MNIST (2-input LUTs, 4,000 + 4,000, 12 epochs)

| Training | Test accuracy % | Gain over no KD, same seeds (points) | Same, last-3-epoch mean | Train accuracy % | Seeds |
|---|---|---|---|---|---|
| No KD (how HeLutNet trains) | 88.19 ± 0.08 | – | – | 92.93 ± 0.16 | 0, 1, 2 |
| Label smoothing 0.1 (control) | 87.87 ± 0.16 | -0.33 (-0.50, -0.29, -0.19) | -0.29 (-0.10, -0.43, -0.34) | 91.86 ± 0.26 | 0, 1, 2 |
| Standard KD | 88.00 ± 0.08 | -0.19 (-0.20, -0.34, -0.04) | -0.03 (+0.18, -0.20, -0.06) | 90.82 ± 0.04 | 0, 1, 2 |
| KD only where teacher is right | 87.98 ± 0.17 | -0.21 (-0.06, -0.47, -0.11) | -0.11 (+0.15, -0.30, -0.17) | 91.11 ± 0.07 | 0, 1, 2 |
| KD weighted by teacher confidence | 88.05 ± 0.40 | -0.15 (-0.55, -0.21, +0.32) | +0.06 (+0.09, -0.06, +0.17) | 91.26 ± 0.18 | 0, 1, 2 |

### Bigger LUTs instead of distillation (CIFAR-10, 4,000 + 4,000, no distillation)

| Model | Seed 0 | Seed 1 | Mean | Train accuracy | FHE latency in HeLutNet's Table 8 |
|---|---|---|---|---|---|
| 2-input LUTs, no distillation | 49.15 | 48.75 | 48.95 | 60.8 | 8.3 ms |
| 2-input LUTs, distillation | 51.06 | 50.99 | 51.03 | 59.9 | 8.3 ms |
| 3-input LUTs, no distillation | 51.14 | 49.85 | 50.50 | 66.6 | 55–83 ms |

### Twice the training (CIFAR-10, 4,000 + 4,000, 20 epochs, seed 0)

| Training | Last epoch | Mean of last 3 epochs | Train accuracy |
|---|---|---|---|
| No distillation, 10 epochs | 49.15 | 49.21 | 61.35 |
| No distillation, 20 epochs | 49.78 | 50.08 | 63.76 |
| Distillation, 10 epochs | 51.06 | 50.95 | 60.00 |
| Distillation, 20 epochs | 51.37 | 51.22 | 61.16 |

### Sanity check: HeLutNet's exact Fashion-MNIST model

The same code trained HeLutNet's Fashion-MNIST model (7-bit thermometer, 8,000 + 8,000 LUTs, n = 2, τ = 20)
to **89.20%** after 12 epochs, against **89.75%** in HeLutNet's Table 8. With distillation it reached 88.83%.
One seed each.

## What it suggests

- **Model size is the first limit on CIFAR-10.** Doubling the LUTs raised test accuracy by 3.9 points (2
  seeds), against 1.6 for distillation and 1.6 for 3-input LUTs. Bigger LUTs raised training accuracy almost
  as much as width (6 points against 8) but test accuracy far less, and they cost 7–10× more under FHE
  while width should cost nothing until the layers outgrow the ciphertext slots. So the fair baseline for any
  other idea is the widest n = 2 model that fits.
- **Standard distillation lowered training accuracy in every setting, by 1 to 6 points, but raised test
  accuracy in only one**, the small CIFAR-10 model. That fits distillation acting as a regulariser that helps a small
  model and holds a bigger one back, but this pilot does not test it, and other settings (a lower α, a
  stronger teacher, longer training) might change the picture.
- The two "trust the teacher only where it is reliable" variants did about as well as plain distillation on
  CIFAR-10 (+1.7 and +1.8 points against +1.6, within the noise). The teacher is right on 91% of CIFAR-10
  training images, so these rules change little here. They might matter with a weaker or noisier teacher,
  which is where your teacher-reliability work connects, but this pilot shows no advantage for them.

## Caveats

- The models are small and briefly trained, and the baseline is 8 points below the published DWN result.
  Gains from distillation often shrink on stronger baselines, and the wider-model check already shows that.
- Three seeds for the main comparison, two for the 3-input LUTs and the wider model, and one for the
  longer-training and exact-model checks. Epoch-to-epoch noise is about half a point.
- This is my CPU re-implementation, not the official `torch-dwn` CUDA package or HeLutNet's code. It matches
  the package's kernel exactly on test inputs, and HeLutNet's own Fashion-MNIST model trained with it lands
  within 0.6 points of the published accuracy.
- Distillation settings were tuned on the small Fashion-MNIST model only; CIFAR-10 and the wider model may do
  better with their own tuning.
- FHE latency was not measured. "Same cost" rests on HeLutNet's Table 8 and on the model's structure.

## Next steps

- Find the widest n = 2 model that fits HeLutNet's ciphertext slots, train it properly (longer, with the
  official GPU package), and use it as the baseline for every other idea.
- Reproduce the DWN paper's CIFAR-10 model (57.5%) with the official GPU package.
- Retest distillation on that baseline with a lower α and a stronger teacher, and drop it if it still does
  not help.
- Run the models through HeLutNet's FHE pipeline to confirm the latency.
- Try convolution-style structure as in Petersen et al., which is the bigger lever for CIFAR-10, and check
  whether it fits HeLutNet's two-layer, no-bootstrapping budget.
