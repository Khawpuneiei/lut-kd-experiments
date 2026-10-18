# First CPU results: small convolutional DWN on CIFAR-10 (16 Oct 2026)

Run on a 4-core CPU in the project's cloud machine. One seed, so treat every number as a first look.

## Setup

- **Teacher:** the pilot's CNN, retrained here: **87.02%** test (15 epochs, 22 min on 4 threads).
- **Student:** small conv DWN, channels 16, 32, 32, depth-3 trees of 2-input LUTs, OR pooling, residual
  init, 1,000-LUT head with random mapping, τ = 2.5. Deployed circuit: 207k gates, 13 levels deep.
- **Training:** 15 epochs, Adam lr 0.01 (× 0.1 after epoch 10), batch 256, seed 0. KD: T = 2, α = 0.9.
  About 7 minutes per epoch on 2 threads, so 1 h 50 min per run (the two runs ran side by side).

## Results

| Student | Test acc, last epoch | Mean of last 3 epochs | Train acc |
|---|---|---|---|
| Flat DWN 8,000 + 8,000, no KD (pilot, 2 seeds) | 52.9% | 52.8% | 68.9% |
| Flat DWN 4,000 + 4,000, KD (pilot, 3 seeds) | 50.9% | – | 59.8% |
| **Small conv DWN, no KD** | **32.7%** | 31.7% | 33.9% |
| **Small conv DWN, logit KD** | **38.5%** | 37.4% | 38.6% |

Per-epoch test accuracy (%):
- no KD: 24.7 27.7 30.1 29.9 33.3 24.7 30.7 29.5 30.9 29.3 32.5 26.1 31.2 31.2 32.7
- KD: 28.9 33.8 32.4 35.8 37.9 35.3 35.7 38.4 36.0 35.5 37.8 37.9 35.7 38.0 38.5

## What it says

- **The conv model underfits badly in this setting.** Train accuracy (34–39%) is barely above test, so it is
  not learning the training set, let alone overfitting. It is 15–20 points below the flat DWN. This does not
  test Petersen's idea fairly yet: their small model reached 60% after hundreds of epochs with a different
  gate relaxation.
- **KD helped a lot more here than on the flat model:** +5.8 points at the last epoch and +5.7 averaged over
  the last three epochs, ahead at every epoch, against +1.6 on the flat model. One seed, but the gap is
  much bigger than the epoch-to-epoch noise. That fits KD helping most when the student struggles to fit.
- Test accuracy without KD jumps around by up to 9 points between epochs; with KD it is steadier.

## Follow-up: learnable head mapping (same model, seed 0)

The head's 1,000 LUTs now learn which 2 feature bits to read, like the flat DWN's first layer.

| Student | Test acc, last epoch | Mean of last 3 epochs | Train acc |
|---|---|---|---|
| Conv DWN, learnable head, no KD | 37.9% | 36.8% | 39.1% |
| Conv DWN, learnable head, logit KD | 40.9% | 41.0% | 42.0% |

Per-epoch test accuracy (%):
- no KD: 27.4 30.5 29.5 30.7 32.5 32.4 30.2 34.2 31.2 34.5 38.8 36.4 34.9 37.5 37.9
- KD: 31.3 36.1 32.9 36.8 34.0 37.3 37.2 37.2 35.1 37.8 43.7 42.6 42.0 40.1 40.9

- The learnable head helps (+5.2 without KD, +2.5 with KD), and KD still helps (+3.0), but the model still
  cannot fit its training set (39–42% train accuracy against 69% for the flat model).
- **Odd sign:** in all four runs, training loss rises again after the learning-rate drop at epoch 10
  (no KD 1.81 → 1.92, KD 3.62 → 3.75), and test accuracy peaks right after the drop and then falls. Training
  loss should not rise at a lower learning rate, so something in the optimisation is unstable.

## What I think is wrong

Every LUT output is rounded to 0 or 1 in the forward pass, and gradients pass through 13 such levels with
DWN's straight-through/EFD approximation. That works for the flat DWN's 2 levels, but errors compound with
depth. Petersen et al. do not train this way: during training each gate outputs a *probability* (a smooth
mix of the 16 possible gates applied to real-valued inputs), and only the final model is rounded to hard
gates. So the next change worth making is a relaxed training mode for the tree levels: each 2-input LUT
outputs the multilinear interpolation of its 4 table entries (in [0, 1] after a sigmoid) instead of a hard
bit, with hard lookups only at test time. More epochs or wider blocks on the current hard-STE version are
unlikely to close a 30-point train-accuracy gap.
