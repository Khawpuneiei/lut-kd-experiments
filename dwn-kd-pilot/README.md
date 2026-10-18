# Distilling a CNN into a DWN lookup-table model (pilot)

**Question:** HeLutNet runs encrypted (FHE) inference with DWN lookup-table models, and its
latency depends on the LUT size `n` (8.3 ms for every n = 2 model in its Table 8,
55–83 ms for n = 3). Can **knowledge distillation from a CNN teacher** make an n = 2 DWN more
accurate? Distillation changes only *training*: the deployed LUT model has the same structure
with or without it, so its FHE cost is the same.

Results are in [RESULTS.md](RESULTS.md).

## Files

| File | What it does |
|---|---|
| `dwn.py` | CPU re-implementation of DWN (`torch-dwn` 1.1.1 only ships CUDA kernels): distributive thermometer, LUT layer with Extended Finite Difference gradients, learnable mapping, GroupSum. |
| `check_efd.py` | Compares `dwn.py` with a line-by-line Python port of `torch-dwn`'s CUDA kernel. Output: `results/check_efd_output.txt`. |
| `common.py` | Loads MNIST, Fashion-MNIST and CIFAR-10, and makes the 90/10 tuning split. |
| `teacher.py` | Trains a small CNN teacher on 90% of the training set and caches its logits. |
| `student.py` | Trains the DWN student in one of five modes: `ce`, `ce_ls`, `kd`, `kd_gated`, `kd_conf`. |
| `pipeline.py` | The overnight driver: tuning on the held-out split, then every final run in priority order. |
| `summarize.py` | Prints the tables in RESULTS.md from the raw results. |
| `results/` | Raw JSON for every run (`runs_*.jsonl`, `tune_fmnist.jsonl`, `teacher_*.json`), the driver's log with every command (`pipeline_log.txt`), and per-run training logs (`logs/`). |

## Training modes (same architecture in every mode)

- `ce`: cross-entropy on labels only, which is how HeLutNet's models are trained.
- `ce_ls`: cross-entropy with label smoothing 0.1. This is a control: part of KD's gain is often just softer targets, not anything the teacher knows.
- `kd`: standard distillation, `(1 - a) * CE + a * T^2 * KL(teacher_T || student_T)`.
- `kd_gated`: distil only on training images the teacher classifies correctly, with plain CE elsewhere. This is a simple "trust the teacher only where it is reliable" rule.
- `kd_conf`: a soft version of the gate. Each image's KD weight is the teacher's probability on the true label.

## Reproduce

Needs Python 3 with `torch`, `numpy` and `pillow`. Data goes under `$DATA_DIR` (default `/home/claude/data`):
Fashion-MNIST's four `.gz` files in `fmnist/`, and the fast.ai CIFAR-10 mirror
(`https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz`) unpacked into `cifar10/`.
Teacher logits are cached in `$CACHE_DIR` (default `/home/claude/work/cache`).

```bash
python3 check_efd.py                                     # re-implementation check, a few seconds
python3 teacher.py --dataset fmnist --epochs 12 --threads 2
python3 teacher.py --dataset cifar10 --epochs 15 --threads 2
python3 pipeline.py                                      # tuning, then all 36 final runs (about 6 h on 4 CPU cores)
python3 summarize.py
```

One pair of runs is enough to see the CIFAR-10 effect (about 20 minutes each on 2 CPU threads):

```bash
for m in ce kd; do
  python3 student.py --dataset cifar10 --mode $m --bits 3 --n1 4000 --n2 4000 --n 2 --bs 256 \
    --epochs 10 --tau 10 --T 2 --alpha 0.9 --seed 0 --threads 2 --out results/my_runs.jsonl
done
```

The wider-model check was run by hand after the pipeline (about 30 minutes per run here). The run without
distillation is the 52.9% headline result:

```bash
for m in ce kd; do
  python3 student.py --dataset cifar10 --mode $m --bits 3 --n1 8000 --n2 8000 --n 2 --bs 256 \
    --epochs 10 --tau 20 --T 2 --alpha 0.9 --seed 0 --threads 2 --out results/my_runs.jsonl
done
```

On a GPU, the official package can replace `dwn.py`: build the student as
`torch.nn.Sequential(torch_dwn.LUTLayer(inputs, 4000, n=2, mapping="learnable"), torch_dwn.LUTLayer(4000, 4000, n=2), torch_dwn.GroupSum(10, tau=10))`.

## Check on the re-implementation

The LUT layer's forward pass and both gradients (to the LUT table and to the inputs) were
compared against a line-by-line Python port of `torch-dwn`'s CUDA kernel (`efd_cuda_kernel.cu`)
for n = 2, 3 and 6, and the learnable mapping against `torch-dwn`'s formulas. They match
(`torch.equal` for outputs, `torch.allclose` for gradients).
