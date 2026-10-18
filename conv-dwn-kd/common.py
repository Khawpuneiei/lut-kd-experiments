"""Dataset loading, copied from the pilot so this folder runs on its own. Same $DATA_DIR / $CACHE_DIR
layout: CIFAR-10 as the fast.ai PNG mirror under $DATA_DIR/cifar10/, cached to $CACHE_DIR/cifar10.npz."""
import gzip
import os

import numpy as np
import torch

DATA = os.environ.get("DATA_DIR", os.path.expanduser("~/data"))
CACHE = os.environ.get("CACHE_DIR", os.path.expanduser("~/work/cache"))
os.makedirs(CACHE, exist_ok=True)


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _cifar10():
    cache = os.path.join(CACHE, "cifar10.npz")
    if not os.path.exists(cache):
        root = os.path.join(DATA, "cifar10", "cifar10")
        if os.path.isdir(root):                      # fast.ai PNG mirror
            from PIL import Image
            classes = sorted(os.listdir(os.path.join(root, "train")))
            out = {}
            for split in ("train", "test"):
                xs, ys = [], []
                for c, name in enumerate(classes):
                    folder = os.path.join(root, split, name)
                    for fn in sorted(os.listdir(folder)):
                        xs.append(np.asarray(Image.open(os.path.join(folder, fn)).convert("RGB")))
                        ys.append(c)
                out[f"x_{split}"] = np.stack(xs).transpose(0, 3, 1, 2)
                out[f"y_{split}"] = np.array(ys, dtype=np.int64)
        else:                                         # torchvision download (GPU machines usually have it)
            import torchvision
            out = {}
            for split, train in (("train", True), ("test", False)):
                ds = torchvision.datasets.CIFAR10(os.path.join(DATA, "cifar10_tv"), train=train, download=True)
                out[f"x_{split}"] = np.asarray(ds.data).transpose(0, 3, 1, 2)
                out[f"y_{split}"] = np.array(ds.targets, dtype=np.int64)
        np.savez(cache, **out)
    d = np.load(cache)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def load(name):
    """uint8 images (N, C, H, W) and int64 labels for train and test."""
    if name in ("mnist", "fmnist"):
        d = os.path.join(DATA, name)
        xtr = _idx(os.path.join(d, "train-images-idx3-ubyte.gz"))[:, None]
        ytr = _idx(os.path.join(d, "train-labels-idx1-ubyte.gz")).astype(np.int64)
        xte = _idx(os.path.join(d, "t10k-images-idx3-ubyte.gz"))[:, None]
        yte = _idx(os.path.join(d, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    elif name == "cifar10":
        xtr, ytr, xte, yte = _cifar10()
    else:
        raise ValueError(name)
    return (torch.from_numpy(np.ascontiguousarray(xtr)), torch.from_numpy(ytr),
            torch.from_numpy(np.ascontiguousarray(xte)), torch.from_numpy(yte))


def split(name, val):
    """With val=True, the last 10% of the training set (fixed permutation) is held out for tuning."""
    xtr, ytr, xte, yte = load(name)
    if val:
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ytr), generator=g)
        k = len(ytr) // 10
        return xtr[perm[k:]], ytr[perm[k:]], xtr[perm[:k]], ytr[perm[:k]], perm[k:]
    return xtr, ytr, xte, yte, torch.arange(len(ytr))
