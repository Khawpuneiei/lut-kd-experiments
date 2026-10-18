"""Convolutional DWN: Petersen et al.'s convolutional logic gate trees built from DWN LUT layers.

Same maths as the pilot's dwn.py (torch-dwn 1.1.1: LUT lookup, STE binarisation, Extended Finite
Difference gradient), re-stated here so every op is device-aware and runs on CUDA or CPU.

Pieces:
  - LUTLayer / EFDFunction: as in the pilot, with tensors created on the input's device.
  - ConvLUTTree: one depth-d tree of 2-input LUTs per output channel. Each tree reads 2^d bits
    sampled from a 3x3 patch of 2 randomly chosen input channels, and the same tree is applied at
    every position (unfold -> LUT tree on every patch -> fold back), like a CNN filter.
  - OrPool: 2x2 logical OR (max over 0/1 bits), as Petersen's OR pooling.
  - Residual init: each LUT starts as "pass input A through" (plus noise), so a deep stack behaves like
    the identity at the start. Petersen's biggest ablation factor (-8.8 points on CIFAR-10 without it).
  - make_conv_dwn(): thermometer bit-planes -> [ConvLUTTree -> OrPool] x blocks -> flat LUTLayer -> GroupSum.
"""
import math

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- LUT core (from the pilot)

def efd_coefficients(n, alpha=None, beta=None):
    alpha = 0.5 * 0.75 ** (n - 1) if alpha is None else alpha
    beta = 0.25 / 0.75 if beta is None else beta
    K = 1 << n
    coef = torch.zeros(n, K, K)
    for l in range(n):
        for a in range(K):
            for a2 in range(K):
                dist = bin((a & ~(1 << l)) ^ (a2 & ~(1 << l))).count("1")
                sign = 1.0 if (a2 >> l) & 1 else -1.0
                coef[l, a, a2] = sign * alpha * beta ** dist
    return coef


class EFDFunction(torch.autograd.Function):
    """LUT lookup with STE binarisation forward, EFD backward. identity=True: LUT j reads inputs
    j*n .. j*n+n-1 (no gather), used inside the trees and after a learnable mapping."""

    @staticmethod
    def forward(ctx, x, mapping, luts, coef, identity):
        B = x.shape[0]
        J, n = mapping.shape
        K = 1 << n
        dev = x.device
        xs = (x if identity else x[:, mapping.reshape(-1)]).reshape(B, J, n)
        addr = (xs * (2.0 ** torch.arange(n, device=dev))).sum(-1).long()
        flat_idx = addr + torch.arange(J, device=dev).unsqueeze(0) * K
        out = luts.reshape(-1)[flat_idx]
        ctx.save_for_backward(mapping, luts, coef, flat_idx)
        ctx.in_size, ctx.identity = x.shape[1], identity
        return (out > 0).float()

    @staticmethod
    def backward(ctx, g):
        mapping, luts, coef, flat_idx = ctx.saved_tensors
        J, n = mapping.shape
        K = 1 << n
        g = g.contiguous()
        luts_grad = torch.zeros(J * K, dtype=g.dtype, device=g.device)
        luts_grad.index_add_(0, flat_idx.reshape(-1), g.reshape(-1))
        luts_grad = luts_grad.reshape(J, K)
        x_grad = None
        if ctx.needs_input_grad[0]:
            G = torch.einsum("jk,lak->lja", luts, coef)                 # (n, J, K)
            ws = [G[l].reshape(-1)[flat_idx] * g for l in range(n)]      # each (B, J)
            if ctx.identity:
                x_grad = torch.stack(ws, dim=-1).reshape(g.shape[0], J * n)
            else:
                x_grad = torch.zeros(g.shape[0], ctx.in_size, dtype=g.dtype, device=g.device)
                for l in range(n):
                    x_grad.index_add_(1, mapping[:, l].contiguous(), ws[l])
        return x_grad, None, luts_grad, None, None


class LearnableMappingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, tau):
        mapping = weights.argmax(dim=1)
        ctx.save_for_backward(x, weights)
        ctx.tau = tau
        return x[:, mapping]

    @staticmethod
    def backward(ctx, g):
        x, weights = ctx.saved_tensors
        weights_grad = g.T @ (2 * x - 1)
        x_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = g @ torch.softmax(weights / ctx.tau, dim=1)
        return x_grad, weights_grad, None


def random_mapping(input_size, n, output_size):
    total = n * output_size
    num_complete, offset = divmod(total, input_size)
    parts = [torch.randperm(input_size) for _ in range(num_complete)] + [torch.randperm(input_size)[:offset]]
    return torch.cat(parts).reshape(output_size, n).long()


def residual_tables(num_luts, n, noise=0.1):
    """LUT tables that pass input bit 0 ("A") through: +1 where bit 0 of the address is set, else -1,
    plus uniform noise so the gates are not all identical. Address bit l is input l (see EFDFunction)."""
    K = 1 << n
    a = torch.arange(K)
    base = ((a & 1) * 2 - 1).float()                                    # [-1, +1, -1, +1, ...]
    return base.unsqueeze(0).repeat(num_luts, 1) + noise * (torch.rand(num_luts, K) * 2 - 1)


class LUTLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, n, mapping="random", init="uniform", lm_tau=0.001):
        super().__init__()
        self.n = n
        self.learnable = mapping == "learnable"
        if self.learnable:
            self.lm_weights = torch.nn.Parameter(torch.rand(output_size * n, input_size))
            self.lm_tau = lm_tau
            self.register_buffer("mapping", torch.arange(output_size * n).reshape(output_size, n))
        elif mapping == "identity":
            self.register_buffer("mapping", torch.arange(output_size * n).reshape(output_size, n))
        else:
            self.register_buffer("mapping", random_mapping(input_size, n, output_size))
        self.identity = mapping in ("learnable", "identity")
        if init == "residual":
            self.luts = torch.nn.Parameter(residual_tables(output_size, n))
        else:
            self.luts = torch.nn.Parameter(torch.rand(output_size, 2 ** n) * 2 - 1)
        self.register_buffer("coef", efd_coefficients(n))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.luts.clamp_(-1, 1)
        if self.learnable:
            x = LearnableMappingFunction.apply(x, self.lm_weights, self.lm_tau)
        return EFDFunction.apply(x, self.mapping, self.luts, self.coef, self.identity)


class GroupSum(torch.nn.Module):
    def __init__(self, k, tau):
        super().__init__()
        self.k, self.tau = k, tau

    def forward(self, x):
        return x.reshape(x.shape[0], self.k, -1).sum(-1) / self.tau


# ----------------------------------------------------------------------------- convolutional pieces

class ConvLUTTree(torch.nn.Module):
    """Cout trees, each a complete binary tree of depth `depth` of 2-input LUTs, applied at every position.

    Tree c reads 2**depth bits chosen (fixed at init) from the 3x3 patch of 2 input channels, so the
    deployed circuit has (2**depth - 1) LUTs per tree per position. Trees are shared across positions,
    so the trainable tables are only Cout * (2**depth - 1) * 4 numbers.
    """

    def __init__(self, cin, cout, depth=3, kernel=3, in_channels_per_tree=2, init="residual"):
        super().__init__()
        self.cin, self.cout, self.depth, self.kernel = cin, cout, depth, kernel
        leaves = 1 << depth
        taps = kernel * kernel * in_channels_per_tree
        assert leaves <= taps, "a tree needs at most kernel*kernel*channels leaf bits"
        # Which 2 input channels and which 8 of their 18 patch bits each tree reads. Index into the
        # unfolded patch vector of length cin*kernel*kernel, laid out channel-major as F.unfold does.
        chans = torch.stack([torch.randperm(cin)[:in_channels_per_tree] if cin >= in_channels_per_tree
                             else torch.randint(0, cin, (in_channels_per_tree,)) for _ in range(cout)])  # (cout, 2)
        kk = kernel * kernel
        sel = torch.empty(cout, leaves, dtype=torch.long)
        for c in range(cout):
            tap_idx = torch.cat([chans[c, i] * kk + torch.arange(kk) for i in range(in_channels_per_tree)])
            sel[c] = tap_idx[torch.randperm(taps)[:leaves]]
        self.register_buffer("sel", sel.reshape(-1))                    # (cout * leaves,)
        # Levels of the tree: leaves/2 LUTs per tree, then leaves/4, ..., then 1.
        levels, width = [], leaves
        while width > 1:
            width //= 2
            levels.append(LUTLayer(cout * width * 2, cout * width, n=2, mapping="identity", init=init))
        self.levels = torch.nn.ModuleList(levels)

    def forward(self, x):                                                # x: (B, cin, H, W) 0/1 floats
        B, _, H, W = x.shape
        pad = self.kernel // 2
        patches = F.unfold(x, self.kernel, padding=pad)                   # (B, cin*k*k, H*W)
        patches = patches.transpose(1, 2).reshape(B * H * W, -1)          # (B*H*W, cin*k*k)
        h = patches[:, self.sel]                                         # (B*H*W, cout*leaves)
        for layer in self.levels:                                        # tree level by level
            h = layer(h)                                                 # (B*H*W, cout*width)
        return h.reshape(B, H * W, self.cout).transpose(1, 2).reshape(B, self.cout, H, W)

    def luts_per_position(self):
        return self.cout * ((1 << self.depth) - 1)


class OrPool(torch.nn.Module):
    """2x2 logical OR over 0/1 bits. max == OR on {0,1}, and its gradient goes to the active input."""

    def forward(self, x):
        return F.max_pool2d(x, 2)


class ConvDWN(torch.nn.Module):
    def __init__(self, cin, hw, channels, head_luts, num_classes, tau, depth=3, init="residual",
                 head_mapping="random"):
        super().__init__()
        blocks, c, size = [], cin, hw
        for w in channels:
            blocks += [ConvLUTTree(c, w, depth=depth, init=init), OrPool()]
            c, size = w, size // 2
        self.blocks = torch.nn.ModuleList(blocks)
        self.feat_shape = (c, size, size)
        assert head_luts % num_classes == 0
        self.head = LUTLayer(c * size * size, head_luts, n=2, mapping=head_mapping)
        self.groupsum = GroupSum(num_classes, tau)
        self.hw, self.channels, self.depth = hw, list(channels), depth

    def forward(self, x, return_features=False):
        feats = []
        for m in self.blocks:
            x = m(x)
            if isinstance(m, OrPool):
                feats.append(x)
        logits = self.groupsum(self.head(x.flatten(1)))
        return (logits, feats) if return_features else logits

    def circuit_stats(self):
        """Deployed-circuit size: LUT evaluations per image (trees are re-used at every position), and
        the number of LUT levels an input bit passes through (OR pools count as one 2-input gate level)."""
        luts, size = 0, self.hw
        for m in self.blocks:
            if isinstance(m, ConvLUTTree):
                luts += m.luts_per_position() * size * size
            else:
                size //= 2
        # OR pooling: a 4-input OR is 3 two-input gates, one per output bit of each pooled map.
        or_gates = sum(3 * c * (self.hw // 2 ** (i + 1)) ** 2 for i, c in enumerate(self.channels))
        head = self.head.luts.shape[0]
        levels = len(self.channels) * (self.depth + 1) + 1
        return {"lut_evals": luts, "or_gates": or_gates, "head_luts": head,
                "gates_total": luts + or_gates + head, "depth_levels": levels,
                "trainable_tables": sum(p.numel() for n, p in self.named_parameters() if n.endswith("luts"))}


def make_conv_dwn(cin, hw, channels=(32, 64, 64), head_luts=4000, num_classes=10, tau=10.0,
                  depth=3, init="residual", head_mapping="random"):
    return ConvDWN(cin, hw, channels, head_luts, num_classes, tau, depth, init, head_mapping)


def make_flat_dwn(input_size, n1, n2, num_classes, tau, n=2):
    """The pilot's flat model, for the baseline rows."""
    assert n2 % num_classes == 0
    return torch.nn.Sequential(
        LUTLayer(input_size, n1, n, mapping="learnable"),
        LUTLayer(n1, n2, n, mapping="random"),
        GroupSum(num_classes, tau),
    )


class DistributiveThermometer:
    """Per-pixel-position quantile thresholds (same as the pilot). binarize() returns (N, F, bits)."""

    def __init__(self, num_bits):
        self.num_bits = int(num_bits)
        self.thresholds = None

    def fit(self, x):                                                     # x: (N, F) float
        data = torch.sort(x, dim=0)[0]
        idx = torch.tensor([int(data.shape[0] * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)])
        self.thresholds = data[idx].T.contiguous()                        # (F, bits)
        return self

    def binarize(self, x):
        out = torch.empty(x.shape[0], x.shape[1], self.num_bits, dtype=torch.uint8)
        for s in range(0, x.shape[0], 4096):
            out[s:s + 4096] = x[s:s + 4096].unsqueeze(-1) > self.thresholds
        return out


def to_bitplanes(bits, c, h, w):
    """(N, C*H*W, bits) from the thermometer -> (N, C*bits, H, W) bit-planes for the conv model."""
    n, _, nb = bits.shape
    return bits.reshape(n, c, h, w, nb).permute(0, 1, 4, 2, 3).reshape(n, c * nb, h, w).contiguous()
