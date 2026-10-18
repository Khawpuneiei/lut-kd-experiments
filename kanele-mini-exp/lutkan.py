"""A KANELÉ-style student: a KAN whose edge functions are lookup tables.

KANELÉ (Hoang, Gupta, Harris, FPGA '26) trains a KAN with b-bit quantised activations and
then enumerates every edge's spline over its 2^b possible inputs, giving one table per edge.
Once the input is b-bit, the spline is *only ever read at 2^b points*, so this file skips the
spline and learns those 2^b table entries directly. That is exactly the "logical LUT" KANELÉ
synthesises; what is lost is the spline's smoothness prior, which is noted in README.md.

A layer is: quantise input to b bits -> look up table[i, j, x_i] for every edge (i, j)
-> sum over i -> output j. Implemented as one-hot(x) @ W, with W of shape (in * 2^b, out).

Pruning follows KANELÉ: an edge whose table (minus its mean, which is just a bias) has a small
L2 norm is masked, with the threshold ramped up over training.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """Signed uniform b-bit quantiser with a learnable scale (as in KANELÉ, via Brevitas).
    Returns a *continuous* table position in [0, 2^b - 1]; the layer rounds it. Keeping it
    continuous here is what lets the gradient reach the scale and the previous layer."""

    def __init__(self, bits, init_scale=1.0):
        super().__init__()
        self.levels = 2 ** bits
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, x):
        half = self.levels // 2
        return (x / self.log_scale.exp() + half).clamp(0, self.levels - 1)


class LUTEdgeLayer(nn.Module):
    """KAN layer with a 2^b-entry table on every edge. Input: continuous positions (B, in).

    Forward uses the rounded position (a true table lookup, what the hardware does). Backward
    treats the table as piecewise linear between neighbouring entries, which is the straight-
    through estimator KANELÉ's smooth spline gives for free."""

    def __init__(self, in_features, out_features, bits):
        super().__init__()
        self.in_features, self.out_features, self.levels = in_features, out_features, 2 ** bits
        self.table = nn.Parameter(torch.randn(in_features, self.levels, out_features) * (1.0 / math.sqrt(in_features)))
        self.register_buffer("mask", torch.ones(in_features, out_features))

    def forward(self, pos):
        B, L = pos.shape[0], self.levels
        w = (self.table * self.mask.unsqueeze(1)).reshape(self.in_features * L, self.out_features)
        hard = F.one_hot(pos.round().long(), L).float().reshape(B, -1) @ w
        if not self.training:
            return hard
        lo = pos.floor().clamp(max=L - 2)
        frac = (pos - lo).unsqueeze(-1)
        soft_w = F.one_hot(lo.long(), L).float() * (1 - frac) + F.one_hot(lo.long() + 1, L).float() * frac
        soft = soft_w.reshape(B, -1) @ w
        return hard.detach() + soft - soft.detach()

    def edge_norms(self):
        """L2 norm of each edge's table after removing its mean (KANELÉ's spline-component norm)."""
        centred = self.table - self.table.mean(1, keepdim=True)
        return centred.pow(2).sum(1).sqrt()                                  # (in, out)

    def prune(self, threshold):
        with torch.no_grad():
            self.mask.mul_((self.edge_norms() >= threshold).float())
        return int(self.mask.sum().item())

    def active_edges(self):
        return int(self.mask.sum().item())


class LUTKAN(nn.Module):
    """input quantiser -> LUT edge layer -> quantiser -> ... -> LUT edge layer (class scores)."""

    def __init__(self, in_features, hidden, num_classes, in_bits, act_bits):
        super().__init__()
        widths = [in_features] + list(hidden) + [num_classes]
        self.quant = nn.ModuleList([Quantizer(in_bits)] + [Quantizer(act_bits, init_scale=4.0) for _ in hidden])
        self.layers = nn.ModuleList([LUTEdgeLayer(a, b, in_bits if i == 0 else act_bits)
                                     for i, (a, b) in enumerate(zip(widths[:-1], widths[1:]))])

    def forward(self, x):
        for q, layer in zip(self.quant, self.layers):
            x = layer(q(x))
        return x

    def prune(self, threshold):
        return [layer.prune(threshold) for layer in self.layers]

    def active_edges(self):
        return [layer.active_edges() for layer in self.layers]

    def table_bits(self, out_bits=8):
        """Rough size of the synthesised design: entries per surviving edge * bits per entry."""
        return sum(layer.active_edges() * layer.levels * out_bits for layer in self.layers)


def prune_threshold(epoch, target, t0, tf):
    """KANELÉ's ramp: 0 before t0, rising exponentially to 95% of `target` at tf, then `target`."""
    if epoch < t0:
        return 0.0
    if epoch >= tf:
        return target
    frac = (epoch - t0 + 1) / (tf - t0 + 1)
    return target * 0.95 * (math.exp(3 * frac) - 1) / (math.exp(3) - 1)
