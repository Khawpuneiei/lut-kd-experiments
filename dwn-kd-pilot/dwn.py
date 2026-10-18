"""CPU re-implementation of Differentiable Weightless Networks (DWN).

Mirrors torch-dwn 1.1.1 (Bacellar et al., ICML 2024; the training code HeLutNet adopts):
  - DistributiveThermometer: per-feature quantile thresholds.
  - LUTLayer: each LUT reads n binary inputs and outputs the addressed table entry,
    binarised with a straight-through estimator (STE). LUT entries are real-valued,
    clamped to [-1, 1] during training.
  - Extended Finite Difference (EFD) gradient for the inputs, with the package's
    defaults alpha = 0.5 * 0.75**(n-1), beta = 0.25 / 0.75 (exponential decay).
  - LearnableMapping for the first layer (argmax forward, softmax-weighted backward).
  - GroupSum (popcount per class) / tau as the logits.

torch-dwn only ships CUDA kernels, so this file re-implements the same maths with
vectorised PyTorch ops so it runs on a CPU-only machine.
"""
import torch


class DistributiveThermometer:
    def __init__(self, num_bits):
        self.num_bits = int(num_bits)
        self.thresholds = None

    def fit(self, x):
        data = torch.sort(x, dim=0)[0]
        idx = torch.tensor([int(data.shape[0] * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)])
        self.thresholds = data[idx].T.contiguous()  # (features, bits)
        return self

    def binarize(self, x):
        out = torch.empty(x.shape[0], x.shape[1] * self.num_bits, dtype=torch.uint8)
        for s in range(0, x.shape[0], 4096):  # chunked to keep memory low
            out[s:s + 4096] = (x[s:s + 4096].unsqueeze(-1) > self.thresholds).flatten(1)
        return out


def efd_coefficients(n, alpha=None, beta=None):
    """coef[l, a, a2] so that dOut/dx_l at address a = sum_a2 luts[a2] * coef[l, a, a2]."""
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
    """LUT lookup + STE binarisation, with the EFD backward of torch-dwn's CUDA kernel.

    Inputs are exactly 0/1 (thermometer bits or binarised LUT outputs), so the address is
    computed with float arithmetic, which is equivalent to the kernel's `input > 0` bits.
    `identity=True` means LUT j reads inputs j*n .. j*n+n-1 (used after a learnable mapping),
    which skips a gather and a scatter.
    """

    @staticmethod
    def forward(ctx, x, mapping, luts, coef, identity):
        B = x.shape[0]
        J, n = mapping.shape
        K = 1 << n
        xs = (x if identity else x[:, mapping.reshape(-1)]).reshape(B, J, n)
        addr = (xs * (2.0 ** torch.arange(n))).sum(-1).long()        # (B, J)
        flat_idx = addr + torch.arange(J).unsqueeze(0) * K            # (B, J)
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
        luts_grad = torch.zeros(J * K, dtype=g.dtype).index_add_(0, flat_idx.reshape(-1), g.reshape(-1)).reshape(J, K)
        x_grad = None
        if ctx.needs_input_grad[0]:
            G = torch.einsum("jk,lak->lja", luts, coef)               # (n, J, K)
            ws = [G[l].reshape(-1)[flat_idx] * g for l in range(n)]    # each (B, J)
            if ctx.identity:
                x_grad = torch.stack(ws, dim=-1).reshape(g.shape[0], J * n)
            else:
                x_grad = torch.zeros(g.shape[0], ctx.in_size, dtype=g.dtype)
                for l in range(n):
                    x_grad.index_add_(1, mapping[:, l].contiguous(), ws[l])
        return x_grad, None, luts_grad, None, None


class LearnableMappingFunction(torch.autograd.Function):
    """torch-dwn's learnable mapping. Weights are stored transposed, (outputs, inputs), so the
    per-step argmax runs over contiguous memory; the maths is unchanged."""

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


class LUTLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, n, mapping="random", lm_tau=0.001):
        super().__init__()
        self.n = n
        self.learnable = mapping == "learnable"
        if self.learnable:
            self.lm_weights = torch.nn.Parameter(torch.rand(output_size * n, input_size))
            self.lm_tau = lm_tau
            self.register_buffer("mapping", torch.arange(output_size * n).reshape(output_size, n))
        else:
            self.register_buffer("mapping", random_mapping(input_size, n, output_size))
        self.luts = torch.nn.Parameter(torch.rand(output_size, 2 ** n) * 2 - 1)
        self.register_buffer("coef", efd_coefficients(n))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.luts.clamp_(-1, 1)
        if self.learnable:
            x = LearnableMappingFunction.apply(x, self.lm_weights, self.lm_tau)
        return EFDFunction.apply(x, self.mapping, self.luts, self.coef, self.learnable)


def random_mapping(input_size, n, output_size):
    """Same scheme as torch-dwn: concatenated random permutations of the inputs."""
    total = n * output_size
    num_complete, offset = divmod(total, input_size)
    parts = [torch.randperm(input_size) for _ in range(num_complete)] + [torch.randperm(input_size)[:offset]]
    return torch.cat(parts).reshape(output_size, n).long()


class GroupSum(torch.nn.Module):
    def __init__(self, k, tau):
        super().__init__()
        self.k, self.tau = k, tau

    def forward(self, x):
        return x.reshape(x.shape[0], self.k, -1).sum(-1) / self.tau


def make_dwn(input_size, n1, n2, n, num_classes, tau):
    assert n2 % num_classes == 0
    return torch.nn.Sequential(
        LUTLayer(input_size, n1, n, mapping="learnable"),
        LUTLayer(n1, n2, n, mapping="random"),
        GroupSum(num_classes, tau),
    )
