"""Checks dwn.py against a line-by-line Python port of torch-dwn 1.1.1's CUDA kernel
(efd_cuda_kernel.cu) and against torch-dwn's own learnable-mapping formulas."""
import sys

import torch

sys.path.insert(0, ".")
from dwn import EFDFunction, LearnableMappingFunction, efd_coefficients

torch.manual_seed(0)


def kernel_port(x, mapping, luts, g, n):
    alpha, beta = 0.5 * 0.75 ** (n - 1), 0.25 / 0.75
    B, In = x.shape
    J, K = mapping.shape[0], 1 << n
    out, xg, lg = torch.zeros(B, J), torch.zeros(B, In), torch.zeros(J, K)
    for i in range(B):
        for j in range(J):
            addr = 0
            for l in range(n):
                addr |= int(x[i, mapping[j, l]] > 0) << l
            out[i, j] = float(luts[j, addr] > 0)  # EFD forward followed by torch-dwn's STE
            lg[j, addr] += g[i, j]
            for l in range(n):
                w, a_off = 0.0, addr & ~(1 << l)
                for a2 in range(K):
                    fd = luts[j, a2].item() * alpha * beta ** bin(a_off ^ (a2 & ~(1 << l))).count("1")
                    w += fd if (a2 >> l) & 1 else -fd
                xg[i, mapping[j, l]] += w * g[i, j]
    return out, xg, lg


ok = True
for n in (2, 3, 6):
    for identity in (False, True):
        B, J = 5, 7
        In = J * n if identity else 9
        x = (torch.rand(B, In) > 0.5).float().requires_grad_(True)
        mapping = torch.arange(J * n).reshape(J, n) if identity else torch.randint(0, In, (J, n))
        luts = (torch.rand(J, 1 << n) * 2 - 1).requires_grad_(True)
        g = torch.randn(B, J)
        out = EFDFunction.apply(x, mapping, luts, efd_coefficients(n), identity)
        out.backward(g)
        o2, xg2, lg2 = kernel_port(x.detach(), mapping, luts.detach(), g, n)
        res = (torch.equal(out, o2), torch.allclose(x.grad, xg2, atol=1e-5), torch.allclose(luts.grad, lg2, atol=1e-5))
        ok &= all(res)
        print(f"EFD n={n} identity={identity}: forward {res[0]}, input grad {res[1]}, LUT grad {res[2]}")

# Learnable mapping: torch-dwn stores weights as (inputs, outputs); dwn.py stores the transpose.
B, In, Out, tau = 6, 11, 8, 0.001
x = (torch.rand(B, In) > 0.5).float().requires_grad_(True)
w_ref = torch.rand(In, Out)
g = torch.randn(B, Out)
w = w_ref.T.contiguous().requires_grad_(True)
y = LearnableMappingFunction.apply(x, w, tau)
y.backward(g)
ref_y = x.detach()[:, w_ref.argmax(dim=0)]
ref_wg = (2 * x.detach() - 1).T @ g
ref_xg = g @ torch.softmax(w_ref / tau, dim=0).T
res = (torch.equal(y, ref_y), torch.allclose(w.grad, ref_wg.T), torch.allclose(x.grad, ref_xg))
ok &= all(res)
print(f"Learnable mapping: forward {res[0]}, weight grad {res[1]}, input grad {res[2]}")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
