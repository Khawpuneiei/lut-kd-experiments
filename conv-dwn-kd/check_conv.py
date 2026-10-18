"""Sanity checks for conv_dwn.py. Runs in seconds on CPU, no data needed.

1. A ConvLUTTree is a true convolution: the same tree at every position, so a spatially constant
   input gives a spatially constant output, and shifting the input shifts the output (away from borders).
2. ConvLUTTree matches a brute-force per-patch evaluation of the same LUT tables.
3. Residual init passes input bit "A" through: block output equals the selected leaf-0 bit of each tree.
4. Gradients reach every LUT table in a 3-block model (no all-zero grads), on CPU and, if present, CUDA.
5. Forward/backward of EFDFunction agree with the pilot's dwn.py on random inputs (if the pilot is present).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_dwn import ConvLUTTree, EFDFunction, make_conv_dwn  # noqa: E402

torch.manual_seed(0)


def brute_force(tree, x):
    """Evaluate tree-by-tree, patch-by-patch with plain Python LUT lookups."""
    B, C, H, W = x.shape
    k, pad = tree.kernel, tree.kernel // 2
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    leaves = 1 << tree.depth
    sel = tree.sel.reshape(tree.cout, leaves)
    out = torch.zeros(B, tree.cout, H, W)
    tables = [lvl.luts.detach() for lvl in tree.levels]
    for b in range(B):
        for i in range(H):
            for j in range(W):
                patch = xp[b, :, i:i + k, j:j + k].reshape(-1)         # channel-major, like unfold
                for c in range(tree.cout):
                    bits = [int(patch[t]) for t in sel[c]]
                    width = leaves
                    for lvl, tab in enumerate(tables):
                        width //= 2
                        nxt = []
                        for u in range(width):
                            a0, a1 = bits[2 * u], bits[2 * u + 1]
                            nxt.append(int(tab[c * width + u, a0 + 2 * a1] > 0))
                        bits = nxt
                    out[b, c, i, j] = bits[0]
    return out


def main():
    ok = True

    # 1 + 2: convolution property and brute force agreement
    tree = ConvLUTTree(cin=4, cout=6, depth=3, init="gaussian").eval()
    x = (torch.rand(2, 4, 8, 8) > 0.5).float()
    y = tree(x)
    y_bf = brute_force(tree, x)
    same = torch.equal(y, y_bf)
    print(f"[2] ConvLUTTree == brute force per-patch evaluation: {same}")
    ok &= same
    const = torch.ones(1, 4, 8, 8)
    yc = tree(const)[:, :, 1:-1, 1:-1]
    flat = all((yc[0, c] == yc[0, c, 0, 0]).all() for c in range(6))
    print(f"[1] constant input -> constant interior output: {flat}")
    ok &= flat
    xs = torch.roll(x, shifts=(1, 1), dims=(2, 3))
    ys = tree(xs)
    shift_ok = torch.equal(ys[:, :, 2:-1, 2:-1], torch.roll(y, shifts=(1, 1), dims=(2, 3))[:, :, 2:-1, 2:-1])
    print(f"[1] shifted input -> shifted output (interior): {shift_ok}")
    ok &= shift_ok

    # 3: residual init = pass-through of leaf bit 0 of each tree (noise < 1 never flips the sign)
    tree_r = ConvLUTTree(cin=4, cout=6, depth=3, init="residual").eval()
    y_r = tree_r(x)
    patches = torch.nn.functional.unfold(x, 3, padding=1).transpose(1, 2).reshape(-1, 36)
    leaf0 = patches[:, tree_r.sel.reshape(6, 8)[:, 0]].reshape(2, 64, 6).transpose(1, 2).reshape(2, 6, 8, 8)
    res_ok = torch.equal(y_r, leaf0)
    print(f"[3] residual-init tree passes its leaf-0 bit through: {res_ok}")
    ok &= res_ok

    # 4: gradients reach every table, CPU and CUDA
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        model = make_conv_dwn(9, 32, (8, 16, 16), head_luts=100, tau=1.0).to(dev)
        xb = (torch.rand(4, 9, 32, 32, device=dev) > 0.5).float()
        logits = model(xb)
        logits.sum().backward()
        zero = [n for n, p in model.named_parameters() if n.endswith("luts") and p.grad.abs().sum() == 0]
        g_ok = not zero
        print(f"[4] {dev}: logits {tuple(logits.shape)}, all LUT tables got gradient: {g_ok} {zero}")
        print(f"    circuit stats for (8,16,16): {model.circuit_stats()}")
        ok &= g_ok
    big = make_conv_dwn(9, 32, (32, 64, 64), head_luts=4000, tau=10.0)
    print(f"    circuit stats for the README's (32,64,64) model: {big.circuit_stats()}")

    # 5: agreement with the pilot's dwn.py
    pilot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dwn-kd-pilot")
    if os.path.exists(os.path.join(pilot, "dwn.py")):
        sys.path.insert(0, pilot)
        import dwn as pilot_dwn
        for n in (2, 3):
            J, I = 20, 30
            mapping = pilot_dwn.random_mapping(I, n, J)
            luts = torch.rand(J, 1 << n) * 2 - 1
            coef = pilot_dwn.efd_coefficients(n)
            xin = (torch.rand(16, I) > 0.5).float()
            x1, x2 = xin.clone().requires_grad_(), xin.clone().requires_grad_()
            l1, l2 = luts.clone().requires_grad_(), luts.clone().requires_grad_()
            o1 = pilot_dwn.EFDFunction.apply(x1, mapping, l1, coef, False)
            o2 = EFDFunction.apply(x2, mapping, l2, coef, False)
            g = torch.randn_like(o1)
            o1.backward(g)
            o2.backward(g)
            agree = torch.equal(o1, o2) and torch.allclose(x1.grad, x2.grad) and torch.allclose(l1.grad, l2.grad)
            print(f"[5] n={n}: EFDFunction matches the pilot's dwn.py (forward + both grads): {agree}")
            ok &= agree
    else:
        print("[5] pilot dwn.py not found next to this folder; skipped")

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
