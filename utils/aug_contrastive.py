import torch
import torch.nn.functional as F

# ----------------------------
# Time-series RandAugment-style bank
# ----------------------------

def _op_jitter(x, sigma):
    if sigma <= 0:
        return x
    return x + torch.randn_like(x) * sigma

def _op_scaling(x, sigma):
    if sigma <= 0:
        return x
    # per-sample scale
    s = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device, dtype=x.dtype) * sigma
    return x * s

def _op_shift(x, max_shift):
    if max_shift <= 0:
        return x
    # per-sample roll
    B = x.size(0)
    out = x.clone()
    # random shift in [-max_shift, +max_shift]
    k = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    for i in range(B):
        if k[i].item() != 0:
            out[i] = torch.roll(out[i], shifts=int(k[i].item()), dims=-1)
    return out

def aug_t(x, args):
    """
    RandAugment-style:
      - build a bank of ops
      - randomly pick K ops each call
      - apply them sequentially
    """
    K = int(getattr(args, "aug_k", 2))  # NEW: number of ops to apply
    # base magnitudes (keep your current args)
    jit = float(getattr(args, "aug_jit", 0.0))
    scl = float(getattr(args, "aug_scl", 0.0))
    shf = int(getattr(args, "aug_shf", 0))

    # bank (only include ops that are active)
    bank = []
    if jit > 0: bank.append(("jitter", lambda z: _op_jitter(z, jit)))
    if scl > 0: bank.append(("scaling", lambda z: _op_scaling(z, scl)))
    if shf > 0: bank.append(("shift",   lambda z: _op_shift(z, shf)))

    if len(bank) == 0:
        return x

    # choose K ops without replacement (like RandAugment)
    K = max(1, min(K, len(bank)))
    idx = torch.randperm(len(bank), device=x.device)[:K]

    out = x
    for j in idx.tolist():
        out = bank[j][1](out)
    return out


# ----------------------------
# Frequency augmentation (make mask per-sample)
# ----------------------------

def aug_f(a, args):
    """
    a: amplitude-like tensor, mask along the last dimension
    - jitter always possible
    - band-mask with prob p
    - IMPORTANT: mask is per-sample (per B), not batch-wise shared
    """
    f_jit = float(getattr(args, "f_jit", 0.0))
    p = float(getattr(args, "f_mask_p", 0.0))
    w = int(getattr(args, "f_mask_w", 0))

    out = a
    if f_jit > 0:
        out = out + torch.randn_like(out) * f_jit

    if p <= 0 or w <= 0:
        return out

    B = out.size(0)
    Fdim = out.size(-1)
    w = min(w, Fdim)

    # decide per-sample whether to mask
    do_mask = (torch.rand(B, device=out.device) < p)  # [B]
    if not do_mask.any():
        return out

    # sample per-sample start positions
    max_start = Fdim - w
    starts = torch.randint(0, max_start + 1, (B,), device=out.device)  # [B]

    # build per-sample band mask (broadcast over middle dims)
    idx = torch.arange(Fdim, device=out.device).view(1, Fdim)          # [1, F]
    band_ok = (idx < starts.view(B, 1)) | (idx >= (starts + w).view(B, 1))  # [B, F]

    # reshape to broadcast over all dims except last
    # e.g., out shape [B, ... , F]
    shape = [B] + [1] * (out.dim() - 2) + [Fdim]
    band_ok = band_ok.view(*shape).to(out.dtype)

    # apply only where do_mask True
    do_mask_view = do_mask.view(B, *([1] * (out.dim() - 1))).to(out.dtype)
    mask = (1.0 - do_mask_view) + do_mask_view * band_ok  # if not masking => 1, else => band_ok
    return out * mask
