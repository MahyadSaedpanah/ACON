import torch

def aug_t(x, args):
    """
    Time augmentation.
    x: [B, C, T]
    """
    jit = getattr(args, "aug_jit", 0.02)
    scl = getattr(args, "aug_scl", 0.10)
    shf = getattr(args, "aug_shf", 8)

    # jitter
    if jit > 0:
        x = x + torch.randn_like(x) * jit

    # scaling
    if scl > 0:
        s = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device) * scl
        x = x * s

    # time shift
    if shf and shf > 0:
        xs = []
        for i in range(x.size(0)):
            k = torch.randint(-shf, shf + 1, (1,), device=x.device).item()
            xs.append(torch.roll(x[i], shifts=k, dims=-1))
        x = torch.stack(xs, dim=0)

    return x


def aug_f(a, args):
    """
    Frequency augmentation (lightweight).
    This operates on the amplitude vector/tensor returned by get_amplitude(): a_cls.
    It masks along the last dimension.
    """
    f_jit = getattr(args, "f_jit", 0.01)
    f_mask_p = getattr(args, "f_mask_p", 0.2)
    f_mask_w = getattr(args, "f_mask_w", 8)

    if f_jit > 0:
        a = a + torch.randn_like(a) * f_jit

    if f_mask_p > 0 and torch.rand(1, device=a.device).item() < f_mask_p:
        last_dim = a.size(-1)
        w = min(f_mask_w, last_dim)
        start = torch.randint(0, max(1, last_dim - w + 1), (1,), device=a.device).item()
        mask = torch.ones_like(a)
        mask[..., start:start + w] = 0.0
        a = a * mask

    return a
