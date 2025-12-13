# utils/augmentations.py
import torch
import torch.nn.functional as F

# --------- Time-domain ---------

def time_jitter(x, sigma=0.03):
    std = x.std(dim=-1, keepdim=True)
    noise = torch.randn_like(x) * (sigma * std + 1e-6)
    return x + noise

def time_scaling(x, sigma=0.1):
    B, C, T = x.shape
    scale = torch.randn(B, 1, 1, device=x.device) * sigma + 1.0
    return x * scale

def time_shift(x, max_frac=0.1):
    B, C, T = x.shape
    max_shift = int(T * max_frac)
    if max_shift < 1:
        return x
    shift = torch.randint(low=-max_shift, high=max_shift + 1, size=(1,), device=x.device).item()
    return torch.roll(x, shifts=shift, dims=-1)

def time_neighborhood_segment(x, min_frac=0.6, max_frac=0.9):
    B, C, T = x.shape
    L = int(T * torch.empty(1, device=x.device).uniform_(min_frac, max_frac).item())
    if L >= T:
        return x
    start = torch.randint(0, T - L + 1, (1,), device=x.device).item()
    seg = x[..., start:start+L]   # [B, C, L]
    seg = seg.unsqueeze(1)        # [B, 1, C, L]
    seg_up = F.interpolate(seg, size=(C, T), mode="bilinear", align_corners=False)
    return seg_up.squeeze(1)

def augment_time(x):
    x_aug = x
    if torch.rand(1) < 0.5:
        x_aug = time_jitter(x_aug, sigma=0.02)
    if torch.rand(1) < 0.5:
        x_aug = time_scaling(x_aug, sigma=0.05)
    # اگر خواستی ملایم‌تر باشه، این دو تا رو موقتاً خاموش کن:
    if torch.rand(1) < 0.3:
        x_aug = time_shift(x_aug)
    if torch.rand(1) < 0.2:
        x_aug = time_neighborhood_segment(x_aug)
    return x_aug

# --------- Frequency-domain ---------

def augment_freq_spectrum(x_fft, fft_mode, E=1, alpha=0.5, low_band_only=True):
    """
    x_fft: [B, C, K, F] یا [B, C, F]
    fft_mode: تعداد فرکانس‌های گسسته مورد استفاده (self.fft_mode)
    """
    a = x_fft.abs()
    phase = x_fft / (a + 1e-6)

    if a.dim() == 4:
        a_work = a.mean(dim=2)
        phase_work = phase.mean(dim=2)
    else:
        a_work = a
        phase_work = phase

    B, C, F = a_work.shape
    max_idx = min(fft_mode, F // 2) if low_band_only else min(fft_mode, F)

    a_aug = a_work.clone()

    for b in range(B):
        for _ in range(E):
            idx = torch.randint(0, max_idx, (1,), device=x_fft.device).item()
            if torch.rand(1) < 0.5:
                a_aug[b, :, idx] = 0.0  # remove
            else:
                Am = a_work[b].max()
                target_amp = alpha * Am
                mask = a_aug[b, :, idx] < target_amp
                a_aug[b, mask, idx] = target_amp

    if a.dim() == 4:
        a_aug_full = a_aug.unsqueeze(2).expand_as(a)
        phase_full = phase
        x_fft_aug = a_aug_full * (phase_full / (phase_full.abs() + 1e-6))
    else:
        x_fft_aug = a_aug * (phase_work / (phase_work.abs() + 1e-6))

    return x_fft_aug