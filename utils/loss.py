"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import torch
import torch.nn.functional as F


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)



def nt_xent_pair(anchor, pos, extra_neg=None, tau=0.1, reduction='mean'):
    """
    NT-Xent برای batch از جفت‌های (anchor_i, pos_i).
    anchor: [N, D]
    pos:    [N, D]
    extra_neg: اختیاری [K, D] برای اضافه کردن نگاتیوهای بیشتر
    tau: temperature (τ)
    خروجی: اگر reduction='mean' → اسکالر، اگر 'none' → [N]
    """
    anchor = F.normalize(anchor, dim=-1)
    pos = F.normalize(pos, dim=-1)

    if extra_neg is not None:
        extra_neg = F.normalize(extra_neg, dim=-1)
        candidates = torch.cat([pos, extra_neg], dim=0)   # [N+K, D]
    else:
        candidates = pos  # [N, D]

    logits = anchor @ candidates.t() / tau  # [N, N+K] یا [N, N]
    N = anchor.size(0)
    device = anchor.device

    labels = torch.arange(N, device=device)  # مثبت هر i، همون index i
    loss = F.cross_entropy(logits, labels, reduction=reduction)
    return loss


def _tfc_consistency_core(view_a, view_a_aug, view_b, view_b_aug, tau=0.1, margin=0.1):
    """
    view_a:  نمایی که می‌خواهیم anchor (student) باشد
    view_b:  نمای مقابل که بیشتر نقش teacher نرم را دارد
    """
    # S_ab: original a  → original b  (باید کوچک شود)
    extra_ab = torch.cat([view_a_aug, view_b_aug], dim=0)
    S_ab = nt_xent_pair(
        anchor=view_a,
        pos=view_b,
        extra_neg=extra_ab,
        tau=tau,
        reduction='none'
    )

    # S_aeb: original a → augmented b  (باید بزرگ‌تر از S_ab باشد)
    extra_aeb = torch.cat([view_a, view_a_aug, view_b], dim=0)
    S_aeb = nt_xent_pair(
        anchor=view_a,
        pos=view_b_aug,
        extra_neg=extra_aeb,
        tau=tau,
        reduction='none'
    )

    # S_eab: augmented a → original b
    extra_eab = torch.cat([view_a, view_b_aug], dim=0)
    S_eab = nt_xent_pair(
        anchor=view_a_aug,
        pos=view_b,
        extra_neg=extra_eab,
        tau=tau,
        reduction='none'
    )

    # S_aeeb: augmented a → augmented b
    extra_aeeb = torch.cat([view_a, view_b], dim=0)
    S_aeeb = nt_xent_pair(
        anchor=view_a_aug,
        pos=view_b_aug,
        extra_neg=extra_aeeb,
        tau=tau,
        reduction='none'
    )

    # hinge-style: می‌خواهیم S_ab + margin <= هر سه S_*
    term1 = F.relu(S_ab - S_aeb  + margin)
    term2 = F.relu(S_ab - S_eab  + margin)
    term3 = F.relu(S_ab - S_aeeb + margin)

    LC_i = term1 + term2 + term3   # [N]
    LC = LC_i.mean()               # scalar
    return LC


def tfc_consistency_loss_src(z_t, z_t_aug, z_f, z_f_aug, tau=0.1, margin=0.1):
    """
    L_C روی SOURCE:
    - time = student (anchor)
    - freq = teacher نرم
    """
    return _tfc_consistency_core(
        view_a=z_t,
        view_a_aug=z_t_aug,
        view_b=z_f,
        view_b_aug=z_f_aug,
        tau=tau,
        margin=margin
    )


def tfc_consistency_loss_tgt(z_t, z_t_aug, z_f, z_f_aug, tau=0.1, margin=0.1):
    """
    L_C روی TARGET:
    - freq = student (anchor)
    - time = teacher نرم
    """
    return _tfc_consistency_core(
        view_a=z_f,        # anchor = freq
        view_a_aug=z_f_aug,
        view_b=z_t,        # teacher = time
        view_b_aug=z_t_aug,
        tau=tau,
        margin=margin
    )