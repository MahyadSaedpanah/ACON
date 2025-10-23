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
    
    
def contrastive_loss(features, labels, temp=0.7, neg_thresh=0.1):
    """
    Improved supervised contrastive loss with hard-negative filtering.
    Args:
        features: torch.Tensor [N, D] (already normalized)
        labels: torch.Tensor [N]
        temp: float - temperature (higher = smoother similarities)
        neg_thresh: float - filter negatives with very low similarity
    """
    device = features.device
    N = features.size(0)

    # Normalize
    features = F.normalize(features, dim=1)

    # Similarity matrix (cosine)
    sim_matrix = torch.matmul(features, features.T) / temp
    sim_matrix = torch.exp(sim_matrix)

    # Positive mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask.fill_diagonal_(0)

    # Hard-negative filtering: ignore very dissimilar pairs
    sim_mask = (sim_matrix > neg_thresh).float()
    sim_matrix = sim_matrix * sim_mask

    # Positive similarity
    pos_sim = sim_matrix * mask

    # Denominator (sum over all valid pairs)
    denom = sim_matrix.sum(dim=1, keepdim=True)

    # Avoid division by zero
    loss = -torch.log((pos_sim.sum(dim=1) + 1e-8) / (denom.squeeze() + 1e-8))

    # Normalize by number of valid positive pairs
    valid_pairs = mask.sum()
    if valid_pairs > 0:
        loss = loss.sum() / valid_pairs
    else:
        loss = torch.tensor(0.0, device=device)

    return loss
