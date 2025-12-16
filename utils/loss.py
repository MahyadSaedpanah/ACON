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



class VICRegLoss(torch.nn.Module):
    def __init__(self, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    @staticmethod
    def _off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # invariance
        sim_loss = F.mse_loss(z1, z2)

        # center
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # variance
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

        # covariance
        b, d = z1.shape
        cov_z1 = (z1.T @ z1) / (b - 1)
        cov_z2 = (z2.T @ z2) / (b - 1)
        cov_loss = self._off_diagonal(cov_z1).pow(2).sum().div(d) + \
                   self._off_diagonal(cov_z2).pow(2).sum().div(d)

        return self.sim_coeff * sim_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss


class NTXentLoss(torch.nn.Module):
    """
    Cross-modal InfoNCE / NT-Xent for paired embeddings.
    z1, z2: [B, D], positives are (z1[i], z2[i]).
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = (z1 @ z2.t()) / self.tau
        labels = torch.arange(z1.size(0), device=z1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_12 + loss_21)

