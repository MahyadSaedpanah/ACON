"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F



def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.t_feat_dim, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.t_feat_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        self.out_dim = configs.t_feat_dim

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat





class TemporalClassifierHead(nn.Module):

    def __init__(self, in_dim, num_classes, bias=True):
        super(TemporalClassifierHead, self).__init__()
        self.head = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x):
        predictions = self.head(x)
        return predictions
    

class FrequencyClassifierHead(nn.Module):

    def __init__(self, in_dim, num_classes, bias=True):
        super(FrequencyClassifierHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x, get_feat=False):
        x = self.linear1(x)
        predictions = self.linear2(x)
        if get_feat:
            return predictions, x
        else:
            return predictions
    
    

class Discriminator(nn.Module):

    def __init__(self, in_dim, disc_hid_dim, layer_num=3):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        if layer_num == 3:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, disc_hid_dim),
                nn.ReLU(),
                nn.Linear(disc_hid_dim, disc_hid_dim),
                nn.ReLU(),
                nn.Linear(disc_hid_dim, 2)
            )
        elif layer_num == 2:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, disc_hid_dim),
                nn.ReLU(),
                nn.Linear(disc_hid_dim, 2)
            )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out




class FrequencyEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, mode, normalize=False):
        super(FrequencyEncoder, self).__init__()
        self.normalize = normalize
        self.mode = mode
        self.out_channels = out_channels
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, mode, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        dim_num = input.dim()
        if dim_num == 3:
            return torch.einsum("bix,iox->box", input, weights)
        elif dim_num == 4:
            # (b, c, period_num, period_length)
            return torch.einsum("bixy,ioy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.size(0)
        x_ft = torch.fft.rfft(x,norm='ortho', dim=-1)
        
        if self.normalize:
            x_ft = F.normalize(x_ft, dim=-1)
    
        dim_num = x_ft.dim()
        if dim_num == 3:
            out_ft = torch.zeros(batchsize, self.out_channels, self.mode,  device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :] = self.compl_mul1d(x_ft[:, :, :self.mode], self.weights1)
        elif dim_num == 4:
            out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(2), self.mode,  device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :, :] = self.compl_mul1d(x_ft[:, :, :, :self.mode], self.weights1)
        # print(out_ft)
        return out_ft



class BetterGCN(nn.Module):
    """
    GCN دو لایه با residual + LayerNorm + attention pooling روی نودها
    خروجی همیشه out_dim=128
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=128, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.pool_att = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self.out_dim = out_dim

    def forward(self, X, A):
        B, N, F = X.shape

        # --- لایه اول ---
        D = A.sum(-1, keepdim=True) + 1e-6
        A_norm = A / D
        H = torch.bmm(A_norm, X)
        H = self.lin1(H)
        H = self.norm1(H)
        H = torch.relu(H)
        H = self.dropout(H)

        # --- لایه دوم ---
        D = A.sum(-1, keepdim=True) + 1e-6
        A_norm = A / D
        H2 = torch.bmm(A_norm, H)
        H2 = self.lin2(H2)
        H2 = self.norm2(H2)

        # --- residual connection ---
        if H.shape[-1] == H2.shape[-1]:
            H2 = H2 + H

        # --- attention pooling روی نودها ---
        alpha = torch.softmax(self.pool_att(H2), dim=1)  # [B, N, 1]
        out = (alpha * H2).sum(dim=1)                    # [B, out_dim]

        return out



class GraphCorrelationModule(nn.Module):
    """
    نسخه گرافی بر اساس ساختار نسخه اصلی ACON
    - Temporal: کل فیچرها (بدون کاهش)
    - Frequency: خلاصه‌شده با AdaptiveAvgPool1d(avg_mode)
    - سپس تبدیل به نودها و پردازش با BetterGCN
    """
    def __init__(self, t_dim, f_dim, avg_mode=16,
                 node_embed=16, gnn_hidden=64, out_dim=128, dropout=0.1):
        super(GraphCorrelationModule, self).__init__()

        self.t_dim = t_dim
        self.f_dim = f_dim
        self.avg_mode = avg_mode
        self.num_nodes = t_dim + avg_mode

        # خلاصه کردن فرکانس (مثل نسخه اصلی)
        self.avg_pool = nn.AdaptiveAvgPool1d(avg_mode)

        # تع嵌‌سازی نودها
        self.node_mlp = nn.Sequential(
            nn.Linear(1, node_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # adjacency با MLP ساده
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_embed, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # GCN
        self.gcn = BetterGCN(node_embed, hidden_dim=gnn_hidden,
                             out_dim=out_dim, dropout=dropout)

        self.out_dim = out_dim

    def _build_adj(self, Z):
        B, N, E = Z.shape
        Zi = Z.unsqueeze(2).expand(B, N, N, E)
        Zj = Z.unsqueeze(1).expand(B, N, N, E)
        edges = torch.cat([Zi, Zj], dim=-1)              # [B, N, N, 2E]
        A = torch.sigmoid(self.edge_mlp(edges)).squeeze(-1)  # [B, N, N]

        # self-loop
        I = torch.eye(N, device=Z.device).unsqueeze(0).expand(B, N, N)
        return torch.clamp(A + 0.1 * I, 0., 1.)

    def forward(self, t_feat, f_feat):
        B = t_feat.size(0)

        # Temporal نودها
        t_nodes = t_feat.unsqueeze(-1)  # [B, t_dim, 1]

        # Frequency نودها (avg pooling → avg_mode)
        f_feat = f_feat.unsqueeze(1)              # [B, 1, d_F]
        f_nodes = self.avg_pool(f_feat).squeeze(1).unsqueeze(-1)  # [B, avg_mode, 1]

        # ترکیب نودها
        nodes = torch.cat([t_nodes, f_nodes], dim=1)  # [B, N, 1]

        # تع嵌‌سازی
        Z = self.node_mlp(nodes)   # [B, N, node_embed]

        # adjacency
        A = self._build_adj(Z)     # [B, N, N]

        # GCN
        out = self.gcn(Z, A)       # [B, out_dim]

        return out

