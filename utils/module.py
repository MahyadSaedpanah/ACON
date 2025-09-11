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
    

class FrequencyAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FrequencyAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1) 


class GraphCorrelation(nn.Module):
    def __init__(self, t_dim, f_dim, hidden_dim=128, out_dim=128, mode="fine"):
        super(GraphCorrelation, self).__init__()
        self.mode = mode
        self.proj_t = nn.Linear(t_dim, hidden_dim)
        self.proj_f = nn.Linear(f_dim, hidden_dim)

        # این دو خط جدید برای ساخت نود به ازای هر بُعد
        self.scalar2emb_t = nn.Linear(1, hidden_dim)
        self.scalar2emb_f = nn.Linear(1, hidden_dim)

        self.edge_mlp = nn.Linear(2*hidden_dim, 1)
        self.gcn = nn.Linear(hidden_dim, out_dim)

        # چون در readout قراره mean+max رو concat کنی، خروجی دو برابر میشه
        self.out_dim = 2 * out_dim

    def forward(self, t_feat, f_feat):
        # t_feat: [B, d_T], f_feat: [B, d_F]
        Ft = self.scalar2emb_t(t_feat.unsqueeze(-1))  # [B, d_T, hidden_dim]
        Fz = self.scalar2emb_f(f_feat.unsqueeze(-1))  # [B, d_F, hidden_dim]
        V = torch.cat([Ft, Fz], dim=1)                # [B, d_T+d_F, hidden_dim]
    
        # adjacency
        Vn = F.normalize(V, dim=-1)
        A = torch.einsum("bid,bjd->bij", Vn, Vn)      # [B, N, N]
        I = torch.eye(A.size(1), device=A.device).unsqueeze(0)
        A = A + I
    
        # message passing
        H = torch.bmm(A, V)                           # [B, N, hidden_dim]
        H = torch.relu(self.gcn(H))                   # [B, N, out_dim]
    
        # readout
        h_mean = H.mean(dim=1)                        # [B, out_dim]
        h_max  = H.max(dim=1).values                  # [B, out_dim]
        h = torch.cat([h_mean, h_max], dim=1)         # [B, 2*out_dim]
        return h


