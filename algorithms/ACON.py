"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import ConditionalEntropyLoss
from algorithms.algorithms_base import Algorithm
from utils.module import *
from utils.module import FrequencyAttentionModule


    
class ACON(Algorithm):
    """
    ACON: https://openreview.net/pdf?id=cIBSsXowMr
    """

    def __init__(self, configs, device, args):
        super(ACON, self).__init__(configs)

        # hyperparameters
        self.args = args
        self.device = device
        self.period = configs.period
        self.avg_mode = configs.avg_mode
        self.fft_mode = self.period // 2 + 1
        assert self.avg_mode < self.fft_mode
        self.kl_t = args.kl_t

        # model
        self.t_feature_extractor = CNN(configs)
        self.t_classifier = TemporalClassifierHead(self.t_feature_extractor.out_dim, configs.num_classes)
        self.domain_classifier = Discriminator(self.t_feature_extractor.out_dim, self.args.disc_hid_dim)
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels, self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)
        amp_dim = self.fft_mode * configs.input_channels
        self.freq_attention = FrequencyAttentionModule(amp_dim).to(device)
        self.avg_pooling = nn.AdaptiveAvgPool1d(self.avg_mode)
        

        # optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractor.parameters()},
	        {'params': self.t_classifier.parameters()},
            {'params': self.f_feature_extractor.parameters()},
            {'params': self.f_classifier.parameters()},
            {'params': self.freq_attention.parameters()}],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
       
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
      
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.kl = nn.KLDivLoss(reduction=args.kl_reduction)


    def period_data(self, x, period):
        B = x.size(0)
        N = x.size(1)
        # padding
        if x.size(2) % period != 0:
            length = ((x.size(-1) // period) + 1) * period
            padding = torch.zeros([x.shape[0], x.shape[1], (length - (x.size(2)))]).to(x.device)
            out = torch.cat([x, padding], dim=2)
        else:
            length = x.size(2)
            out = x
        # reshape
        out = out.reshape(B, N, length // period, period).contiguous()
        # print(out.shape)
        return out

    def get_amplitude(self, x_fft):
        a = x_fft.abs()
        if a.dim() == 4:
            a = a.mean(dim=2)
        a_disc = a[:, :, :self.fft_mode]
        a_disc = self.avg_pooling(a_disc.mean(dim=1)).softmax(-1)
        a_cls = a[:, :, :self.fft_mode]
        a_cls = a_cls.reshape(a_cls.size(0), -1)
        return a_cls, a_disc
    
    
    def update(self, src_x, src_y, trg_x):
        bs = src_x.size(0)

        # Domain labels
        domain_label_src = torch.ones(bs).to(self.device)
        domain_label_trg = torch.zeros(bs).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # --- Temporal features & predictions ---
        src_t_feat = self.t_feature_extractor(src_x)
        trg_t_feat = self.t_feature_extractor(trg_x)
        src_t_pred = self.t_classifier(src_t_feat)
        trg_t_pred = self.t_classifier(trg_t_feat)

        # --- Frequency features with attention ---
        src_f_complex = self.f_feature_extractor(self.period_data(src_x, self.period))
        trg_f_complex = self.f_feature_extractor(self.period_data(trg_x, self.period))

        src_f_feat_raw = self.get_amplitude(src_f_complex)[0]
        trg_f_feat_raw = self.get_amplitude(trg_f_complex)[0]

        src_attn = self.freq_attention(src_f_feat_raw)         # [B, D]
        trg_attn = self.freq_attention(trg_f_feat_raw)

        src_f_feat = src_attn * src_f_feat_raw                 # weighted amplitude
        trg_f_feat = trg_attn * trg_f_feat_raw

        src_f_pred = self.f_classifier(src_f_feat, get_feat=False)
        trg_f_pred = self.f_classifier(trg_f_feat, get_feat=False)

        # --- Domain discriminator ---
        ft_attn_concat = torch.cat([src_attn, trg_attn], dim=0)  # [2B, D]
        ft_feat_concat = torch.cat([src_t_feat, trg_t_feat], dim=0)  # [2B, Ft]

        # feat_x_pred = torch.bmm(ft_attn_concat.unsqueeze(2), ft_feat_concat.unsqueeze(1)).view(bs*2, -1).detach()
        # disc_prediction = self.domain_classifier(feat_x_pred)
        # Average attention per sample
        attn_avg = ft_attn_concat.mean(dim=1, keepdim=True)     # [B, 1]

        # Attention-weighted temporal feature
        feat_x_pred = attn_avg * ft_feat_concat                 # [B, Ft]

        disc_prediction = self.domain_classifier(feat_x_pred)   # ⬅️ اکنون ابعاد OK است

        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)

        # Update discriminator
        self.optimizer_disc.zero_grad()
        # disc_loss.backward()
        disc_loss.backward(retain_graph=True)
        self.optimizer_disc.step()

        # Reverse labels for adversarial update
        domain_label_src = torch.zeros(bs).long().to(self.device)
        domain_label_trg = torch.ones(bs).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # --- Recalculate discriminator loss for feature extractor ---
        # feat_x_pred = torch.bmm(ft_attn_concat.unsqueeze(2), ft_feat_concat.unsqueeze(1)).view(bs*2, -1)
        # disc_prediction = self.domain_classifier(feat_x_pred)
        attn_avg = ft_attn_concat.mean(dim=1, keepdim=True)
        feat_x_pred = attn_avg * ft_feat_concat
        disc_prediction = self.domain_classifier(feat_x_pred)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # --- Classification losses ---
        src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)

        # --- KL align losses ---
        align_s_tf_loss = self.kl(F.log_softmax(src_t_pred / self.kl_t, dim=-1),
                                  F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5)
        align_t_tf_loss = self.kl(F.log_softmax(trg_f_pred / self.kl_t, dim=-1),
                                  F.softmax(trg_t_pred / self.kl_t, dim=-1))

        # --- Conditional entropy ---
        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)

        # --- Attention loss L_A ---
        log_prob = F.log_softmax(src_f_pred, dim=-1)
        L_A = F.nll_loss(log_prob, src_y)

        # --- Total loss ---
        loss = (
            self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss) +
            self.args.domain_trade_off * domain_loss +
            self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f) +
            self.args.align_t_trade_off * align_t_tf_loss +
            self.args.align_s_trade_off * align_s_tf_loss +
            self.args.attn_trade_off * L_A
        )

        # Update feature extractor, classifiers, and attention
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Src_t_cls_loss': src_t_cls_loss.item(),
            'Src_f_cls_loss': src_f_cls_loss.item(),
            'Domain_loss': domain_loss.item(),
            'align source tf loss': align_s_tf_loss.item(),
            'align target tf loss': align_t_tf_loss.item(),
            'cond_ent_loss_t': entropy_trg_t.item(),
            'cond_ent_loss_f': entropy_trg_f.item(),
            'attn_loss': L_A.item(),
            'domain acc': domain_acc.item()
        }

    
    '''return predictions'''
    def predict(self, data):
        self.t_feature_extractor.eval()
        self.t_classifier.eval()
        with torch.no_grad():
            t_feat = self.t_feature_extractor(data)
            pred = self.t_classifier(t_feat)
        return pred
        
       

    def save_model(self, path):
        torch.save({
            't_encoder': self.t_feature_extractor.state_dict(),
            't_classifier': self.t_classifier.state_dict(),
            'domain_classifier':self.domain_classifier.state_dict(),
            'f_encoder':self.f_feature_extractor.state_dict(),
            'f_classifier':self.f_classifier.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.t_feature_extractor.load_state_dict(checkpoint['t_encoder'])
        self.t_classifier.load_state_dict(checkpoint['t_classifier'])
        self.f_feature_extractor.load_state_dict(checkpoint['f_encoder'])
        self.f_classifier.load_state_dict(checkpoint['f_classifier'])

    def get_domain_acc(self, pred, label):
        pred = torch.argmax(pred, dim=1)
        res = torch.sum(torch.eq(pred, label)) / label.size(0)
        return res



