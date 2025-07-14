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
from utils.module import CrossAttentionGate


    
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
        self.domain_classifier = Discriminator(self.t_feature_extractor.out_dim*self.avg_mode, self.args.disc_hid_dim)
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels, self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)
        # ---- cross-attention gates for source-only refinement ----
        dim_t = self.t_feature_extractor.out_dim
        dim_f = configs.input_channels * self.fft_mode
        self.cross_attn_source_t = CrossAttentionGate(dim_q=dim_t, dim_kv=dim_f, dim_out=dim_t)
        self.cross_attn_source_f = CrossAttentionGate(dim_q=dim_f, dim_kv=dim_t, dim_out=dim_f)
        self.avg_pooling = nn.AdaptiveAvgPool1d(self.avg_mode)
        

        # optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractor.parameters()},
	        {'params': self.t_classifier.parameters()},
            {'params': self.f_feature_extractor.parameters()},
            {'params': self.f_classifier.parameters()}],
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

        # === 1. استخراج و پالایش Source ===
        # خام
        src_t_raw = self.t_feature_extractor(src_x)                               # (B, dim_t)
        src_f_raw = self.f_feature_extractor(self.period_data(src_x, self.period))# complex output
        src_a_cls, _ = self.get_amplitude(src_f_raw)                              # (B, dim_t)
        _, src_f_feat = self.f_classifier(src_a_cls, get_feat=True)               # (B, dim_f)
    
        # کراس‌ا‌تِنشِن فقط روی source
        src_t_feat = self.cross_attn_source_t(src_t_raw, src_f_feat)              # (B, dim_t)
        src_f_feat = self.cross_attn_source_f(src_f_feat, src_t_raw)              # (B, dim_f)
    
        # پیش‌بینی‌های source
        src_t_pred = self.t_classifier(src_t_feat)                                # (B, num_classes)
        src_f_pred = self.f_classifier(src_f_feat)                                # (B, num_classes)
    
        # === 2. استخراج و پیش‌بینی Target (بدون Attention) ===
        trg_t_feat = self.t_feature_extractor(trg_x)                              # (B, dim_t)
        trg_t_pred = self.t_classifier(trg_t_feat)                                # (B, num_classes)
    
        trg_f_raw = self.f_feature_extractor(self.period_data(trg_x, self.period))# complex output
        trg_a_cls, _ = self.get_amplitude(trg_f_raw)                              # (B, dim_t)
        trg_f_pred, trg_f_feat = self.f_classifier(trg_a_cls, get_feat=True)      # (B, num_classes), (B, dim_f)
    
        # === 3. Domain Discriminator روی Outer Product ===
        # concat features for domain cls (temporal path)
        feat_concat = torch.cat((src_t_feat, trg_t_feat), dim=0)                  # (2B, dim_t)
        # concat amplitude distributions for discriminator (frequency path)
        src_a_disc = self.avg_pooling(src_f_feat).softmax(-1)                     # (B, avg_mode)
        trg_a_disc = self.avg_pooling(trg_f_feat).softmax(-1)                     # (B, avg_mode)
        ft_a_concat = torch.cat((src_a_disc, trg_a_disc), dim=0)                  # (2B, avg_mode)
    
        # outer product → (2B, dim_t * avg_mode)
        feat_x_pred = torch.bmm(ft_a_concat.unsqueeze(2), feat_concat.unsqueeze(1)) \
                      .view(bs*2, -1)
        # disc_prediction = self.domain_classifier(feat_x_pred)
        disc_prediction = self.domain_classifier(feat_x_pred.detach())
        domain_label_src = torch.ones(bs, dtype=torch.long, device=self.device)
        domain_label_trg = torch.zeros(bs, dtype=torch.long, device=self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
    
        # === 4. Update Discriminator ===
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()
    
        # === 5. Update Feature Extractors & Classifiers ===
        # prepare fake labels
        domain_label_src = torch.zeros(bs, dtype=torch.long, device=self.device)
        domain_label_trg = torch.ones(bs, dtype=torch.long, device=self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
    
        # repeat predictions for gradient
        feat_x_pred = torch.bmm(ft_a_concat.unsqueeze(2), feat_concat.unsqueeze(1)) \
                      .view(bs*2, -1)
        disc_prediction = self.domain_classifier(feat_x_pred)
    
        # compute losses
        domain_loss   = self.cross_entropy(disc_prediction, domain_label_concat)
        src_t_cls_loss= self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss= self.cross_entropy(src_f_pred.squeeze(), src_y)
        align_s_tf_loss = self.kl(
            F.log_softmax(src_t_pred/self.kl_t, dim=-1),
            F.softmax(src_f_pred/self.kl_t, dim=-1)+1e-5
        )
        align_t_tf_loss = self.kl(
            F.log_softmax(trg_f_pred/self.kl_t, dim=-1),
            F.softmax(trg_t_pred/self.kl_t, dim=-1)
        )
        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)
    
        loss = (
            self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss)
          + self.args.domain_trade_off * domain_loss
          + self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f)
          + self.args.align_t_trade_off   * align_t_tf_loss
          + self.args.align_s_trade_off   * align_s_tf_loss
        )
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return {
            'Src_t_cls_loss': src_t_cls_loss.item(),
            'Src_f_cls_loss': src_f_cls_loss.item(),
            'Domain_loss':    domain_loss.item(),
            'align source tf loss': align_s_tf_loss.item(),
            'align target tf loss': align_t_tf_loss.item(),
            'cond_ent_loss_t': entropy_trg_t.item(),
            'cond_ent_loss_f': entropy_trg_f.item(),
            'domain acc':     self.get_domain_acc(disc_prediction, domain_label_concat).item()
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



