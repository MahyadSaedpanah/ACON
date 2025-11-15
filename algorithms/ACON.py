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
        self.mc_samples = getattr(args, 'mc_samples', 10)
        self.eps = 1e-6

        # model
        self.t_feature_extractor = CNN(configs)
        self.t_classifier = TemporalClassifierHead(self.t_feature_extractor.out_dim, configs.num_classes)
        # self.domain_classifier = Discriminator(self.t_feature_extractor.out_dim*self.avg_mode, self.args.disc_hid_dim)
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels, self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)


        # --- Graph module
        self.graph_module = GraphCorrelationModule(
            t_dim=self.t_feature_extractor.out_dim,
            f_dim=self.f_classifier.linear1.in_features,
            avg_mode=configs.avg_mode,    # از config دیتاست
            node_embed=16, gnn_hidden=64, out_dim=128, dropout=0.1
        )



        # discriminator
        self.domain_classifier = Discriminator(
            self.graph_module.out_dim,
            self.args.disc_hid_dim
        )

        self.avg_pooling = nn.AdaptiveAvgPool1d(self.avg_mode)
        
        # optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractor.parameters()},
	        {'params': self.t_classifier.parameters()},
            {'params': self.f_feature_extractor.parameters()},
            {'params': self.f_classifier.parameters()},
            {'params': self.graph_module.parameters(), 'lr': args.lr * 0.01}
            ],
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
    

    # MC Dropout
    def mc_pred_and_var(self, x, is_freq):
        T = self.mc_samples

        self.t_classifier.train()
        self.f_classifier.train()

        preds = []
        with torch.no_grad():
            for _ in range(T):
                if is_freq:
                    p, _ = self.f_classifier(x, get_feat=True)
                else:
                    p = self.t_classifier(x)
                preds.append(p)

        self.t_classifier.eval()
        self.f_classifier.eval()

        preds = torch.stack(preds)
        mean_pred = preds.mean(0)
        u = preds.var(0).mean(-1)
        return mean_pred, u
    

    def kl_loss_weighted(self, pred1, pred2, u1, u2, temp=1.0, eps=1e-6):
        # نرمال‌سازی دما
        log_p1 = F.log_softmax(pred1 / temp, dim=-1)
        p2 = F.softmax(pred2 / temp, dim=-1)

        # وزن: معکوس عدم قطعیت
        weight = 1.0 / (u1 + u2 + eps)  # [B]

        # KL معمولی
        kl = F.kl_div(log_p1, p2, reduction='none').sum(-1)  # [B]

        # KL وزن‌دار
        kl_weighted = (kl * weight).mean()

        return kl_weighted


    def update(self, src_x, src_y, trg_x):
        bs = src_x.size(0)
    
        # -------------------------------
        # 1) برچسب‌های دامنه (واقعی)
        # -------------------------------
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()
    
        # -------------------------------
        # 2) فیچرهای زمانی + پیش‌بینی
        # -------------------------------
        src_t_feat = self.t_feature_extractor(src_x)      # [B, d_T]
        src_t_pred = self.t_classifier(src_t_feat)        # [B, num_classes]
        trg_t_feat = self.t_feature_extractor(trg_x)
        trg_t_pred = self.t_classifier(trg_t_feat)
    
        # -------------------------------
        # 3) فیچرهای فرکانسی + پیش‌بینی
        # -------------------------------
        src_f_feat = self.f_feature_extractor(self.period_data(src_x, self.period))  # [B, d_F]
        trg_f_feat = self.f_feature_extractor(self.period_data(trg_x, self.period))
    
        src_a_cls, _ = self.get_amplitude(src_f_feat)  # amplitude خام (برای classification)
        trg_a_cls, _ = self.get_amplitude(trg_f_feat)
    
        src_f_pred, src_f_feat = self.f_classifier(src_a_cls, True)  # [B, num_classes], [B, d_F]
        trg_f_pred, trg_f_feat = self.f_classifier(trg_a_cls, True)
    
        # MC Dropout برای source
        src_t_pred_mc, u_T_src = self.mc_pred_and_var(src_t_feat, is_freq=False)
        src_f_pred_mc, u_F_src = self.mc_pred_and_var(src_a_cls, is_freq=True)
        
        # MC Dropout برای target
        trg_t_pred_mc, u_T_trg = self.mc_pred_and_var(trg_t_feat, is_freq=False)
        trg_f_pred_mc, u_F_trg = self.mc_pred_and_var(trg_a_cls, is_freq=True)

        # -------------------------------
        # 4) گراف (AttentionTopK + GCN)
        # -------------------------------
        h_src = self.graph_module(src_t_feat, src_f_feat)  # [B, out_dim]
        h_trg = self.graph_module(trg_t_feat, trg_f_feat)
        h_concat = torch.cat([h_src, h_trg], dim=0)        # [2B, out_dim]
    
        # -------------------------------
        # 5) Discriminator - مرحله اول (domain real labels)
        # -------------------------------
        disc_prediction = self.domain_classifier(h_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)
    
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()
    
        # -------------------------------
        # 6) Discriminator - مرحله دوم (fake labels برای فریب دادن)
        # -------------------------------
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
    
        disc_prediction = self.domain_classifier(h_concat)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)
    
        # -------------------------------
        # 7) Classification losses
        # -------------------------------
        src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)
    
        # -------------------------------
        # 8) Alignment losses
        # -------------------------------
        # align_s_tf_loss = self.kl(
        #     F.log_softmax(src_t_pred / self.kl_t, dim=-1),
        #     F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5
        # )
        # align_t_tf_loss = self.kl(
        #     F.log_softmax(trg_f_pred / self.kl_t, dim=-1),
        #     F.softmax(trg_t_pred / self.kl_t, dim=-1)
        # )

        align_s_tf_loss = self.kl_loss_weighted(
            src_t_pred_mc, src_f_pred_mc, u_T_src, u_F_src, temp=self.kl_t
        )
        align_t_tf_loss = self.kl_loss_weighted(
            trg_t_pred_mc, trg_f_pred_mc, u_T_trg, u_F_trg, temp=self.kl_t
        )

        # -------------------------------
        # 9) Conditional entropy loss (روی target)
        # -------------------------------
        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)
    
        # -------------------------------
        # 10) Total loss
        # -------------------------------
        loss = self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss) \
               + self.args.domain_trade_off * domain_loss \
               + self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f) \
               + self.args.align_t_trade_off * align_t_tf_loss \
               + self.args.align_s_trade_off * align_s_tf_loss
    
        # -------------------------------
        # 11) Update feature extractors + graph + classifiers
        # -------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # -------------------------------
        # 12) Return logging info
        # -------------------------------
        return {
            'Src_t_cls_loss': src_t_cls_loss.item(),
            'Src_f_cls_loss': src_f_cls_loss.item(),
            'Domain_loss': domain_loss.item(),
            'align source tf loss': align_s_tf_loss.item(),
            'align target tf loss': align_t_tf_loss.item(),
            'cond_ent_loss_t': entropy_trg_t.item(),
            'cond_ent_loss_f': entropy_trg_f.item(),
            'domain acc': domain_acc.item(),
            'u_T_src_avg': u_T_src.mean().item(),   # عدم قطعیت زمانی (source)
            'u_F_src_avg': u_F_src.mean().item(),   # عدم قطعیت فرکانسی (source)
            'u_T_trg_avg': u_T_trg.mean().item(),   # عدم قطعیت زمانی (target)
            'u_F_trg_avg': u_F_trg.mean().item(),   # عدم قطعیت فرکانسی (target)
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


