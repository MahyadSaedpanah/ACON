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

        self.mc_passes = getattr(args, 'mc_passes', 10)
        self.unc_warmup_epochs = getattr(args, 'unc_warmup_epochs', 20)
        self.uncertainty_weight = getattr(args, 'uncertainty_weight', 1.0)
        self.eps = 1e-8

        # model
        self.t_feature_extractor = CNN(configs)
        self.t_classifier = TemporalClassifierHead(self.t_feature_extractor.out_dim, configs.num_classes)
        # self.domain_classifier = Discriminator(self.t_feature_extractor.out_dim*self.avg_mode, self.args.disc_hid_dim)
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels, self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)


        # --- Graph module (attention + top-k + GCN) ---
        self.graph_module = GraphCorrelationModule(
            t_dim=self.t_feature_extractor.out_dim,
            f_dim=self.f_classifier.linear1.in_features,
            avg_mode=configs.avg_mode,    # از config دیتاست
            node_embed=16, gnn_hidden=64, out_dim=128, dropout=0.1
        )



        # discriminator روی خروجی گراف
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
    
    def compute_uncertainty(self, model_fn, x, M=None, is_freq=False):
        M = M or self.mc_passes
        model_fn.train()  # Dropout فعال باشه

        preds = []
        with torch.no_grad():
            for _ in range(M):
                if is_freq:
                    p, _ = model_fn(x, get_feat=True)
                else:
                    p = model_fn(x)
                preds.append(p)  # logits خام

        model_fn.eval()
        stacked = torch.stack(preds)           # [M, B, C]
        var = torch.var(stacked, dim=0)        # [B, C]
        return var.mean(dim=1)                 # [B] → عدم قطعیت
    
    
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
        align_s_tf_loss = self.kl(
            F.log_softmax(src_t_pred / self.kl_t, dim=-1),
            F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5
        )
        
        # Uncertainty in Target on T and F
        u_T_trg = self.compute_uncertainty(self.t_classifier, trg_t_feat, M=self.mc_passes, is_freq=False)
        u_F_trg = self.compute_uncertainty(self.f_classifier, trg_a_cls, M=self.mc_passes, is_freq=True)

        # روش جهت‌دار: فقط شاخه مطمئن به نامطمئن کمک کنه
        kl_T_to_F = F.kl_div(
            F.log_softmax(trg_t_pred / self.kl_t, dim=-1),
            F.softmax(trg_f_pred.detach() / self.kl_t, dim=-1) + 1e-8,
            reduction='none'
        ).sum(-1)

        kl_F_to_T = F.kl_div(
            F.log_softmax(trg_f_pred / self.kl_t, dim=-1),
            F.softmax(trg_t_pred.detach() / self.kl_t, dim=-1) + 1e-8,
            reduction='none'
        ).sum(-1)

        # هر شاخه فقط وقتی مطمئن باشه (u پایین باشه) اجازه کمک داره
        weight_T = torch.clamp(3.0 / (u_T_trg + 0.5), min=0.1, max=3.0)  # T مطمئن → وزن بالا
        weight_F = torch.clamp(3.0 / (u_F_trg + 0.5), min=0.1, max=3.0)  # F مطمئن → وزن بالا

        current_epoch = getattr(self, 'current_epoch', 0)
        warmup_ratio = min(current_epoch / self.unc_warmup_epochs, 1.0)
        adaptive_weight = self.uncertainty_weight * warmup_ratio

        align_t_tf_loss = adaptive_weight * (
            (weight_T * kl_T_to_F).mean() + 
            (weight_F * kl_F_to_T).mean()
        )

        # --- DEBUG: بررسی مقادیر کلیدی ---
        # --- DEBUG جدید: مخصوص DUAML (Directional Uncertainty-Aware Mutual Learning) ---
        # if current_epoch % 10 == 0 or align_t_tf_loss.item() < 1e-5:
        #     print(f"\n[DEBUG Epoch {current_epoch}] DUAML ANALYSIS")
        #     print(f"  u_T_trg (Temporal uncertainty):  mean={u_T_trg.mean().item():.6f}, min={u_T_trg.min().item():.6f}, max={u_T_trg.max().item():.6f}")
        #     print(f"  u_F_trg (Frequency uncertainty): mean={u_F_trg.mean().item():.6f}, min={u_F_trg.min().item():.6f}, max={u_F_trg.max().item():.6f}")
            
        #     # وزن‌های جهت‌دار
        #     print(f"  weight_T (T → F teaching strength): mean={weight_T.mean().item():.6f}, min={weight_T.min().item():.6f}, max={weight_T.max().item():.6f}")
        #     print(f"  weight_F (F → T teaching strength): mean={weight_F.mean().item():.6f}, min={weight_F.min().item():.6f}, max={weight_F.max().item():.6f}")
            
        #     # KL خام قبل از وزن‌دهی
        #     print(f"  kl_T_to_F (raw divergence T→F):   mean={kl_T_to_F.mean().item():.6f}, max={kl_T_to_F.max().item():.6f}")
        #     print(f"  kl_F_to_T (raw divergence F→T):   mean={kl_F_to_T.mean().item():.6f}, max={kl_F_to_T.max().item():.6f}")
            
        #     # لاس نهایی
        #     print(f"  adaptive_weight (warmup):         {adaptive_weight:.6f}")
        #     print(f"  align_t_tf_loss (final):          {align_t_tf_loss.item():.6f}")
            
        #     # تحلیل هوشمندانه خودکار (این خط‌ها رو هم اضافه کن!)
        #     active_T = (weight_T > 1.0).float().mean().item()
        #     active_F = (weight_F > 1.0).float().mean().item()
        #     print(f"  → Active Teaching Ratio: T→F: {active_T*100:5.1f}%  |  F→T: {active_F*100:5.1f}%")
            
        #     if active_T > 0.7:
        #         print(f"  → Temporal branch is DOMINANT teacher")
        #     elif active_F > 0.7:
        #         print(f"  → Frequency branch is DOMINANT teacher")
        #     elif active_T < 0.3 and active_F < 0.3:
        #         print(f"  → Both branches are uncertain → minimal teaching (safe mode)")
        #     else:
        #         print(f"  → Balanced mutual teaching")
                
        #     print("  " + "-"*70 + "\n")
    
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
            'u_T_trg_avg': u_T_trg.mean().item(),
            'u_F_trg_avg': u_F_trg.mean().item()
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


