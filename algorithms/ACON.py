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
from utils.idea_logger import IdeaLogger


    
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

        # --- Projection heads for contrastive learning ---
        self.t_projector = nn.Sequential(
            nn.Linear(self.t_feature_extractor.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        freq_flat_dim = configs.input_channels * self.fft_mode

        self.f_projector = nn.Sequential(
            nn.Linear(freq_flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        

        # optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractor.parameters()},
	          {'params': self.t_classifier.parameters()},
            {'params': self.f_feature_extractor.parameters()},
            {'params': self.f_classifier.parameters()},
            {'params': self.graph_module.parameters(), 'lr': args.lr * 0.01},
            {'params': self.t_projector.parameters()},
            {'params': self.f_projector.parameters()}
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

        self.mc_passes = getattr(args, "mc_passes", 10)
        self.uncertainty_weight = getattr(args, "uncertainty_weight", 1.0)
        

        self.idea_logger = IdeaLogger(log_dir=getattr(args, "log_dir", "."))




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


    # SigLIP loss
    def siglip_loss(self, anchor, positive, temperature=0.3, bias=0.0):
        """
        True SigLIP loss: sigmoid-based contrastive learning.
        Uses binary cross-entropy with +1/-1 labels (not 0/1).
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # Similarity matrix [B, B]
        logits = torch.matmul(anchor, positive.T) / temperature + bias
        
        # Labels: +1 for diagonal (positive pairs), -1 for off-diagonal (negatives)
        B = logits.size(0)
        labels = 2 * torch.eye(B, device=logits.device) - 1  # [B, B]: diagonal=1, off-diagonal=-1
        
        # SigLIP loss: -log(sigmoid(labels * logits))
        # این معادل است با: positive → -log(sigmoid(logits)), negative → -log(sigmoid(-logits))
        loss_i2j = -F.logsigmoid(labels * logits).mean()
        
        # دوطرفه: j→i (transpose)
        loss_j2i = -F.logsigmoid(labels * logits.T).mean()
        
        return (loss_i2j + loss_j2i) / 2

    
    
    def update(self, src_x, src_y, trg_x):
        bs = src_x.size(0)
    
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()
    
        src_t_feat = self.t_feature_extractor(src_x)
        src_t_pred = self.t_classifier(src_t_feat)
        trg_t_feat = self.t_feature_extractor(trg_x)
        trg_t_pred = self.t_classifier(trg_t_feat)
    
        src_f_feat = self.f_feature_extractor(self.period_data(src_x, self.period))
        trg_f_feat = self.f_feature_extractor(self.period_data(trg_x, self.period))
    
        src_a_cls, _ = self.get_amplitude(src_f_feat)
        trg_a_cls, _ = self.get_amplitude(trg_f_feat)
    
        src_f_pred, src_f_feat = self.f_classifier(src_a_cls, True)
        trg_f_pred, trg_f_feat = self.f_classifier(trg_a_cls, True)

        src_f_contrast = src_f_feat  # این همون [B, C*fft_mode] هست → درست برای projector
        trg_f_contrast = trg_f_feat
    
        h_src = self.graph_module(src_t_feat, src_f_feat)
        h_trg = self.graph_module(trg_t_feat, trg_f_feat)
        h_concat = torch.cat([h_src, h_trg], dim=0)
    
        disc_prediction = self.domain_classifier(h_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)
    
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()
    
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
    
        disc_prediction = self.domain_classifier(h_concat)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)
    
        src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)
    
        eps = 1e-5
    
        align_s_tf_loss = self.kl(
            F.log_softmax(src_t_pred / self.kl_t, dim=-1),
            F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5
        )

        kl_src = F.kl_div(
            F.log_softmax(src_t_pred / self.kl_t, dim=-1),
            F.softmax(src_f_pred / self.kl_t, dim=-1),
            reduction='none'
        ).sum(dim=1)


        uncert_trg_t = self.compute_uncertainty(self.t_classifier, trg_t_feat)
        eps = 1e-5
        
        scaled_uncert = uncert_trg_t / (uncert_trg_t.max().detach() + eps)
        
        weight = 1 / (scaled_uncert + eps)
        weight = torch.clamp(weight, min=0.1, max=10.0)
        
        kl_trg = F.kl_div(
            F.log_softmax(trg_f_pred / self.kl_t, dim=-1),
            F.softmax(trg_t_pred / self.kl_t, dim=-1),
            reduction='none'
        ).sum(dim=1)
        
        align_t_tf_loss = self.uncertainty_weight * (weight * kl_trg).mean()
        
    
        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)


        # SigLIP Contrastive Loss
        src_t_proj = F.normalize(self.t_projector(src_t_feat), dim=-1)
        trg_t_proj = F.normalize(self.t_projector(trg_t_feat), dim=-1)
        src_f_proj = F.normalize(self.f_projector(src_f_contrast), dim=-1)   # ← دقیقاً همون src_f_feat که به گراف می‌ره!
        trg_f_proj = F.normalize(self.f_projector(trg_f_contrast), dim=-1)   # ← دقیقاً همون trg_f_feat که به گراف می‌ره!
    
        L_src_contrastive = self.siglip_loss(src_f_proj, src_t_proj, temperature=self.args.c_src_temp)
        L_tgt_contrastive = self.siglip_loss(trg_t_proj, trg_f_proj, temperature=self.args.c_trg_temp)
        contrastive_loss = L_src_contrastive + 1.0 * L_tgt_contrastive

    
        loss = self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss) \
            + self.args.domain_trade_off * domain_loss \
            + self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f) \
            + self.args.align_t_trade_off * align_t_tf_loss \
            + self.args.align_s_trade_off * align_s_tf_loss \
            + self.args.contrastive_trade_off * contrastive_loss
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

            
        self.idea_logger.log(
                epoch=self.current_epoch,
                trg_t_pred=trg_t_pred,
                kl_src=kl_src,
                kl_trg=kl_trg,
                uncert_trg_t=uncert_trg_t if 'uncert_trg_t' in locals() else None,
                align_t_tf_loss=align_t_tf_loss,
                align_s_tf_loss=align_s_tf_loss
        )

        # Contrastive log
                # === TARGET DOMAIN ANALYTICS (هر 10 epoch یک بار، بدون هیچ هزینه‌ای) ===
                # === SOURCE + TARGET DOMAIN ANALYTICS (هر 10 epoch یک بار) ===
        if (self.current_epoch + 1) % 10 == 0 or self.current_epoch == 0:
            with torch.no_grad():
                # Source similarity (F→T)
                src_sim = F.cosine_similarity(src_f_proj, src_t_proj).mean().item()
                # Target similarity (T→F) — روی batch فعلی target
                trg_sim = F.cosine_similarity(trg_f_proj, trg_t_proj).mean().item()

            print(f"\n>>> ANALYTICS @ Epoch {self.current_epoch + 1}")
            print(f"    Source Positive Similarity (F→T): {src_sim:.4f}")
            print(f"    Target Positive Similarity (T→F): {trg_sim:.4f}   ← این عدد مهمه!")
            if trg_sim > 0.85:
                print(f"    TRANSFERABILITY ACHIEVED! (Target sim = {trg_sim:.4f})")
            print("-" * 80)


        return {
            'Src_t_cls_loss': src_t_cls_loss.item(),
            'Src_f_cls_loss': src_f_cls_loss.item(),
            'Domain_loss': domain_loss.item(),
            'align source tf loss': align_s_tf_loss.item(),
            'align target tf loss': align_t_tf_loss.item(),
            'cond_ent_loss_t': entropy_trg_t.item(),
            'cond_ent_loss_f': entropy_trg_f.item(),
            'domain acc': domain_acc.item(),
            'contrastive_loss': contrastive_loss.item()
        }

    

    # Uncertainty-Aware Mutual Learning

    def compute_uncertainty(self, model_fn, x, M=None):
        M = M or self.mc_passes
        preds = []
    
        model_fn.train()  # ⬅️ اضافه کن!
    
        for _ in range(M):
            with torch.no_grad():
                y = model_fn(x)
                preds.append(F.softmax(y, dim=1))  # [B, C]
    
        stacked_preds = torch.stack(preds)  # [M, B, C]
        var = torch.var(stacked_preds, dim=0)  # [B, C]
        return var.mean(dim=1)  # [B]
    


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


