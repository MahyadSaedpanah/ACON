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

        self.siglip_bias = nn.Parameter(torch.tensor(0.0))  # learnable bias for SigLIP

        

        # optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractor.parameters()},
	        {'params': self.t_classifier.parameters()},
            {'params': self.f_feature_extractor.parameters()},
            {'params': self.f_classifier.parameters()},
            {'params': self.graph_module.parameters(), 'lr': args.lr * 0.01},
            {'params': self.t_projector.parameters()},
            {'params': self.f_projector.parameters()},
            {'params': self.siglip_bias, 'lr': args.lr * 0.1}
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
    # def siglip_loss(self, anchor, positive, temperature):
    #     """
    #     True SigLIP loss: sigmoid-based contrastive learning.
    #     Uses binary cross-entropy with +1/-1 labels (not 0/1).
    #     """
    #     anchor = F.normalize(anchor, dim=-1)
    #     positive = F.normalize(positive, dim=-1)
        
    #     # Similarity matrix [B, B]
    #     logits = torch.matmul(anchor, positive.T) / temperature + self.siglip_bias
        
    #     # Labels: +1 for diagonal (positive pairs), -1 for off-diagonal (negatives)
    #     B = logits.size(0)
    #     labels = 2 * torch.eye(B, device=logits.device) - 1  # [B, B]: diagonal=1, off-diagonal=-1
        
    #     # SigLIP loss: -log(sigmoid(labels * logits))
    #     # این معادل است با: positive → -log(sigmoid(logits)), negative → -log(sigmoid(-logits))
    #     loss_i2j = -F.logsigmoid(labels * logits).mean()
        
    #     # دوطرفه: j→i (transpose)
    #     loss_j2i = -F.logsigmoid(labels * logits.T).mean()
        
    #     return (loss_i2j + loss_j2i) / 2

    def siglip_loss(self, anchor, positive, temperature=1.0):
        """
        Stable SigLIP loss (Google variant)
        anchor: [B, D]
        positive: [B, D]
        """

        # normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # similarity matrix
        logits = torch.matmul(anchor, positive.T) / temperature
        logits = logits + self.siglip_bias  # learnable bias

        B = logits.size(0)

        # -----------------
        # positive loss
        # -----------------
        pos_logits = logits.diag()   # sim(anchor_i, positive_i)
        loss_pos = -F.logsigmoid(pos_logits).mean()

        # -----------------
        # negative loss
        # -----------------
        # mask diagonal to remove positives from negatives
        neg_logits = logits - torch.eye(B, device=logits.device) * 1e9
        loss_neg = -F.logsigmoid(-neg_logits).mean()

        # final combined
        return (loss_pos + loss_neg) / 2

    
    
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
        src_t_proj = self.t_projector(src_t_feat)
        trg_t_proj = self.t_projector(trg_t_feat)
        src_f_proj = self.f_projector(src_f_contrast)
        trg_f_proj = self.f_projector(trg_f_contrast)

        # L_src_contrastive = self.siglip_loss(src_f_proj, src_t_proj, temperature=self.args.c_src_temp)
        # L_tgt_contrastive = self.siglip_loss(trg_t_proj, trg_f_proj, temperature=self.args.c_trg_temp)

        # SOURCE DOMAIN: Frequency → Temporal (F → T)
        # → frequency discriminative است → به temporal کمک کنه تا discriminability بگیره
        L_src_contrastive = self.siglip_loss(
            anchor=src_f_proj,      # Frequency as anchor (discriminative)
            positive=src_t_proj,    # Temporal as positive
            temperature=self.args.c_src_temp  # 0.9 ~ 1.0 (ضعیف‌تر)
        )

        # TARGET DOMAIN: Temporal → Frequency (T → F)
        # → temporal transferable است → به frequency کمک کنه تا transferability بگیره
        L_tgt_contrastive = self.siglip_loss(
            anchor=trg_t_proj,      # Temporal as anchor (transferable)
            positive=trg_f_proj,    # Frequency as positive
            temperature=self.args.c_trg_temp  # 0.05 ~ 0.10 (قوی‌تر)
        )

        # === Adaptive Ratio Balancing for SigLIP contrastive loss ===
        eps = 1e-6  # جلوگیری از تقسیم بر صفر

        L_src = L_src_contrastive.detach()
        L_tgt = L_tgt_contrastive.detach()

        # small loss → weak gradient → needs stronger weight
        inv_src = 1 / (L_src + eps)
        inv_tgt = 1 / (L_tgt + eps)

        alpha_src = inv_src / (inv_src + inv_tgt)
        alpha_tgt = inv_tgt / (inv_src + inv_tgt)

        contrastive_loss = alpha_src * L_src_contrastive + alpha_tgt * L_tgt_contrastive


        # ترکیب دو loss با ضرایب adaptive
        contrastive_loss = alpha_src * L_src_contrastive + alpha_tgt * L_tgt_contrastive

        # contrastive_loss = 0.1 * L_src_contrastive + 1.0 * L_tgt_contrastive

    
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

        # === Analytics (هر 10 epoch) ===
        ''' if (self.current_epoch + 1) % 25 == 0 or self.current_epoch == 0:
            with torch.no_grad():
                # نرمالیزه کردن برای محاسبه cosine similarity
                src_f_norm = F.normalize(src_f_proj, dim=-1)
                src_t_norm = F.normalize(src_t_proj, dim=-1)
                trg_f_norm = F.normalize(trg_f_proj, dim=-1)
                trg_t_norm = F.normalize(trg_t_proj, dim=-1)
                
                # === 1. Positive Pair Similarities (diagonal) ===
                src_pos_sim = F.cosine_similarity(src_f_norm, src_t_norm).mean().item()
                trg_pos_sim = F.cosine_similarity(trg_t_norm, trg_f_norm).mean().item()
                
                # === 2. Negative Pair Similarities (off-diagonal) ===
                # Source domain
                src_sim_matrix = torch.matmul(src_f_norm, src_t_norm.T)  # [B, B]
                B = src_sim_matrix.size(0)
                src_neg_mask = 1 - torch.eye(B, device=src_sim_matrix.device)
                src_neg_sim = (src_sim_matrix * src_neg_mask).sum() / (B * (B - 1))
                
                # Target domain
                trg_sim_matrix = torch.matmul(trg_t_norm, trg_f_norm.T)
                trg_neg_mask = 1 - torch.eye(B, device=trg_sim_matrix.device)
                trg_neg_sim = (trg_sim_matrix * trg_neg_mask).sum() / (B * (B - 1))
                
                # === 3. Alignment Quality (positive - negative gap) ===
                src_gap = src_pos_sim - src_neg_sim.item()
                trg_gap = trg_pos_sim - trg_neg_sim.item()
                
                # === 4. Intra-domain consistency ===
                src_t_self_sim = torch.matmul(src_t_norm, src_t_norm.T)
                src_f_self_sim = torch.matmul(src_f_norm, src_f_norm.T)
                trg_t_self_sim = torch.matmul(trg_t_norm, trg_t_norm.T)
                trg_f_self_sim = torch.matmul(trg_f_norm, trg_f_norm.T)
                
                src_t_consistency = (src_t_self_sim * src_neg_mask).sum() / (B * (B - 1))
                src_f_consistency = (src_f_self_sim * src_neg_mask).sum() / (B * (B - 1))
                trg_t_consistency = (trg_t_self_sim * trg_neg_mask).sum() / (B * (B - 1))
                trg_f_consistency = (trg_f_self_sim * trg_neg_mask).sum() / (B * (B - 1))
                
                # === 5. Feature Distribution Statistics ===
                src_t_std = src_t_proj.std(dim=0).mean().item()
                src_f_std = src_f_proj.std(dim=0).mean().item()
                trg_t_std = trg_t_proj.std(dim=0).mean().item()
                trg_f_std = trg_f_proj.std(dim=0).mean().item()

            # === Pretty Print ===
            print(f"\n{'='*90}")
            print(f"{'>>> SigLIP DETAILED ANALYTICS':^90}")
            print(f"{'Epoch: ' + str(self.current_epoch + 1):^90}")
            print(f"{'='*90}")
            
            # Source Domain
            print(f"\n  📊 SOURCE DOMAIN (F→T):")
            print(f"     ├─ Positive Similarity (diagonal):      {src_pos_sim:>6.4f}")
            print(f"     ├─ Negative Similarity (off-diagonal):  {src_neg_sim.item():>6.4f}")
            print(f"     ├─ Separation Gap (pos - neg):          {src_gap:>6.4f}  {'✅' if src_gap > 0.3 else '⚠️' if src_gap > 0.15 else '❌'}")
            print(f"     ├─ Temporal Self-Consistency:           {src_t_consistency.item():>6.4f}")
            print(f"     ├─ Frequency Self-Consistency:          {src_f_consistency.item():>6.4f}")
            print(f"     └─ Feature Std (T/F):                   {src_t_std:>6.4f} / {src_f_std:>6.4f}")
            
            # Target Domain
            print(f"\n  📊 TARGET DOMAIN (T→F):")
            print(f"     ├─ Positive Similarity (diagonal):      {trg_pos_sim:>6.4f}")
            print(f"     ├─ Negative Similarity (off-diagonal):  {trg_neg_sim.item():>6.4f}")
            print(f"     ├─ Separation Gap (pos - neg):          {trg_gap:>6.4f}  {'✅' if trg_gap > 0.3 else '⚠️' if trg_gap > 0.15 else '❌'}")
            print(f"     ├─ Temporal Self-Consistency:           {trg_t_consistency.item():>6.4f}")
            print(f"     ├─ Frequency Self-Consistency:          {trg_f_consistency.item():>6.4f}")
            print(f"     └─ Feature Std (T/F):                   {trg_t_std:>6.4f} / {trg_f_std:>6.4f}")
            
            # Model Parameters
            print(f"\n  🎛️  MODEL PARAMETERS:")
            print(f"     ├─ SigLIP Learnable Bias:               {self.siglip_bias.item():>6.4f}")
            print(f"     ├─ Source Temperature:                  {self.args.c_src_temp:>6.4f}")
            print(f"     ├─ Target Temperature:                  {self.args.c_trg_temp:>6.4f}")
            print(f"     └─ Contrastive Loss:                    {contrastive_loss.item():>6.4f}")
            
            # Loss Components
            print(f"\n  📈 LOSS BREAKDOWN:")
            print(f"     ├─ L_src_contrastive:                   {L_src_contrastive.item():>6.4f}")
            print(f"     └─ L_tgt_contrastive:                   {L_tgt_contrastive.item():>6.4f}")
            
            # Overall Assessment
            print(f"\n  🎯 TRANSFERABILITY ASSESSMENT:")
            if trg_pos_sim > 0.85 and trg_gap > 0.3:
                status = "✅ EXCELLENT - High transferability achieved!"
            elif trg_pos_sim > 0.75 and trg_gap > 0.2:
                status = "⚠️  GOOD - Moderate transferability"
            elif trg_pos_sim > 0.60 and trg_gap > 0.1:
                status = "⚠️  FAIR - Needs more training"
            else:
                status = "❌ POOR - Consider adjusting hyperparameters"
            print(f"     {status}")
            
            # Recommendations
            print(f"\n  💡 RECOMMENDATIONS:")
            if trg_gap < 0.15:
                print(f"     ⚠️  Target gap is low - consider increasing contrastive_trade_off")
            if src_neg_sim.item() > 0.5:
                print(f"     ⚠️  Source negatives too similar - consider decreasing temperature")
            if trg_t_std < 0.3 or trg_f_std < 0.3:
                print(f"     ⚠️  Low feature variance - risk of collapse!")
            if abs(trg_t_consistency.item() - trg_f_consistency.item()) > 0.2:
                print(f"     ⚠️  Temporal and Frequency inconsistency mismatch")
            if trg_pos_sim > 0.85 and trg_gap > 0.3:
                print(f"     ✅ All metrics look good!")
            
            print(f"{'='*90}\n")
        '''

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


