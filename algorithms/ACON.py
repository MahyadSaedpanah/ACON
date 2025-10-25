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
    ACON (Base + Temporal-level Contrastive Learning)
    """

    def __init__(self, configs, device, args):
        super(ACON, self).__init__(configs)

        # --- Hyperparameters ---
        self.args = args
        self.device = device
        self.period = configs.period
        self.avg_mode = configs.avg_mode
        self.fft_mode = self.period // 2 + 1
        assert self.avg_mode < self.fft_mode
        self.kl_t = args.kl_t

        # --- Models ---
        self.t_feature_extractor = CNN(configs)
        self.t_classifier = TemporalClassifierHead(self.t_feature_extractor.out_dim, configs.num_classes)
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels,
                                                    self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)

        # --- Graph & Domain Discriminator ---
        self.graph_module = GraphCorrelationModule(
            t_dim=self.t_feature_extractor.out_dim,
            f_dim=self.f_classifier.linear1.in_features,
            avg_mode=configs.avg_mode,
            node_embed=16, gnn_hidden=64, out_dim=128, dropout=0.1
        )
        self.domain_classifier = Discriminator(self.graph_module.out_dim, self.args.disc_hid_dim)
        self.avg_pooling = nn.AdaptiveAvgPool1d(self.avg_mode)

        # --- Optimizers ---
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractor.parameters()},
            {'params': self.t_classifier.parameters()},
            {'params': self.f_feature_extractor.parameters()},
            {'params': self.f_classifier.parameters()},
            {'params': self.graph_module.parameters(), 'lr': args.lr * 0.01}
        ], lr=args.lr, weight_decay=args.weight_decay)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # --- Losses ---
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.kl = nn.KLDivLoss(reduction=args.kl_reduction)
        self.mc_passes = getattr(args, "mc_passes", 10)
        self.uncertainty_weight = getattr(args, "uncertainty_weight", 1.0)

        self.idea_logger = IdeaLogger(log_dir=getattr(args, "log_dir", "."))

    # =========================================================
    # Utility
    # =========================================================
    def period_data(self, x, period):
        B, N, L = x.shape
        if L % period != 0:
            pad_len = ((L // period) + 1) * period - L
            padding = torch.zeros([B, N, pad_len]).to(x.device)
            x = torch.cat([x, padding], dim=2)
        return x.reshape(B, N, -1, period).contiguous()

    def get_amplitude(self, x_fft):
        a = x_fft.abs()
        if a.dim() == 4:
            a = a.mean(dim=2)
        a_disc = a[:, :, :self.fft_mode]
        a_disc = self.avg_pooling(a_disc.mean(dim=1)).softmax(-1)
        a_cls = a[:, :, :self.fft_mode].reshape(a.size(0), -1)
        return a_cls, a_disc

    # =========================================================
    # Main Update
    # =========================================================
    def update(self, src_x, src_y, trg_x):
        bs = src_x.size(0)

        # --- Domain labels ---
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # --- Temporal Features ---
        src_t_feat = self.t_feature_extractor(src_x)
        src_t_pred = self.t_classifier(src_t_feat)
        trg_t_feat = self.t_feature_extractor(trg_x)
        trg_t_pred = self.t_classifier(trg_t_feat)

        # --- Frequency Features ---
        src_f_feat = self.f_feature_extractor(self.period_data(src_x, self.period))
        trg_f_feat = self.f_feature_extractor(self.period_data(trg_x, self.period))
        src_a_cls, _ = self.get_amplitude(src_f_feat)
        trg_a_cls, _ = self.get_amplitude(trg_f_feat)
        src_f_pred, src_f_feat = self.f_classifier(src_a_cls, True)
        trg_f_pred, trg_f_feat = self.f_classifier(trg_a_cls, True)

        # --- Graph fusion ---
        h_src = self.graph_module(src_t_feat, src_f_feat)
        h_trg = self.graph_module(trg_t_feat, trg_f_feat)
        h_concat = torch.cat([h_src, h_trg], dim=0)

        # =========================================================
        # 🔹 Temporal Contrastive Learning (NEW)
        # =========================================================
        with torch.no_grad():
            trg_soft = F.softmax(trg_t_pred, dim=1)
            trg_conf = trg_soft.max(dim=1)[0]
            trg_pseudo = trg_soft.argmax(dim=1)

        contrastive_loss_val = self.temporal_contrastive_loss(
            src_t_feat, trg_t_feat, src_y, trg_pseudo, trg_conf, temp=getattr(self.args, "cl_temp", 0.3)
        )
        # =========================================================

        # --- Domain Adversarial Training ---
        disc_prediction = self.domain_classifier(h_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # --- Reverse labels for domain alignment ---
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
        disc_prediction = self.domain_classifier(h_concat)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # --- Core Classification & Alignment Losses ---
        src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)
        align_s_tf_loss = self.kl(
            F.log_softmax(src_t_pred / self.kl_t, dim=-1),
            F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5
        )

        uncert_trg_t = self.compute_uncertainty(self.t_classifier, trg_t_feat)
        eps = 1e-5
        scaled_uncert = uncert_trg_t / (uncert_trg_t.max().detach() + eps)
        weight = torch.clamp(1 / (scaled_uncert + eps), min=0.1, max=10.0)

        kl_trg = F.kl_div(
            F.log_softmax(trg_f_pred / self.kl_t, dim=-1),
            F.softmax(trg_t_pred / self.kl_t, dim=-1),
            reduction='none'
        ).sum(dim=1)
        align_t_tf_loss = self.uncertainty_weight * (weight * kl_trg).mean()

        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)

        # --- Total Loss ---
        loss = self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss) \
             + self.args.domain_trade_off * domain_loss \
             + self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f) \
             + self.args.align_t_trade_off * align_t_tf_loss \
             + self.args.align_s_trade_off * align_s_tf_loss \
             + self.args.cl_trade_off * contrastive_loss_val  # lightweight contrastive effect

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Logging ---
        self.idea_logger.log(
            epoch=self.current_epoch,
            trg_t_pred=trg_t_pred,
            uncert_trg_t=uncert_trg_t,
            align_t_tf_loss=align_t_tf_loss,
            align_s_tf_loss=align_s_tf_loss,
            # contrastive_loss=contrastive_loss_val.item()
        )

        return {
            'Src_t_cls_loss': src_t_cls_loss.item(),
            'Src_f_cls_loss': src_f_cls_loss.item(),
            'Domain_loss': domain_loss.item(),
            'align source tf loss': align_s_tf_loss.item(),
            'align target tf loss': align_t_tf_loss.item(),
            'cond_ent_loss_t': entropy_trg_t.item(),
            'cond_ent_loss_f': entropy_trg_f.item(),
            'domain acc': domain_acc.item(),
            'contrastive_loss': contrastive_loss_val.item()
        }

    # =========================================================
    # 🔹 Temporal Contrastive Loss (NEW)
    # =========================================================
    def temporal_contrastive_loss(self, src_feats, trg_feats, src_labels, trg_pseudo, trg_conf, temp=0.3):
        device = src_feats.device
        src_feats = F.normalize(src_feats, dim=1)
        trg_feats = F.normalize(trg_feats, dim=1)

        # فقط targetهای با اطمینان بالا
        mask = trg_conf > 0.7
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        trg_feats = trg_feats[mask]
        trg_pseudo = trg_pseudo[mask]
        trg_conf = trg_conf[mask]

        feats = torch.cat([src_feats, trg_feats], dim=0)
        labels = torch.cat([src_labels, trg_pseudo], dim=0)
        conf = torch.cat([
            torch.ones(len(src_labels), device=device),
            trg_conf
        ], dim=0)

        sim = torch.matmul(feats, feats.T) / temp
        sim = torch.exp(sim)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        conf_weight = conf.unsqueeze(1) * conf.unsqueeze(0)
        sim = sim * conf_weight
        pos_sim = sim * pos_mask
        denom = sim.sum(dim=1, keepdim=True) + 1e-8
        loss = -torch.log((pos_sim.sum(dim=1) + 1e-8) / denom.squeeze())
        return loss.mean()

    # =========================================================
    def compute_uncertainty(self, model_fn, x, M=None):
        M = M or self.mc_passes
        preds = []
        model_fn.train()
        for _ in range(M):
            with torch.no_grad():
                preds.append(F.softmax(model_fn(x), dim=1))
        var = torch.var(torch.stack(preds), dim=0)
        return var.mean(dim=1)

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
            'domain_classifier': self.domain_classifier.state_dict(),
            'f_encoder': self.f_feature_extractor.state_dict(),
            'f_classifier': self.f_classifier.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.t_feature_extractor.load_state_dict(checkpoint['t_encoder'])
        self.t_classifier.load_state_dict(checkpoint['t_classifier'])
        self.f_feature_extractor.load_state_dict(checkpoint['f_encoder'])
        self.f_classifier.load_state_dict(checkpoint['f_classifier'])

    def get_domain_acc(self, pred, label):
        return torch.eq(pred.argmax(dim=1), label).float().mean()
