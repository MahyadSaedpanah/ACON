"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import ConditionalEntropyLoss, VICRegLoss, NTXentLoss, WeightedNTXentLoss
from algorithms.algorithms_base import Algorithm
from utils.module import *
from utils.aug_contrastive import aug_t, aug_f

class ACON(Algorithm):
    """
    ACON: https://openreview.net/pdf?id=cIBSsXowMr
    Dynamic Uncertainty-Aware Mutual Learning integrated
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
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels, self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)

        # ======== Contrastive heads (pre-graph) ========
        pd = getattr(args, "proj_dim", 128)

        t_in = self.t_feature_extractor.out_dim
        f_in = self.f_classifier.linear1.in_features

        # instance heads
        self.p_it = nn.Sequential(nn.Linear(t_in, pd), nn.ReLU(), nn.Linear(pd, pd))
        self.p_if = nn.Sequential(nn.Linear(f_in, pd), nn.ReLU(), nn.Linear(pd, pd))

        # shared heads
        self.p_st = nn.Sequential(nn.Linear(t_in, pd), nn.ReLU(), nn.Linear(pd, pd))
        self.p_sf = nn.Sequential(nn.Linear(f_in, pd), nn.ReLU(), nn.Linear(pd, pd))

        # contrastive losses
        self.vic = VICRegLoss(
            sim_coeff=getattr(args, "vic_sim", 25.0),
            var_coeff=getattr(args, "vic_var", 25.0),
            cov_coeff=getattr(args, "vic_cov", 1.0),
        )

        self.ntx = NTXentLoss(temperature=getattr(args, "temp", 0.2))

        self.wntx = WeightedNTXentLoss(
            temperature=getattr(args, "temp", 0.2),
            eps=getattr(args, "facg_eps", 1e-8),
        )
        # =================================================

        # Graph module
        self.graph_module = GraphCorrelationModule(
            t_dim=self.t_feature_extractor.out_dim,
            f_dim=self.f_classifier.linear1.in_features,
            avg_mode=configs.avg_mode,
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
            {'params': self.graph_module.parameters(), 'lr': args.lr * 0.01},
            # contrastive heads
            {'params': self.p_it.parameters()},
            {'params': self.p_if.parameters()},
            {'params': self.p_st.parameters()},
            {'params': self.p_sf.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.kl = nn.KLDivLoss(reduction=args.kl_reduction)

        self.mc_passes = getattr(args, "mc_passes", 10)
        self.uncertainty_weight = getattr(args, "uncertainty_weight", 1.0)

    # ------------------ Utility functions ------------------
    def period_data(self, x, period):
        B, N = x.size(0), x.size(1)
        if x.size(2) % period != 0:
            length = ((x.size(-1) // period) + 1) * period
            padding = torch.zeros([B, N, length - x.size(2)], device=x.device)
            out = torch.cat([x, padding], dim=2)
        else:
            length = x.size(2)
            out = x
        out = out.reshape(B, N, length // period, period).contiguous()
        return out

    def get_amplitude(self, x_fft):
        a = x_fft.abs()
        if a.dim() == 4:
            a = a.mean(dim=2)
        a_disc = a[:, :, :self.fft_mode]
        a_disc = self.avg_pooling(a_disc.mean(dim=1)).softmax(-1)
        a_cls = a[:, :, :self.fft_mode].reshape(a.size(0), -1)
        return a_cls, a_disc

    # ------------------ New Uncertainty Functions ------------------
    def compute_epistemic_uncertainty(self, model_fn, x, M=None, eps=1e-8):
        M = M or self.mc_passes
        preds = []
        was_training = getattr(model_fn, "training", None)
        if hasattr(model_fn, "train"):
            model_fn.train()
        with torch.no_grad():
            for _ in range(M):
                y = model_fn(x)
                p = F.softmax(y, dim=1)
                preds.append(p)
        if was_training is not None:
            model_fn.train(was_training)
        preds = torch.stack(preds, dim=0)      # [M, B, C]
        mean_pred = preds.mean(dim=0)          # [B, C]
        H_total = -(mean_pred * (mean_pred + eps).log()).sum(dim=1)
        H_each  = -(preds * (preds + eps).log()).sum(dim=2)
        H_data  = H_each.mean(dim=0)
        return (H_total - H_data).clamp(min=0.0)

    def dynamic_mutual_loss(self, t_pred, f_pred, t_feat, a_cls, eps=1e-8):
        u_t = self.compute_epistemic_uncertainty(self.t_classifier, t_feat, eps=eps)  # [B]
        u_f = self.compute_epistemic_uncertainty(self.f_classifier, a_cls, eps=eps)  # [B]
        teacher_is_t = (u_t < u_f).float()
        teacher_is_f = 1.0 - teacher_is_t
        kl_tf = F.kl_div(
            F.log_softmax(f_pred / self.kl_t, dim=-1),
            F.softmax(t_pred / self.kl_t, dim=-1),
            reduction='none'
        ).sum(dim=1)
        kl_ft = F.kl_div(
            F.log_softmax(t_pred / self.kl_t, dim=-1),
            F.softmax(f_pred / self.kl_t, dim=-1),
            reduction='none'
        ).sum(dim=1)
        kl_dynamic = teacher_is_t * kl_tf + teacher_is_f * kl_ft
        return kl_dynamic.mean()

    # ------------------ Update Function ------------------
    def update(self, src_x, src_y, trg_x):
        bs = src_x.size(0)

        # Domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # Feature extraction
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

        # ===================== Factorization-Aware Gating (Target) =====================
        use_facg = getattr(self.args, "facg", False)
        if use_facg:
            eps = getattr(self.args, "facg_eps", 1e-8)
            warmup = getattr(self.args, "facg_warmup", 0)
            if hasattr(self, "current_epoch") and self.current_epoch < warmup:
                w_sh = torch.zeros(trg_t_pred.size(0), device=self.device)
            else:
                p_t = F.softmax(trg_t_pred, dim=1)
                p_f = F.softmax(trg_f_pred, dim=1)
                agree = F.cosine_similarity(p_t, p_f, dim=1).clamp(0.0, 1.0)
                C = p_t.size(1)
                H_t = -(p_t * (p_t + eps).log()).sum(dim=1)
                H_f = -(p_f * (p_f + eps).log()).sum(dim=1)
                H_norm = (H_t + H_f) / 2.0 / torch.log(torch.tensor(C, device=self.device, dtype=p_t.dtype))
                conf = (1.0 - H_norm).clamp(0.0, 1.0)
                w_sh = (agree * conf).clamp(0.0, 1.0)
            w_sh = w_sh.detach()
            out_w_sh  = w_sh.mean().item()
            out_w_ins = (1.0 - w_sh).mean().item()
            w_ins_mean = (1.0 - w_sh).mean().detach()
        else:
            w_sh = None
            w_ins_mean = None
            out_w_sh, out_w_ins = 1.0, 0.0

        # ===================== Target Instance (VICReg) =====================
        ins_t_w = getattr(self.args, "ins_t", 1.0)
        if ins_t_w > 0:
            trg_xa = aug_t(trg_x, self.args)
            trg_t_feat_a = self.t_feature_extractor(trg_xa)
            trg_f_feat_a0 = self.f_feature_extractor(self.period_data(trg_xa, self.period))
            trg_a_cls_a, _ = self.get_amplitude(trg_f_feat_a0)
            trg_a_cls_a = aug_f(trg_a_cls_a, self.args)
            _, trg_f_feat_a = self.f_classifier(trg_a_cls_a, True)
            itT   = self.p_it(trg_t_feat)
            itT_a = self.p_it(trg_t_feat_a)
            ifT   = self.p_if(trg_f_feat)
            ifT_a = self.p_if(trg_f_feat_a)
            trg_inst_loss = self.vic(itT, itT_a) + self.vic(ifT, ifT_a)
            if use_facg and (w_ins_mean is not None):
                trg_inst_loss = trg_inst_loss * w_ins_mean
        else:
            trg_inst_loss = torch.tensor(0.0, device=self.device)

        # ===================== Target Shared (InfoNCE) =====================
        sh_t_w = getattr(self.args, "sh_t", 1.0)
        if sh_t_w > 0:
            stT = self.p_st(trg_t_feat)
            sfT = self.p_sf(trg_f_feat)
            if use_facg and (w_sh is not None):
                trg_sh_loss = self.wntx(stT, sfT, w_sh)
            else:
                trg_sh_loss = self.ntx(stT, sfT)
        else:
            trg_sh_loss = torch.tensor(0.0, device=self.device)

        # ===================== Source Instance & Shared =====================
        ins_s_w = getattr(self.args, "ins_s", 0.0)
        if ins_s_w > 0:
            src_xa = aug_t(src_x, self.args)
            src_t_feat_a = self.t_feature_extractor(src_xa)
            src_f_feat_a0 = self.f_feature_extractor(self.period_data(src_xa, self.period))
            src_a_cls_a, _ = self.get_amplitude(src_f_feat_a0)
            src_a_cls_a = aug_f(src_a_cls_a, self.args)
            _, src_f_feat_a = self.f_classifier(src_a_cls_a, True)
            itS   = self.p_it(src_t_feat)
            itS_a = self.p_it(src_t_feat_a)
            ifS   = self.p_if(src_f_feat)
            ifS_a = self.p_if(src_f_feat_a)
            src_inst_loss = self.vic(itS, itS_a) + self.vic(ifS, ifS_a)
        else:
            src_inst_loss = torch.tensor(0.0, device=self.device)

        sh_s_w = getattr(self.args, "sh_s", 0.0)
        if sh_s_w > 0:
            stS = self.p_st(src_t_feat)
            sfS = self.p_sf(src_f_feat)
            src_sh_loss = self.ntx(stS, sfS)
        else:
            src_sh_loss = torch.tensor(0.0, device=self.device)

        # ===================== Graph & Domain Loss =====================
        h_src = self.graph_module(src_t_feat, src_f_feat)
        h_trg = self.graph_module(trg_t_feat, trg_f_feat)
        h_concat = torch.cat([h_src, h_trg], dim=0)

        disc_prediction = self.domain_classifier(h_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)

        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        domain_label_src2 = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg2 = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat2 = torch.cat((domain_label_src2, domain_label_trg2), 0)
        disc_prediction = self.domain_classifier(h_concat)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat2)

        # ===================== Classification & Entropy =====================
        src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)
        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)

        # ===================== Target dynamic mutual learning =====================
        align_t_dyn_loss = self.dynamic_mutual_loss(trg_t_pred, trg_f_pred, trg_t_feat, trg_a_cls)
        align_t_tf_loss = self.args.align_t_trade_off * align_t_dyn_loss

        # ===================== Source KL alignment (fixed) =====================
        align_s_tf_loss = F.kl_div(
            F.log_softmax(src_t_pred / self.kl_t, dim=-1),
            F.softmax(src_f_pred / self.kl_t, dim=-1),
            reduction='batchmean'
        )

        # ===================== Total Loss =====================
        loss = self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss) \
            + self.args.domain_trade_off * domain_loss \
            + self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f) \
            + self.args.align_t_trade_off * align_t_tf_loss \
            + self.args.align_s_trade_off * align_s_tf_loss \
            + getattr(self.args, "ins_t", 1.0) * trg_inst_loss \
            + getattr(self.args, "sh_t", 1.0) * trg_sh_loss \
            + getattr(self.args, "ins_s", 0.0) * src_inst_loss \
            + getattr(self.args, "sh_s", 0.0) * src_sh_loss

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
            'domain acc': domain_acc.item(),
            'trg_inst_loss': trg_inst_loss.item(),
            'trg_sh_loss': trg_sh_loss.item(),
            'src_inst_loss': src_inst_loss.item(),
            'src_sh_loss': src_sh_loss.item(),
            'facg_w_sh_mean': out_w_sh,
            'facg_w_ins_mean': out_w_ins,
        }

    # ------------------ Prediction / Save / Load ------------------
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
