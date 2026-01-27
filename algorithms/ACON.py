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

        # model
        self.t_feature_extractor = CNN(configs)
        self.t_classifier = TemporalClassifierHead(self.t_feature_extractor.out_dim, configs.num_classes)
        self.f_feature_extractor = FrequencyEncoder(configs.input_channels, configs.input_channels, self.fft_mode, configs.fft_normalize)
        self.f_classifier = FrequencyClassifierHead(self.fft_mode * configs.input_channels, configs.num_classes)

        # Graph module
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

        self.mc_passes = getattr(args, "mc_passes", 10)

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
        u_f = self.compute_epistemic_uncertainty(self.f_classifier, a_cls, eps=eps)

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

        kl_dynamic = teacher_is_t * kl_tf + teacher_is_f * kl_ft  # [B]

        # ---- REMOVE uncertainty weighting ----
        loss_dyn = kl_dynamic.mean()

        return loss_dyn

    
    def update(self, src_x, src_y, trg_x):
        bs = src_x.size(0)

        # -------------------------
        # 1) Forward (T/F) for source & target
        # -------------------------
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

        # -------------------------
        # 3) Dynamic uncertainty-aware mutual learning
        #    (THIS is the only align loss we keep)
        # -------------------------
        # ---------- Target: dynamic teacher (KEEP) ----------
        align_t_dyn_loss = self.dynamic_mutual_loss(
            t_pred=trg_t_pred,
            f_pred=trg_f_pred,
            t_feat=trg_t_feat,
            a_cls=trg_a_cls
        )
        align_t_tf_loss = self.args.align_t_trade_off * align_t_dyn_loss


        # ---------- Source: FIXED teacher (REVERT to ACON) ----------
        align_s_tf_loss = self.kl(
            F.log_softmax(src_t_pred / self.kl_t, dim=-1),
            F.softmax(src_f_pred / self.kl_t, dim=-1) + 1e-5
        )

        # -------------------------
        # 4) Domain adversarial loss (graph features)
        # -------------------------
        h_src = self.graph_module(src_t_feat, src_f_feat)
        h_trg = self.graph_module(trg_t_feat, trg_f_feat)
        h_concat = torch.cat([h_src, h_trg], dim=0)

        disc_prediction = self.domain_classifier(h_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)
        domain_acc = self.get_domain_acc(disc_prediction, domain_label_concat)

        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # IMPORTANT: rebuild labels if your original code used opposite labels for GRL-style training
        domain_label_src2 = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg2 = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat2 = torch.cat((domain_label_src2, domain_label_trg2), 0)

        disc_prediction = self.domain_classifier(h_concat)
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat2)

        # -------------------------
        # 5) Supervised source classification + target conditional entropy
        # -------------------------
        src_t_cls_loss = self.cross_entropy(src_t_pred.squeeze(), src_y)
        src_f_cls_loss = self.cross_entropy(src_f_pred.squeeze(), src_y)

        entropy_trg_t = self.criterion_cond(trg_t_pred)
        entropy_trg_f = self.criterion_cond(trg_f_pred)

        # -------------------------
        # 6) Total loss
        # -------------------------
        loss = self.args.cls_trade_off * (src_t_cls_loss + src_f_cls_loss) \
            + self.args.domain_trade_off * domain_loss \
            + self.args.entropy_trade_off * (entropy_trg_t + entropy_trg_f) \
            + self.args.align_t_trade_off * align_t_tf_loss \
            + self.args.align_s_trade_off * align_s_tf_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # -------------------------
        # 8) Return metrics
        # -------------------------
        return {
            'Src_t_cls_loss': src_t_cls_loss.item(),
            'Src_f_cls_loss': src_f_cls_loss.item(),
            'Domain_loss': domain_loss.item(),
            'align source tf loss': align_s_tf_loss.item(),
            'align target tf loss': align_t_tf_loss.item(),
            'cond_ent_loss_t': entropy_trg_t.item(),
            'cond_ent_loss_f': entropy_trg_f.item(),
            'domain acc': domain_acc.item(),
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

