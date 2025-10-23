import os
import csv
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np


class AnalyticalLogger:
    """
    Advanced logger for idea analysis.
    Logs temporal-frequency relations, uncertainty, entropy, feature stats, and class-level metrics.
    """

    def __init__(self, log_dir="./idea_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "detailed_analysis.csv")
        self.initialized = False

    def _init_csv(self, keys):
        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
        self.initialized = True

    def log(self, epoch, src_t_pred, src_f_pred, trg_t_pred, trg_f_pred,
            src_y, uncert_trg_t, src_t_feat, src_f_feat, trg_t_feat, trg_f_feat,
            h_src, h_trg):

        with torch.no_grad():
            # ---------- Prediction Softmax ----------
            s_t_soft = F.softmax(src_t_pred, dim=1)
            s_f_soft = F.softmax(src_f_pred, dim=1)
            t_t_soft = F.softmax(trg_t_pred, dim=1)
            t_f_soft = F.softmax(trg_f_pred, dim=1)

            # ---------- Entropy ----------
            ent_s_t = -(s_t_soft * torch.log(s_t_soft + 1e-8)).sum(dim=1).mean().item()
            ent_s_f = -(s_f_soft * torch.log(s_f_soft + 1e-8)).sum(dim=1).mean().item()
            ent_t_t = -(t_t_soft * torch.log(t_t_soft + 1e-8)).sum(dim=1).mean().item()
            ent_t_f = -(t_f_soft * torch.log(t_f_soft + 1e-8)).sum(dim=1).mean().item()

            # ---------- Alignment Divergence ----------
            align_src = (s_t_soft - s_f_soft).abs().mean().item()
            align_trg = (t_t_soft - t_f_soft).abs().mean().item()

            # ---------- Uncertainty ----------
            uncert_stats = {
                'uncert_min': uncert_trg_t.min().item(),
                'uncert_max': uncert_trg_t.max().item(),
                'uncert_mean': uncert_trg_t.mean().item(),
                'uncert_std': uncert_trg_t.std().item(),
            }

            # ---------- Feature Statistics ----------
            feat_stats = {
                'src_t_mean': src_t_feat.mean().item(),
                'src_t_std': src_t_feat.std().item(),
                'src_f_mean': src_f_feat.mean().item(),
                'src_f_std': src_f_feat.std().item(),
                'trg_t_mean': trg_t_feat.mean().item(),
                'trg_f_mean': trg_f_feat.mean().item(),
                'graph_src_mean': h_src.mean().item(),
                'graph_trg_mean': h_trg.mean().item(),
            }

            # ---------- Correlation ----------
            corr_tf_src = torch.corrcoef(torch.stack([
                src_t_feat.mean(dim=1),
                src_f_feat.mean(dim=1)
            ]))[0, 1].item()
            corr_tf_trg = torch.corrcoef(torch.stack([
                trg_t_feat.mean(dim=1),
                trg_f_feat.mean(dim=1)
            ]))[0, 1].item()

            # ---------- Class-level Confidence (Source) ----------
            src_preds = s_t_soft.argmax(dim=1)
            class_conf = []
            for c in src_y.unique():
                mask = src_y == c
                conf = s_t_soft[mask].max(dim=1)[0].mean().item()
                class_conf.append((int(c.item()), conf))
            class_conf = dict(class_conf)

            # ---------- Confusion Matrix (Source Temporal vs Label) ----------
            cm = confusion_matrix(src_y.cpu().numpy(), src_preds.cpu().numpy(), normalize="true")
            class_sep = float(np.mean(np.diag(cm)))  # average per-class accuracy

            # ---------- Pack Everything ----------
            log_dict = {
                'epoch': epoch,
                'ent_src_t': ent_s_t,
                'ent_src_f': ent_s_f,
                'ent_trg_t': ent_t_t,
                'ent_trg_f': ent_t_f,
                'align_src': align_src,
                'align_trg': align_trg,
                'corr_tf_src': corr_tf_src,
                'corr_tf_trg': corr_tf_trg,
                'class_sep': class_sep,
                **feat_stats,
                **uncert_stats
            }

            # Add class confidence values
            for c, v in class_conf.items():
                log_dict[f'class_conf_{c}'] = v

            # ---------- Save ----------
            if not self.initialized:
                self._init_csv(log_dict.keys())

            with open(self.log_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_dict.keys())
                writer.writerow(log_dict)
