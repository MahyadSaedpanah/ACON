import os
import csv
import torch
import torch.nn.functional as F

class IdeaLogger:
    def __init__(self, log_dir=".", filename="log_uncertainty.csv"):
        self.log_path = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "mean_uncertainty_trg",
                    "std_uncertainty_trg",
                    "mean_kl_trg",
                    "mean_kl_src",
                    "mean_confidence_trg",
                    "align_t_tf_loss",
                    "align_s_tf_loss"
                ])

    def log(self, epoch, trg_t_pred=None, kl_src=None, kl_trg=None, uncert_trg_t=None,
            align_t_tf_loss=None, align_s_tf_loss=None):
        with torch.no_grad():
            mean_conf = F.softmax(trg_t_pred, dim=1).max(dim=1)[0].mean().item() if trg_t_pred is not None else 0
            kl_src_mean = kl_src.mean().item() if kl_src is not None else 0
            kl_trg_mean = kl_trg.mean().item() if kl_trg is not None else 0
            uncert_mean = uncert_trg_t.mean().item() if uncert_trg_t is not None else 0
            uncert_std = uncert_trg_t.std().item() if uncert_trg_t is not None else 0
            align_t_loss = align_t_tf_loss.item() if align_t_tf_loss is not None else 0
            align_s_loss = align_s_tf_loss.item() if align_s_tf_loss is not None else 0

        with open(self.log_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                uncert_mean,
                uncert_std,
                kl_trg_mean,
                kl_src_mean,
                mean_conf,
                align_t_loss,
                align_s_loss
            ])
