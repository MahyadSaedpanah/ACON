# utils/contrastive_debug_tool.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter
import os

def extract_features_and_pseudolabels(model, dataloader, device):
    model.eval()
    all_h, all_true, all_conf, all_pseudo = [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).float()
            y = y.to(device)

            # Feature extraction
            t_feat = model.t_feature_extractor(x)
            f_feat_raw = model.f_feature_extractor(model.period_data(x, model.period))
            amp, _ = model.get_amplitude(f_feat_raw)
            _, f_feat = model.f_classifier(amp, get_feat=True)
            h = model.graph_module(t_feat, f_feat)

            # Pseudo-label prediction
            logits = model.t_classifier(t_feat)
            probs = torch.softmax(logits, dim=1)
            conf, pseudo = probs.max(dim=1)

            all_h.append(h.cpu())
            all_true.append(y.cpu())
            all_conf.append(conf.cpu())
            all_pseudo.append(pseudo.cpu())

    return (
        torch.cat(all_h).numpy(),
        torch.cat(all_true).numpy(),
        torch.cat(all_conf).numpy(),
        torch.cat(all_pseudo).numpy()
    )

def analyze_pseudolabel_distribution(pseudo_labels, confidences, save_path):
    with open(save_path, "w") as f:
        f.write("🔍 Pseudo-label Distribution and Confidence Stats\n\n")
        counts = Counter(pseudo_labels)
        for cls in sorted(counts):
            f.write(f"Class {cls}: {counts[cls]} samples\n")
        f.write(f"\nAverage Confidence: {np.mean(confidences):.4f}\n")
        f.write(f"Max Confidence: {np.max(confidences):.4f}\n")
        f.write(f"Min Confidence: {np.min(confidences):.4f}\n")

def plot_tsne(features, labels, save_path, title="t-SNE of Graph Features"):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {label}', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def contrastive_debug(model, dataloader, device, save_dir="contrastive_debug", run_name="run1"):
    os.makedirs(save_dir, exist_ok=True)
    print("📊 Running full contrastive debug tool...")

    h, true_y, confs, pseudo_y = extract_features_and_pseudolabels(model, dataloader, device)

    # Stats
    txt_path = os.path.join(save_dir, f"{run_name}_pseudolabel_stats.txt")
    analyze_pseudolabel_distribution(pseudo_y, confs, txt_path)

    # t-SNE plots
    tsne_true_path = os.path.join(save_dir, f"{run_name}_tsne_true_labels.png")
    tsne_pseudo_path = os.path.join(save_dir, f"{run_name}_tsne_pseudo_labels.png")

    plot_tsne(h, true_y, tsne_true_path, title="t-SNE (True Labels)")
    plot_tsne(h, pseudo_y, tsne_pseudo_path, title="t-SNE (Pseudo Labels)")

    print(f"✅ Saved analysis to: {save_dir}")
