# tsne_visualizer.py
# 📍 ماژول مستقل برای استخراج ویژگی‌های گرافی ACON و رسم t-SNE

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def extract_graph_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).float()
            y = y.to(device)

            # استخراج ویژگی‌های زمانی و فرکانسی
            t_feat = model.t_feature_extractor(x)
            f_feat_raw = model.f_feature_extractor(model.period_data(x, model.period))
            amp, _ = model.get_amplitude(f_feat_raw)
            _, f_feat = model.f_classifier(amp, get_feat=True)

            # استخراج ویژگی نهایی گرافی
            h = model.graph_module(t_feat, f_feat)

            all_features.append(h.cpu())
            all_labels.append(y.cpu())

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels


def plot_tsne(features, labels, title='t-SNE of Graph Features', save_path=None):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {label}', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_tsne(model, dataloader, device, save_path=None):
    print("🔍 Running t-SNE visualization...")
    features, labels = extract_graph_features(model, dataloader, device)
    plot_tsne(features, labels, save_path=save_path)
