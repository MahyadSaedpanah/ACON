import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


def plot_and_save(df, x_col, y_col, save_dir, smooth=True):
    """
    Plot and save a styled curve with optional smoothing and annotations.

    Args:
        df (pd.DataFrame): input dataframe
        x_col (str): x-axis column name (e.g. 'epoch')
        y_col (str): y-axis column name
        save_dir (str): directory to save plots
        smooth (bool): apply smoothing to noisy curves
    """
    if y_col not in df.columns:
        print(f"[Warning] Column '{y_col}' not found.")
        return

    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.2)

    x = df[x_col]
    y = df[y_col].astype(float)

    # --- Optional smoothing with moving average ---
    if smooth and len(y) > 5:
        window = max(3, len(y)//15)
        y_smooth = np.convolve(y, np.ones(window)/window, mode='same')
    else:
        y_smooth = y

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_smooth, color=sns.color_palette("tab10")[0], linewidth=2.5, label=f"{y_col}")
    plt.scatter(x, y, color=sns.color_palette("tab10")[1], s=25, alpha=0.7)

    plt.xlabel(x_col.capitalize(), fontsize=13)
    plt.ylabel(y_col.replace("_", " ").capitalize(), fontsize=13)
    plt.title(f"{y_col.replace('_', ' ').capitalize()} vs {x_col.capitalize()}", fontsize=15, pad=12)

    # annotate last value
    plt.text(x.iloc[-1], y_smooth[-1], f"{y_smooth[-1]:.4f}",
             fontsize=10, color='black', va='bottom', ha='right')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend(frameon=True, loc='best')

    save_path = os.path.join(save_dir, f"{y_col}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to log_uncertainty.csv")
    parser.add_argument("--save_dir", required=True, help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)

    # Rename columns if needed
    COLUMN_RENAMES = {
        'mean_uncertainty_trg': 'uncertainty_mean',
        'std_uncertainty_trg': 'uncertainty_std',
        'mean_kl_src': 'kl_src_mean',
        'mean_kl_trg': 'kl_trg_mean',
        'mean_confidence_trg': 'trg_confidence',
    }
    df.rename(columns=COLUMN_RENAMES, inplace=True)

    metrics = [
        'uncertainty_mean',
        'uncertainty_std',
        'kl_src_mean',
        'kl_trg_mean',
        'trg_confidence',
        'align_t_tf_loss',
        'align_s_tf_loss'
    ]

    for metric in metrics:
        plot_and_save(df, 'epoch', metric, args.save_dir)

if __name__ == "__main__":
    main()
