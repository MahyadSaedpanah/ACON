import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_and_save(df, x_col, y_col, save_dir):
    if y_col not in df.columns:
        print(f"[Warning] Column '{y_col}' not found.")
        return
    plt.figure()
    plt.plot(df[x_col], df[y_col], marker='o')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{y_col}.png"))
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
