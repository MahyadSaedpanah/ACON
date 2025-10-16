# utils/plot_idea.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_idea_logs(csv_path, save_dir):
    df = pd.read_csv(csv_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def plot_column(column_name, ylabel=None):
        if column_name not in df.columns:
            print(f"[Warning] Column '{column_name}' not found.")
            return
        plt.figure()
        plt.plot(df['epoch'], df[column_name], label=column_name)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel or column_name)
        plt.title(f"{column_name} over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{column_name}.png"))
        plt.close()

    columns = [
        'uncertainty_mean',
        'uncertainty_std',
        'kl_src_mean',
        'kl_trg_mean',
        'align_t_tf_loss',
        'align_s_tf_loss',
        'trg_confidence'
    ]

    for col in columns:
        plot_column(col)
