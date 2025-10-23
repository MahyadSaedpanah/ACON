import pandas as pd
import matplotlib.pyplot as plt

# ===== Load Data =====
# df = pd.read_csv("detailed_analysis.csv")
df = pd.read_csv("detailed_analysis.csv", 
                 on_bad_lines='skip',  # خطوط مشکل‌دار را رد کن
                 low_memory=False)
epochs = df['epoch']

# ========== 1. Entropy ==========
plt.figure(figsize=(8,5))
plt.plot(epochs, df['ent_src_t'], label='Entropy Source-T')
plt.plot(epochs, df['ent_src_f'], label='Entropy Source-F')
plt.plot(epochs, df['ent_trg_t'], label='Entropy Target-T')
plt.plot(epochs, df['ent_trg_f'], label='Entropy Target-F')
plt.xlabel('Epoch')
plt.ylabel('Entropy')
plt.title('Entropy Trends (Temporal vs Frequency)')
plt.legend()
plt.grid(True)
plt.savefig('plot1.png', dpi=300, bbox_inches='tight')
plt.close()  # مهم: حافظه را آزاد کن
# ========== 2. Alignment ==========
plt.figure(figsize=(8,5))
plt.plot(epochs, df['align_src'], label='Align Source')
plt.plot(epochs, df['align_trg'], label='Align Target')
plt.xlabel('Epoch')
plt.ylabel('Mean Softmax Distance')
plt.title('Temporal-Frequency Alignment Trends')
plt.legend()
plt.grid(True)
plt.savefig('plot2.png', dpi=300, bbox_inches='tight')
plt.close()  # مهم: حافظه را آزاد کن
# ========== 3. Correlation ==========
plt.figure(figsize=(8,5))
plt.plot(epochs, df['corr_tf_src'], label='Corr T-F Source')
plt.plot(epochs, df['corr_tf_trg'], label='Corr T-F Target')
plt.xlabel('Epoch')
plt.ylabel('Correlation Coefficient')
plt.title('Temporal-Frequency Correlation')
plt.legend()
plt.grid(True)
plt.savefig('plot3.png', dpi=300, bbox_inches='tight')
plt.close()  # مهم: حافظه را آزاد کن
# ========== 4. Uncertainty ==========
plt.figure(figsize=(8,5))
plt.plot(epochs, df['uncert_mean'], label='Uncertainty Mean')
plt.fill_between(epochs,
                 df['uncert_mean'] - df['uncert_std'],
                 df['uncert_mean'] + df['uncert_std'],
                 color='gray', alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Uncertainty')
plt.title('Uncertainty over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('plot4.png', dpi=300, bbox_inches='tight')
plt.close()  # مهم: حافظه را آزاد کن
# ========== 5. Class Confidence ==========
class_cols = [c for c in df.columns if c.startswith('class_conf_')]
plt.figure(figsize=(8,5))
for c in class_cols:
    plt.plot(epochs, df[c], label=c)
plt.xlabel('Epoch')
plt.ylabel('Confidence')
plt.title('Per-Class Confidence Trends')
plt.legend()
plt.grid(True)
plt.savefig('plot5.png', dpi=300, bbox_inches='tight')
plt.close()  # مهم: حافظه را آزاد کن
# ========== 6. Class Separability ==========
plt.figure(figsize=(7,5))
plt.plot(epochs, df['class_sep'], marker='o', label='Class Separability')
plt.xlabel('Epoch')
plt.ylabel('Avg Per-Class Accuracy')
plt.title('Class Separability across Epochs')
plt.legend()
plt.grid(True)
plt.savefig('plot6.png', dpi=300, bbox_inches='tight')
plt.close()  # مهم: حافظه را آزاد کن