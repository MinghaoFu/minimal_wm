import matplotlib.pyplot as plt
import numpy as np

# ===================== Data =====================
ns_vals = [6, 12, 18, 24]
ns_labels = [r"$n_s$=6", r"$n_s$=12", r"$n_s$=18", r"$n_s$=24"]

# Methods (consistent order)
methods = ["Ours", "iVAE", "FactorVAE", "Slow-VAE", r"$\beta$-VAE"]

# For each d_z, metrics -> method -> values
# d_z = 2
mcc_dz2 = {
    "Ours": [0.73, 0.89, 0.90, 0.95],
    "iVAE": [0.62, 0.80, 0.84, 0.88],
    "FactorVAE": [0.54, 0.72, 0.78, 0.83],
    "Slow-VAE": [0.58, 0.76, 0.82, 0.86],
    r"$\beta$-VAE": [0.50, 0.68, 0.74, 0.80],
}

r2_dz2 = {
    "Ours": [62.6, 75.3, 90.3, 95.5],
    "iVAE": [52.0, 68.5, 83.0, 90.0],
    "FactorVAE": [45.0, 61.0, 76.0, 85.0],
    "Slow-VAE": [49.0, 65.0, 80.0, 88.0],
    r"$\beta$-VAE": [40.0, 58.0, 73.0, 82.0],
}

acc_dz2 = {
    "Ours": [51.3, 65.7, 73.6, 79.8],
    "iVAE": [41.0, 56.5, 63.0, 69.0],
    "FactorVAE": [36.0, 50.8, 60.5, 65.0],
    "Slow-VAE": [39.5, 54.0, 66.0, 68.2],
    r"$\beta$-VAE": [33.5, 48.0, 56.0, 63.5],
}

# d_z = 5
mcc_dz5 = {
    "Ours": [0.81, 0.85, 0.91, 0.92],
    "iVAE": [0.70, 0.78, 0.84, 0.87],
    "FactorVAE": [0.62, 0.71, 0.79, 0.83],
    "Slow-VAE": [0.66, 0.74, 0.82, 0.85],
    r"$\beta$-VAE": [0.58, 0.68, 0.76, 0.80],
}

r2_dz5 = {
    "Ours": [73.6, 77.2, 79.3, 89.2],
    "iVAE": [63.0, 69.0, 73.0, 82.0],
    "FactorVAE": [56.0, 63.0, 68.0, 77.0],
    "Slow-VAE": [60.0, 66.0, 71.0, 80.0],
    r"$\beta$-VAE": [52.0, 60.0, 65.0, 75.0],
}

acc_dz5 = {
    "Ours": [43.5, 59.9, 67.0, 72.2],
    "iVAE": [33.0, 49.5, 54.0, 62.0],
    "FactorVAE": [28.0, 45.5, 53.0, 58.0],
    "Slow-VAE": [31.0, 47.0, 56.0, 60.5],
    r"$\beta$-VAE": [24.5, 41.0, 49.0, 55.5],
}

# d_z = 9
mcc_dz9 = {
    "Ours": [0.31, 0.65, 0.75, 0.74],
    "iVAE": [0.22, 0.52, 0.64, 0.68],
    "FactorVAE": [0.18, 0.45, 0.58, 0.63],
    "Slow-VAE": [0.20, 0.48, 0.60, 0.65],
    r"$\beta$-VAE": [0.15, 0.40, 0.54, 0.60],
}

r2_dz9 = {
    "Ours": [12.6, 67.6, 82.3, 86.1],
    "iVAE": [8.0, 55.0, 72.0, 78.0],
    "FactorVAE": [6.0, 49.0, 66.0, 73.0],
    "Slow-VAE": [7.0, 52.0, 69.0, 76.0],
    r"$\beta$-VAE": [5.0, 45.0, 62.0, 70.0],
}

acc_dz9 = {
    "Ours": [10.9, 35.5, 40.4, 68.4],
    "iVAE": [6.0, 24.0, 30.0, 52.0],
    "FactorVAE": [4.5, 20.0, 26.0, 45.0],
    "Slow-VAE": [5.5, 22.0, 27.0, 48.0],
    r"$\beta$-VAE": [3.5, 18.0, 23.0, 41.0],
}

# ===================== Styling =====================
colors = {
    # Okabe-Ito palette (colorblind-friendly)
    "Ours": "#D55E00",        # vermillion
    "iVAE": "#0072B2",        # blue
    "FactorVAE": "#009E73",   # bluish green
    "Slow-VAE": "#CC79A7",    # reddish purple
    r"$\beta$-VAE": "#E69F00" # orange
}

markers = {
    "Ours": "o",
    "iVAE": "s",
    "FactorVAE": "D",
    "Slow-VAE": "^",
    r"$\beta$-VAE": "v",
}

marker_size = 7
linewidth = 2.0
linestyle = "--"
figsize = (12, 8.5)

# Font sizes (tuned for paper figure)
title_fontsize = 15
label_fontsize = 15
tick_fontsize = 12
legend_fontsize = 18

# ===================== Plotting =====================

fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=512)

metric_panels = [
    ("MCC", [mcc_dz2, mcc_dz5, mcc_dz9]),
    (r"$R^2$", [r2_dz2, r2_dz5, r2_dz9]),
    ("Acc.", [acc_dz2, acc_dz5, acc_dz9]),
]

dz_labels = [r"$d_z=2$", r"$d_z=5$", r"$d_z=9$"]
x = np.arange(len(ns_vals))

for col, (metric_name, dz_blocks) in enumerate(metric_panels):
    for row, dz_data in enumerate(dz_blocks):
        ax = axes[row][col]
        for method in methods:
            ax.plot(
                x,
                dz_data[method],
                color=colors[method],
                marker=markers[method],
                markersize=marker_size,
                linestyle=linestyle,
                linewidth=linewidth,
                label=method,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ns_labels, rotation=0, fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

        if row == 0:
            ax.set_title(metric_name, fontsize=title_fontsize)
        if col == 0:
            ax.set_ylabel(dz_labels[row], fontsize=label_fontsize)

# Shared legend (one line)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.03), fontsize=legend_fontsize)

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace=0.25, hspace=0.35)
png_path = "synthetic.png"
pdf_path = "synthetic.pdf"
plt.savefig(png_path, dpi=512, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
plt.show()
