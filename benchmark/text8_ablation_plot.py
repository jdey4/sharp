#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter

sns.set_context("talk")

# ==============================
# Paths
# ==============================
paths = {
    "Sleep": "../pickle_files/text8_sleep_ablation_partial_threeway/sleep_partial.pkl",
    "No Sleep": "../pickle_files/text8_sleep_ablation_partial_threeway/no_sleep_partial.pkl",
    "No Pattern Slowdown": "../pickle_files/text8_no_slow_heads_only_partial/no_slow_heads_partial.pkl",
    "Wake-only All-Trainable": "../pickle_files/text8_wake_only_all_trainable_partial/wake_only_all_trainable_partial.pkl",
}

colors = {
    "Sleep": "r",
    "No Sleep": "b",
    "No Pattern Slowdown": "g",
    "Wake-only All-Trainable": "purple",
}

# ==============================
# Load data
# ==============================
dfs = {}

for label, path in paths.items():
    if os.path.exists(path):
        df = pd.read_pickle(path).sort_values("samples seen").reset_index(drop=True)
        dfs[label] = df

        print(f"\nLoaded {label}: {path}")
        print("n =", len(df))
        print("min samples =", df["samples seen"].min())
        print("max samples =", df["samples seen"].max())

        required_cols = ["samples seen", "forward_bpc", "current_bpc", "backward_bpc"]
        available_cols = [c for c in required_cols if c in df.columns]

        print(df[available_cols].head())
        print(df[available_cols].tail())
    else:
        print(f"\nMissing {label}: {path}")

# ==============================
# Smoothing without x-axis shift
# ==============================
window = 30

def smooth_curve_no_shift(df, key, window=30):
    x = df["samples seen"].values
    y = df[key].values

    mask = ~pd.isna(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return x, y

    y_s = (
        pd.Series(y)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .values
    )

    return x, y_s

# ==============================
# Plot settings
# ==============================
panels = [
    ("forward_bpc", "Forward"),
    ("current_bpc", "Current"),
    ("backward_bpc", "Backward"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

for ax, (metric, title) in zip(axes, panels):
    for label, df in dfs.items():
        if metric not in df.columns:
            print(f"Skipping {label}: missing {metric}")
            continue

        x, y = smooth_curve_no_shift(df, metric, window=window)

        ax.plot(
            x,
            y,
            linewidth=2.5,
            color=colors[label],
            label=label,
        )

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 1 decimal point on y-axis
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ==============================
# Shared legend, x-label, y-label
# ==============================
handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, .1),
)

# Common y-axis label
fig.text(
    0.015,
    0.6,
    "BPC",
    va="center",
    rotation="vertical",
    fontsize=20,
)

# Common x-axis label
fig.supxlabel("Samples Seen", fontsize=24, y=0.1)

plt.tight_layout(rect=[0.03, 0.04, 1, 1])

# ==============================
# Save
# ==============================
os.makedirs("../plots", exist_ok=True)

pdf_path = "../plots/text8_ablation_forward_current_backward.pdf"
png_path = "../plots/text8_ablation_forward_current_backward.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nSaved:", pdf_path)
print("Saved:", png_path)
# %%