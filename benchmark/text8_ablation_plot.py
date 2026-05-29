#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

sns.set_context("talk")

# ==============================
# Paths
# ==============================
paths = {
    "Sleep": "../pickle_files/text8_sleep_ablation_partial_threeway/sleep_partial.pkl",
    "No Sleep": "../pickle_files/text8_sleep_ablation_partial_threeway/no_sleep_partial.pkl",
    "constant-learning rate\n pattern blocks": "../pickle_files/text8_no_slow_heads_only_partial/no_slow_heads_partial.pkl",
    "Wake-only (All pattern &\n memory blocks trainable)": "../pickle_files/text8_wake_only_all_trainable_partial/wake_only_all_trainable_partial.pkl",
}

colors = {
    "Sleep": "r",
    "No Sleep": "b",
    "constant-learning rate\n pattern blocks": "g",
    "Wake-only (All pattern &\n memory blocks trainable)": "purple",
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

fontsize_title = 28
fontsize_axis = 26
fontsize_tick = 25
fontsize_legend = 20

# Wide figure with normal right-side legend space
fig, axes = plt.subplots(
    1,
    3,
    figsize=(20.0, 5.0),
    sharex=True
)

# ==============================
# Plot
# ==============================
for ax, (metric, title) in zip(axes, panels):
    for label, df in dfs.items():
        if metric not in df.columns:
            print(f"Skipping {label}: missing {metric}")
            continue

        x, y = smooth_curve_no_shift(df, metric, window=window)

        ax.plot(
            x,
            y,
            linewidth=3,
            color=colors[label],
            label=label,
        )

    ax.set_title(title, fontsize=fontsize_title, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize_tick
    )

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ==============================
# Shared legend on the right
# ==============================
handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="center left",
    bbox_to_anchor=(0.79, 0.53),
    frameon=False,
    fontsize=fontsize_legend,
    handlelength=2.0,
    handletextpad=0.65,
    labelspacing=0.65,
    borderaxespad=0.0,
)

# ==============================
# Shared labels
# ==============================
fig.text(
    0.018,
    0.55,
    "BPC",
    va="center",
    ha="center",
    rotation="vertical",
    fontsize=fontsize_axis,
)

fig.supxlabel(
    "Samples Seen",
    fontsize=fontsize_axis,
    y=0.0
)

# Leave right-side room for normal fig.legend
plt.subplots_adjust(
    left=0.065,
    right=0.79,
    top=0.86,
    bottom=0.23,
    wspace=0.30
)

# ==============================
# Save
# ==============================
os.makedirs("../plots", exist_ok=True)

pdf_path = "../plots/text8_ablation_forward_current_backward.pdf"
png_path = "../plots/text8_ablation_forward_current_backward.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nSaved:", pdf_path)
print("Saved:", png_path)
# %%