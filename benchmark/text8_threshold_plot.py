#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# Style
# ============================================================
sns.set_context("talk")


# ============================================================
# Paths
# ============================================================
partial_root = "../pickle_files/text8_threshold_sweep_partial"
plot_dir = "../plots"
os.makedirs(plot_dir, exist_ok=True)

# Optional: if tau=1e-2 comes from existing sleep run instead
# of the threshold sweep, set this path. Otherwise leave as None.
tau_1e2_existing_path = None
# Example:
# tau_1e2_existing_path = "../pickle_files/text8_sleep_ablation_partial_threeway/sleep_partial.pkl"


# ============================================================
# Load threshold partial files
# ============================================================
def load_threshold_sweep(partial_root):
    dfs = []

    paths = sorted(
        glob.glob(os.path.join(partial_root, "*", "*_partial.pkl"))
    )

    if len(paths) == 0:
        raise RuntimeError(f"No threshold partial files found in: {partial_root}")

    for path in paths:
        try:
            df = (
                pd.read_pickle(path)
                .sort_values("samples seen")
                .reset_index(drop=True)
            )

            if len(df) == 0:
                print(f"Skipping empty file: {path}")
                continue

            if "threshold" not in df.columns:
                print(f"Skipping {path}: missing threshold column")
                continue

            required = [
                "samples seen",
                "threshold",
                "threshold_tag",
                "memory_update_percent_window",
                "forward_bpc",
                "current_bpc",
                "backward_bpc",
            ]

            missing = [c for c in required if c not in df.columns]
            if len(missing) > 0:
                print(f"Skipping {path}: missing columns {missing}")
                continue

            dfs.append(df)

            print(f"\nLoaded: {path}")
            print(
                df[
                    [
                        "threshold",
                        "threshold_tag",
                        "samples seen",
                        "memory_update_percent_window",
                        "forward_bpc",
                        "current_bpc",
                        "backward_bpc",
                    ]
                ].tail()
            )

        except Exception as e:
            print(f"Could not load {path}: {e}")

    if len(dfs) == 0:
        raise RuntimeError(
            f"No usable threshold partial files found in: {partial_root}"
        )

    return pd.concat(dfs, ignore_index=True)


df = load_threshold_sweep(partial_root)


# ============================================================
# Optionally add tau=1e-2 from existing run
# ============================================================
if tau_1e2_existing_path is not None and os.path.exists(tau_1e2_existing_path):
    df_tau = (
        pd.read_pickle(tau_1e2_existing_path)
        .sort_values("samples seen")
        .reset_index(drop=True)
    )

    required_cols = ["forward_bpc", "current_bpc", "backward_bpc"]

    if all(c in df_tau.columns for c in required_cols):
        df_tau = df_tau.copy()
        df_tau["threshold"] = 1e-2
        df_tau["threshold_tag"] = "tau_1em02"

        if "memory_update_percent_window" not in df_tau.columns:
            df_tau["memory_update_percent_window"] = np.nan

        df = pd.concat([df, df_tau], ignore_index=True)
        print(f"\nAdded existing tau=1e-2 run: {tau_1e2_existing_path}")
    else:
        print(
            f"\nExisting tau=1e-2 file lacks three-way BPC columns: "
            f"{tau_1e2_existing_path}"
        )


# ============================================================
# Threshold labels
# ============================================================
def threshold_label(tau):
    tau = float(tau)

    if tau == 0.0:
        return r"$\tau=0$"

    exponent = int(np.round(np.log10(tau)))
    return rf"$\tau=10^{{{exponent}}}$"


thresholds = sorted(df["threshold"].dropna().unique())
labels = {tau: threshold_label(tau) for tau in thresholds}

print("\nThresholds found:")
for tau in thresholds:
    print(f"{tau} -> {labels[tau]}")


# ============================================================
# Smoothing without shifting x-axis
# ============================================================
window = 30


def smooth_curve_no_shift(x, y, window=30):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(y)
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


# ============================================================
# Plot
# ============================================================
panels = [
    ("memory_update_percent_window", "Wake Memory Update"),
    ("forward_bpc", "Forward"),
    ("current_bpc", "Current"),
    ("backward_bpc", "Backward"),
]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(28, 5),
    sharex=True
)

palette = sns.color_palette("tab10", n_colors=len(thresholds))
color_map = {tau: palette[i] for i, tau in enumerate(thresholds)}

for ax, (metric, title) in zip(axes, panels):
    for tau in thresholds:
        df_tau = (
            df[df["threshold"] == tau]
            .sort_values("samples seen")
            .reset_index(drop=True)
        )

        x = df_tau["samples seen"].values
        y = df_tau[metric].values

        x_s, y_s = smooth_curve_no_shift(x, y, window=window)

        ax.plot(
            x_s,
            y_s,
            linewidth=4,
            color=color_map[tau],
            label=labels[tau],
        )

    ax.set_title(title, fontsize=32)
    ax.set_xticks([0, 5e7, 1e8])
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", which="major", labelsize=27)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# First panel y-label
axes[0].set_ylabel("Update Rate (%)", fontsize=30)

# No repeated y-labels on BPC panels
axes[1].set_ylabel("")
axes[2].set_ylabel("")
axes[3].set_ylabel("")

# Common BPC y-label for right three panels
fig.text(
    0.22,
    0.57,
    "BPC",
    va="center",
    rotation="vertical",
    fontsize=30,
)

# ============================================================
# Shared legend on the right
# ============================================================
handles, legend_labels = axes[0].get_legend_handles_labels()

leg = fig.legend(
    handles,
    legend_labels,
    title="Threshold",
    loc="center left",
    bbox_to_anchor=(0.845, 0.52),
    frameon=False,
    fontsize=24,
    title_fontsize=24,
    handlelength=2.2,
    handletextpad=0.7,
    labelspacing=0.75,
    borderaxespad=0.0,
)

# Center title relative to legend box
leg.get_title().set_ha("center")
try:
    leg._legend_box.align = "center"
except Exception:
    pass

# One global x-axis label
fig.supxlabel("Samples Seen", fontsize=30, y=0.0)

# Leave room on the right for legend
plt.subplots_adjust(
    left=0.06,
    right=0.82,
    top=0.86,
    bottom=0.22,
    wspace=0.30
)


# ============================================================
# Save
# ============================================================
pdf_path = os.path.join(plot_dir, "text8_threshold_sweep_four_panels.pdf")
png_path = os.path.join(plot_dir, "text8_threshold_sweep_four_panels.png")

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nSaved:", pdf_path)
print("Saved:", png_path)
# %%