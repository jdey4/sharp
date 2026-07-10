#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import (
    FormatStrFormatter,
    FixedLocator,
    LogFormatterMathtext,
    NullLocator,
)


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

# Optional: use an existing run for tau=1e-2
tau_1e2_existing_path = None

# Example:
# tau_1e2_existing_path = (
#     "../pickle_files/"
#     "text8_sleep_ablation_partial_threeway/"
#     "sleep_partial.pkl"
# )


# ============================================================
# Load threshold partial files
# ============================================================
def load_threshold_sweep(partial_root):
    dfs = []

    paths = sorted(
        glob.glob(
            os.path.join(
                partial_root,
                "*",
                "*_partial.pkl",
            )
        )
    )

    if len(paths) == 0:
        raise RuntimeError(
            f"No threshold partial files found in: {partial_root}"
        )

    for path in paths:
        try:
            df_temp = (
                pd.read_pickle(path)
                .sort_values("samples seen")
                .reset_index(drop=True)
            )

            if len(df_temp) == 0:
                print(f"Skipping empty file: {path}")
                continue

            required_columns = [
                "samples seen",
                "threshold",
                "threshold_tag",
                "memory_update_percent_window",
                "forward_bpc",
                "current_bpc",
                "backward_bpc",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df_temp.columns
            ]

            if missing_columns:
                print(
                    f"Skipping {path}: "
                    f"missing columns {missing_columns}"
                )
                continue

            dfs.append(df_temp)

            print(f"\nLoaded: {path}")
            print(
                df_temp[
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

        except Exception as error:
            print(f"Could not load {path}: {error}")

    if len(dfs) == 0:
        raise RuntimeError(
            f"No usable threshold partial files found in: {partial_root}"
        )

    return pd.concat(
        dfs,
        ignore_index=True,
    )


df = load_threshold_sweep(partial_root)


# ============================================================
# Optionally add tau=1e-2 from an existing run
# ============================================================
if (
    tau_1e2_existing_path is not None
    and os.path.exists(tau_1e2_existing_path)
):
    df_tau = (
        pd.read_pickle(tau_1e2_existing_path)
        .sort_values("samples seen")
        .reset_index(drop=True)
    )

    required_bpc_columns = [
        "forward_bpc",
        "current_bpc",
        "backward_bpc",
    ]

    if all(
        column in df_tau.columns
        for column in required_bpc_columns
    ):
        df_tau = df_tau.copy()

        df_tau["threshold"] = 1e-2
        df_tau["threshold_tag"] = "tau_1em02"

        if "memory_update_percent_window" not in df_tau.columns:
            df_tau["memory_update_percent_window"] = np.nan

        df = pd.concat(
            [df, df_tau],
            ignore_index=True,
        )

        print(
            "\nAdded existing tau=1e-2 run: "
            f"{tau_1e2_existing_path}"
        )

    else:
        print(
            "\nExisting tau=1e-2 file lacks "
            "three-way BPC columns: "
            f"{tau_1e2_existing_path}"
        )


# ============================================================
# Threshold labels
# ============================================================
def threshold_label(tau):
    tau = float(tau)

    if tau == 0.0:
        return r"$\tau=0$"

    exponent = int(
        np.round(
            np.log10(tau)
        )
    )

    return rf"$\tau=10^{{{exponent}}}$"


thresholds = sorted(
    df["threshold"]
    .dropna()
    .unique()
)

labels = {
    tau: threshold_label(tau)
    for tau in thresholds
}

print("\nThresholds found:")

for tau in thresholds:
    print(f"{tau} -> {labels[tau]}")


# ============================================================
# Smoothing without shifting x-axis
# ============================================================
window = 30


def smooth_curve_no_shift(x, y, window=30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite_mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[finite_mask]
    y = y[finite_mask]

    if len(y) == 0:
        return x, y

    y_smoothed = (
        pd.Series(y)
        .rolling(
            window=window,
            min_periods=1,
            center=True,
        )
        .mean()
        .to_numpy()
    )

    return x, y_smoothed


# ============================================================
# Plot configuration
# ============================================================
panels = [
    (
        "memory_update_percent_window",
        "Wake Memory Update",
    ),
    (
        "forward_bpc",
        "Forward",
    ),
    (
        "current_bpc",
        "Current",
    ),
    (
        "backward_bpc",
        "Backward",
    ),
]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(28, 5),
    sharex=True,
)

palette = sns.color_palette(
    "tab10",
    n_colors=len(thresholds),
)

color_map = {
    tau: palette[index]
    for index, tau in enumerate(thresholds)
}

positive_update_values = []


# ============================================================
# Draw curves
# ============================================================
for ax, (metric, title) in zip(axes, panels):

    for tau in thresholds:
        df_tau = (
            df[df["threshold"] == tau]
            .sort_values("samples seen")
            .reset_index(drop=True)
        )

        x = df_tau["samples seen"].to_numpy()
        y = df_tau[metric].to_numpy()

        x_smoothed, y_smoothed = smooth_curve_no_shift(
            x,
            y,
            window=window,
        )

        # Logarithmic axes cannot display zero or negative values.
        if metric == "memory_update_percent_window":
            positive_mask = (
                np.isfinite(y_smoothed)
                & (y_smoothed > 0)
            )

            positive_update_values.extend(
                y_smoothed[positive_mask]
            )

            y_to_plot = np.where(
                positive_mask,
                y_smoothed,
                np.nan,
            )

        else:
            y_to_plot = y_smoothed

        # Only the blue tau=0 curve in Panel 1 is thicker.
        is_panel_1_blue = (
            metric == "memory_update_percent_window"
            and np.isclose(float(tau), 0.0)
        )

        linewidth = 6.0 if is_panel_1_blue else 4.0
        zorder = 3 if is_panel_1_blue else 2

        ax.plot(
            x_smoothed,
            y_to_plot,
            linewidth=linewidth,
            color=color_map[tau],
            label=labels[tau],
            zorder=zorder,
        )

    # Increased gap between each title and its panel.
    ax.set_title(
        title,
        fontsize=32,
        pad=24,
    )

    ax.set_xticks(
        [
            0,
            5e7,
            1e8,
        ]
    )

    ax.set_xlabel("")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=27,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric == "memory_update_percent_window":
        ax.set_yscale(
            "log",
            base=10,
        )

        # Set exactly three ticks later.
        ax.yaxis.set_major_locator(
            NullLocator()
        )

        ax.yaxis.set_minor_locator(
            NullLocator()
        )

    else:
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )


# ============================================================
# Panel 1: exactly three logarithmic y-axis ticks
# ============================================================
if len(positive_update_values) > 0:
    positive_update_values = np.asarray(
        positive_update_values,
        dtype=float,
    )

    minimum_positive = np.nanmin(
        positive_update_values
    )

    maximum_positive = np.nanmax(
        positive_update_values
    )

    lower_exponent = np.floor(
        np.log10(minimum_positive)
    )

    upper_exponent = np.ceil(
        np.log10(maximum_positive)
    )

    log_lower_limit = 10.0 ** lower_exponent
    log_upper_limit = 10.0 ** upper_exponent

    # Ensure the plot reaches 100%.
    log_upper_limit = max(
        log_upper_limit,
        100.0,
    )

else:
    print(
        "\nWarning: no positive update-rate values were found. "
        "Using fallback logarithmic limits."
    )

    log_lower_limit = 1e-4
    log_upper_limit = 1e2


# Visual midpoint on a logarithmic axis.
log_middle_tick = np.sqrt(
    log_lower_limit * log_upper_limit
)

log_ticks = [
    log_lower_limit,
    log_middle_tick,
    log_upper_limit,
]

axes[0].set_ylim(
    log_lower_limit,
    log_upper_limit,
)

axes[0].yaxis.set_major_locator(
    FixedLocator(log_ticks)
)

axes[0].yaxis.set_major_formatter(
    LogFormatterMathtext(base=10)
)

axes[0].yaxis.set_minor_locator(
    NullLocator()
)


# ============================================================
# Axis labels
# ============================================================
axes[0].set_ylabel(
    "Update Rate (%)",
    fontsize=30,
)

axes[1].set_ylabel("")
axes[2].set_ylabel("")
axes[3].set_ylabel("")


# Common BPC label for Panels 2–4
fig.text(
    0.22,
    0.57,
    "BPC",
    va="center",
    rotation="vertical",
    fontsize=30,
)


# ============================================================
# Shared legend
# ============================================================
handles, legend_labels = (
    axes[0].get_legend_handles_labels()
)

legend = fig.legend(
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

legend.get_title().set_ha("center")

try:
    legend._legend_box.align = "center"
except Exception:
    pass


# ============================================================
# Global x-axis label and layout
# ============================================================
fig.supxlabel(
    "Samples Seen",
    fontsize=30,
    y=0.0,
)

plt.subplots_adjust(
    left=0.06,
    right=0.82,
    top=0.79,
    bottom=0.22,
    wspace=0.30,
)


# ============================================================
# Save
# ============================================================
pdf_path = os.path.join(
    plot_dir,
    "text8_threshold_sweep_four_panels_log_update.pdf",
)

png_path = os.path.join(
    plot_dir,
    "text8_threshold_sweep_four_panels_log_update.png",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print("\nSaved:", pdf_path)
print("Saved:", png_path)

# %%