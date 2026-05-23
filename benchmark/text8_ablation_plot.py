import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
# ==============================
# Paths
# ==============================
sleep_path = "../pickle_files/text8_sleep_ablation_partial/sleep_partial.pkl"
nosleep_path = "../pickle_files/text8_sleep_ablation_partial_sleepless_again/no_sleep_partial.pkl"

# ==============================
# Load + sort
# ==============================
df_sleep = pd.read_pickle(sleep_path).sort_values("samples seen")
df_nosleep = pd.read_pickle(nosleep_path).sort_values("samples seen")

# ==============================
# Moving average (clean smoothing)
# ==============================
def moving_avg(x, w=7):
    return np.convolve(x, np.ones(w)/w, mode='valid')

window = 20

def smooth_curve(df, key):
    x = df["samples seen"].values
    y = df[key].values
    y_s = moving_avg(y, window)
    x_s = x[window-1:]
    return x_s, y_s

# choose metric
metric = "eval_bpc"   # or "train_bpc_window"

x_s, y_s = smooth_curve(df_sleep, metric)
x_n, y_n = smooth_curve(df_nosleep, metric)

# ==============================
# Plot
# ==============================
plt.figure(figsize=(7, 5))

plt.plot(x_s, y_s, linewidth=2.5, c='r', label="Sleep")
plt.plot(x_n, y_n, linewidth=2.5, c='b', label="No Sleep")

plt.xlabel("Samples Seen", fontsize=20)
plt.ylabel("Bits per Token (BPC)", fontsize=20)

plt.title("Sleep vs No Sleep (Text8)", fontsize=22)

plt.legend(frameon=False)

# remove top/right borders (clean look)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig("../plots/text8_sleep_ablation.pdf")
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_context('talk')

# ==============================
# Paths
# ==============================
sleep_path = "../pickle_files/text8_sleep_ablation_partial/sleep_partial.pkl"
nosleep_path = "../pickle_files/text8_sleep_ablation_partial_sleepless_again/no_sleep_partial.pkl"
no_slow_path = "../pickle_files/text8_no_slow_heads_only_partial/no_slow_heads_partial.pkl"

# ==============================
# Load + sort
# ==============================
df_sleep = pd.read_pickle(sleep_path).sort_values("samples seen")
df_nosleep = pd.read_pickle(nosleep_path).sort_values("samples seen")

df_no_slow = None
if os.path.exists(no_slow_path):
    df_no_slow = pd.read_pickle(no_slow_path).sort_values("samples seen")
    print("Loaded no-slow-heads partial:")
    print(df_no_slow.tail())
else:
    print(f"No no-slow-heads partial file found at: {no_slow_path}")

# ==============================
# Moving average
# ==============================
def moving_avg(x, w=7):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode='valid')

window = 20

def smooth_curve(df, key, window=20):
    x = df["samples seen"].values
    y = df[key].values

    if len(y) < window:
        return x, y

    y_s = moving_avg(y, window)
    x_s = x[window - 1:]

    return x_s, y_s

# choose metric
metric = "eval_bpc"   # or "train_bpc_window"

x_s, y_s = smooth_curve(df_sleep, metric, window)
x_n, y_n = smooth_curve(df_nosleep, metric, window)

if df_no_slow is not None:
    x_ns, y_ns = smooth_curve(df_no_slow, metric, window)

# ==============================
# Plot
# ==============================
plt.figure(figsize=(7, 5))

plt.plot(x_s, y_s, linewidth=2.5, c='r', label="Sleep")
plt.plot(x_n, y_n, linewidth=2.5, c='b', label="No Sleep")

if df_no_slow is not None:
    plt.plot(
        x_ns,
        y_ns,
        linewidth=2.5,
        c='g',
        label="No Pattern Slowdown"
    )

plt.xlabel("Samples Seen", fontsize=20)
plt.ylabel("Bits per Token (BPC)", fontsize=20)
plt.title("Sleep Ablations on Text8", fontsize=22)

plt.legend(frameon=False)

# remove top/right borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

os.makedirs("../plots", exist_ok=True)
plt.savefig("../plots/text8_sleep_ablation_with_no_slow_heads.pdf")
plt.show()
# %%
