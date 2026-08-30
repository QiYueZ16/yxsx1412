# ============================================================
# plot_burgers_seed5_weights.py
# Burgers equation - loss weight evolution (paper style)
# Seed = 5
#
# Data source:
#   plt/burgers/results/{strategy_folder}/seed_5/weight_history.npz
#   keys: iteration, weights (n_points, 3)  [λ_pde, λ_bc, λ_ic]
#
# Figure: 1x2 subplots
#   (a) Bi-GS-PINN  - adaptive OAW weights, sum = 1
#   (b) PINN        - standard, constant equal weights [1/3, 1/3, 1/3]
#
# Adam phase only (iterations 0..10000; training config A10000_L10000),
# dashed divider at x = 10000 (Adam -> L-BFGS boundary, no text).
# Style: linear y, grid, frameoff legend.
# Output: burgers_seed{SEED}_weight_evolution.pdf / .png
# ============================================================


import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

SEED = 5

# Panel order: display name -> results folder prefix
PANELS = [
    ("Bi-GS-PINN", "bi_gs"),
    ("PINN", "standard"),
]

# Component colors (tab10), consistent across all figures
COLORS = {
    "lambda_pde": "#1f77b4",
    "lambda_bc": "#d62728",
    "lambda_ic": "#9467bd",
}

COMPONENT_LABELS = [
    ("lambda_pde", "λ_pde"),
    ("lambda_bc", "λ_BC"),
    ("lambda_ic", "λ_IC"),
]

# weights 列序固定为 [λ_pde, λ_bc, λ_ic]
COMPONENT_INDEX = {
    "lambda_pde": 0,
    "lambda_bc": 1,
    "lambda_ic": 2,
}

# Shared x range (Adam phase only; adam_epochs = 10000 in config A10000_L10000)
X_LIM = [0.0, 10000.0]

ADAM_EPOCHS = 10000

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    CURRENT_DIR,
    "results"
)

SAVE_DIR = os.path.join(
    CURRENT_DIR,
    "figures"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


def find_strategy_dir(prefix):
    """在 results/ 下按前缀找到实际策略文件夹（目录名带时间戳）"""
    for name in sorted(os.listdir(RESULTS_DIR)):
        if re.match(rf"^{prefix}_", name) and os.path.isdir(
                os.path.join(RESULTS_DIR, name)):
            return name
    raise FileNotFoundError(
        f"\n找不到策略文件夹: 前缀 '{prefix}_'\n"
        f"请检查 results 目录。"
    )


# ============================================================
# 2. Load weight histories
# ============================================================

print("=" * 70)
print(f"Burgers weight evolution (Seed {SEED})")
print("=" * 70)

histories = {}

for display_name, prefix in PANELS:

    strategy_dir = find_strategy_dir(prefix)

    npz_path = os.path.join(
        RESULTS_DIR,
        strategy_dir,
        f"seed_{SEED}",
        "weight_history.npz"
    )

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"\n找不到数据文件:\n{npz_path}"
        )

    data = np.load(npz_path)

    histories[display_name] = {
        "iteration": data["iteration"],
        "weights": data["weights"],
    }

    W = data["weights"]

    # Adam 段内最后一个记录点
    adam_mask = data["iteration"] <= ADAM_EPOCHS
    adam_last = int(data["iteration"][adam_mask][-1])

    print(
        f"\n[{display_name}]  shape = {W.shape}"
    )

    print(
        f"  iter 0     : {np.round(W[0], 4)}"
    )

    print(
        f"  Adam end   : {np.round(W[adam_mask][-1], 4)} (iter {adam_last})"
    )

    print(
        f"  final      : {np.round(W[-1], 4)}"
    )

    if display_name == "Bi-GS-PINN":
        s = W[adam_mask][-1].sum()
        print(
            f"  sum(Adam end) = {s:.4f}"
        )

    chg = sum(
        1 for i in range(1, len(W))
        if not np.allclose(W[i], W[i - 1])
    )

    print(
        f"  changes    : {chg}"
    )

# ============================================================
# 3. Common paper-style settings
# ============================================================

LEGEND_KW = dict(
    loc="best",
    frameon=False,
    fontsize=11,
)

GRID_KW = dict(
    which="both",
    linestyle=":",
    linewidth=0.6,
    alpha=0.4,
)


def style_axis(ax):
    ax.set_xlim(X_LIM)
    ax.set_xlabel(
        "Iteration",
        fontsize=12
    )
    ax.tick_params(
        labelsize=10
    )
    ax.grid(
        **GRID_KW
    )
    ax.legend(
        **LEGEND_KW
    )


def add_adam_lbfgs_divider(ax):
    """x = ADAM_EPOCHS 处灰色虚线（Adam 段结束 = L-BFGS 开始）"""
    ax.axvline(
        ADAM_EPOCHS,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )


# ============================================================
# 4. Draw figure (1 x 2)
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Drawing weight evolution figure..."
)

print(
    "=" * 70
)

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 4.8)
)

for col, (display_name, prefix) in enumerate(
    PANELS
):

    ax = axes[col]

    h = histories[display_name]

    for key, label in COMPONENT_LABELS:

        ax.plot(
            h["iteration"],
            h["weights"][:, COMPONENT_INDEX[key]],
            color=COLORS[key],
            label=label,
            lw=1.8
        )

    ax.set_ylim(0.0, 0.7)

    ax.set_ylabel(
        "Loss weight λ",
        fontsize=12
    )

    ax.set_title(
        f"({chr(97 + col)}) {display_name}",
        fontsize=13,
        fontweight="bold"
    )

    style_axis(ax)

    add_adam_lbfgs_divider(ax)

plt.tight_layout()

pdf_path = os.path.join(
    SAVE_DIR,
    f"burgers_seed{SEED}_weight_evolution.pdf"
)

png_path = os.path.join(
    SAVE_DIR,
    f"burgers_seed{SEED}_weight_evolution.png"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"\nSaved: {pdf_path}"
)

print(
    f"       {png_path}"
)

print(
    "\n" + "=" * 70
)

print(
    "DONE!"
)

print(
    "=" * 70
)
