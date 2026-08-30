# ============================================================
# plot_burgers_seed5_compare.py
# Burgers equation - loss / L2 convergence comparison (paper style)
# Seed = 5
#
# Data source:
#   plt/burgers/results/{strategy_folder}/seed_5/training_history.npz
#   keys: iteration, loss, l2, strategy, seed, adam_epochs, lbfgs_max_iter
#
# Figures (all png 300dpi + pdf):
#   1) burgers_seed{SEED}_loss_l2_compare   (1x2 subplots: Loss | L2)
#   2) burgers_seed{SEED}_loss_compare      (Loss only)
#   3) burgers_seed{SEED}_l2_compare        (L2 only)
#
# Style: log y axes, xlim [0, 21000], dashed divider at Adam->L-BFGS,
#        tab10 colors, frameoff legend (paper style).
# ============================================================


import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

SEED = 5

# Display name -> results folder prefix
FOLDER_PREFIX = {
    "PINN": "standard",
    "PCGrad": "pcgrad",
    "GradNorm": "gradnorm",
    "MOO-VARI": "moo_vari",
    "Bi-GS-PINN": "bi_gs",
}

# Plot colors (tab10), consistent across all figures
COLORS = {
    "PINN": "#1f77b4",
    "PCGrad": "#ff7f0e",
    "GradNorm": "#2ca02c",
    "MOO-VARI": "#d62728",
    "Bi-GS-PINN": "#9467bd",
}

METHODS = list(FOLDER_PREFIX.keys())

# Shared x range (moo_vari's L-BFGS tail reaches iteration 20400)
X_LIM = [0.0, 21000.0]

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
# 2. Load training histories
# ============================================================

print("=" * 70)
print(f"Burgers loss / L2 comparison (Seed {SEED})")
print("=" * 70)

histories = {}

for method in METHODS:

    strategy_dir = find_strategy_dir(
        FOLDER_PREFIX[method]
    )

    npz_path = os.path.join(
        RESULTS_DIR,
        strategy_dir,
        f"seed_{SEED}",
        "training_history.npz"
    )

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"\n找不到数据文件:\n{npz_path}"
        )

    data = np.load(npz_path)

    histories[method] = {
        "iteration": data["iteration"],
        "loss": data["loss"],
        "l2": data["l2"],
        "adam_epochs": int(data["adam_epochs"]),
    }

    print(
        f"[{method:10s}] "
        f"iters {data['iteration'][0]:5d}-{data['iteration'][-1]:5d} "
        f"| final loss = {data['loss'][-1]:.4e} "
        f"| final L2 = {data['l2'][-1]:.4e}"
    )

# 分割线位置：Adam 阶段步数（五策略一致，取第一个即代表）
adam_epochs = histories[METHODS[0]]["adam_epochs"]

print(
    f"\nAdam phase = {adam_epochs} iterations "
    f"(divider at x={adam_epochs})"
)

# ============================================================
# 3. Common paper-style plot settings
# ============================================================

PLOT_KW = dict(
    lw=1.8,
)

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
    """x = adam_epochs 处灰色虚线 + 阶段标注"""
    ax.axvline(
        adam_epochs,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )
    ax.text(
        adam_epochs + 250,
        ax.get_ylim()[1],
        "Adam → L-BFGS",
        fontsize=10,
        color="dimgray",
        va="top"
    )


# ============================================================
# 4. Draw single-axis figure (loss or l2)
# ============================================================

def draw_single_figure(
    y_key,
    y_label,
    out_name
):
    """画一张独立对比图（loss 或 l2）"""

    fig, ax = plt.subplots(
        figsize=(7.5, 5.2)
    )

    for method in METHODS:

        h = histories[method]

        ax.semilogy(
            h["iteration"],
            h[y_key],
            color=COLORS[method],
            label=method,
            **PLOT_KW
        )

    ax.set_ylabel(
        y_label,
        fontsize=12
    )

    style_axis(ax)

    add_adam_lbfgs_divider(ax)

    pdf_path = os.path.join(
        SAVE_DIR,
        f"{out_name}.pdf"
    )

    png_path = os.path.join(
        SAVE_DIR,
        f"{out_name}.png"
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
        f"\nSaved: {pdf_path}\n       {png_path}"
    )


# ============================================================
# 5. Draw combined figure (loss | l2, one fig two panels)
# ============================================================

def draw_combined_figure():

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.8)
    )

    for method in METHODS:

        h = histories[method]

        # (a) Loss
        axes[0].semilogy(
            h["iteration"],
            h["loss"],
            color=COLORS[method],
            label=method,
            **PLOT_KW
        )

        # (b) L2
        axes[1].semilogy(
            h["iteration"],
            h["l2"],
            color=COLORS[method],
            label=method,
            **PLOT_KW
        )

    # (a) Loss panel
    axes[0].set_ylabel(
        "Loss",
        fontsize=12
    )

    style_axis(axes[0])

    add_adam_lbfgs_divider(axes[0])

    axes[0].text(
        -0.12,
        1.05,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="right"
    )

    # (b) L2 panel
    axes[1].set_ylabel(
        "L2 Relative Error",
        fontsize=12
    )

    style_axis(axes[1])

    add_adam_lbfgs_divider(axes[1])

    axes[1].text(
        -0.12,
        1.05,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="right"
    )

    plt.tight_layout()

    pdf_path = os.path.join(
        SAVE_DIR,
        f"burgers_seed{SEED}_loss_l2_compare.pdf"
    )

    png_path = os.path.join(
        SAVE_DIR,
        f"burgers_seed{SEED}_loss_l2_compare.png"
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
        f"\nSaved: {pdf_path}\n       {png_path}"
    )


# ============================================================
# 6. Draw everything
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Drawing figures..."
)

print(
    "=" * 70
)

draw_combined_figure()

draw_single_figure(
    y_key="loss",
    y_label="Loss",
    out_name=f"burgers_seed{SEED}_loss_compare"
)

draw_single_figure(
    y_key="l2",
    y_label="L2 Relative Error",
    out_name=f"burgers_seed{SEED}_l2_compare"
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

print(
    f"\nFigures saved to: {SAVE_DIR}"
)
