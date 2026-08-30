# ============================================================
# plot_3DMMS_seed8_error_5x4.py
# 3D+time MMS - absolute error comparison for paper
# Seed = 8
#
# Layout: 5 rows (methods) x 4 columns (variables u, v, w, h)
#
#                 u             v             w             h
# PINN        [error]       [error]       [error]       [error]
# PCGrad      [error]       [error]       [error]       [error]
# GradNorm    [error]       [error]       [error]       [error]
# MOO-VARI    [error]       [error]       [error]       [error]
# Bi-GS-PINN  [error]       [error]       [error]       [error]
#
# Slice: p = 0.5, t = 0.5, grid 201 x 201 (deterministic, no resampling)
# Error: |u_pred - u_exact| per variable (absolute)
#
# Two colorbar variants (both saved, png + pdf):
#   1) independent : each panel uses its own vmax (= error.max())
#   2) shared      : all 20 panels share the global vmax
#
# Design unified with the paper's other figures
# (Fisher-KPP error_5x3): jet colormap, per-panel colorbars,
# rotated (a)-(e) row labels, math column titles, log-free panels.
#
# Models: plt/3DMMS/results/{strategy_folder}/seed_8/model_final.pth
# ============================================================


import os
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ============================================================
# 1. Configuration
# ============================================================

SEED = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Domain (identical to training code)
# ------------------------------------------------------------

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
P_MIN, P_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# ------------------------------------------------------------
# MMS parameters (identical to training code)
# ------------------------------------------------------------

A = 1.0
B = 1.0
C = 0.1
D = 1.0

ALPHA_MMS = 0.1
BETA_MMS = 0.1
GAMMA_MMS = 0.1

# ------------------------------------------------------------
# Network architecture (identical to training code)
# ------------------------------------------------------------

LAYERS = [4, 64, 64, 64, 4]

# ------------------------------------------------------------
# Visualization slice and grid
# ------------------------------------------------------------

P_SLICE = 0.5
T_SLICE = 0.5

NX = 201
NY = 201


# ============================================================
# 2. Methods and directories
# ============================================================

METHODS = [
    "PINN",
    "PCGrad",
    "GradNorm",
    "MOO-VARI",
    "Bi-GS-PINN",
]

# Display name -> results folder prefix
FOLDER_PREFIX = {
    "PINN": "standard",
    "PCGrad": "pcgrad",
    "GradNorm": "gradnorm",
    "MOO-VARI": "moo_vari",
    "Bi-GS-PINN": "bi_gs",
}

# Row labels (same convention as Fisher-KPP error_5x3)
ROW_LABELS = {
    "PINN": "(a) PINN",
    "PCGrad": "(b) PCGrad",
    "GradNorm": "(c) GradNorm",
    "MOO-VARI": "(d) MOO-VARI",
    "Bi-GS-PINN": "(e) Bi-GS-PINN",
}

VARIABLES = ["u", "v", "w", "h"]

VARIABLE_TITLES = {
    "u": r"$u$",
    "v": r"$v$",
    "w": r"$w$",
    "h": r"$h$",
}

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

os.makedirs(SAVE_DIR, exist_ok=True)


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
# 3. Exact MMS solution (numpy)
# ============================================================

def exact_solution(X_np):
    """
    Exact manufactured solution, shape (N, 4), columns u,v,w,h.
    X_np: shape (N, 4), columns x,y,p,t.
    """
    x = X_np[:, 0:1]
    y = X_np[:, 1:2]
    p = X_np[:, 2:3]
    t = X_np[:, 3:4]

    u = (
        A * np.sin(np.pi * x) * np.cos(np.pi * y)
        * np.exp(-t) * np.exp(-ALPHA_MMS * p)
    )
    v = (
        B * np.cos(np.pi * x) * np.sin(np.pi * y)
        * np.exp(-t) * np.exp(-BETA_MMS * p)
    )
    w = (
        C * np.sin(np.pi * p) * np.cos(np.pi * x) * np.sin(np.pi * y)
        * np.exp(-t)
    )
    h = (
        D * np.cos(np.pi * x) * np.cos(np.pi * y)
        * np.exp(-t) * np.exp(-GAMMA_MMS * p)
    )

    return np.hstack([u, v, w, h])


# ============================================================
# 4. PINN network (identical to training code)
# ============================================================

class PINN(nn.Module):

    def __init__(self, layers):

        super().__init__()

        self.lb = torch.tensor(
            [X_MIN, Y_MIN, P_MIN, T_MIN],
            dtype=torch.float32,
            device=DEVICE
        )

        self.ub = torch.tensor(
            [X_MAX, Y_MAX, P_MAX, T_MAX],
            dtype=torch.float32,
            device=DEVICE
        )

        modules = []

        for i in range(len(layers) - 1):
            modules.append(
                nn.Linear(layers[i], layers[i + 1])
            )
            if i != len(layers) - 2:
                modules.append(nn.Tanh())

        self.net = nn.Sequential(*modules)

    def forward(self, X):

        # Same normalization as training: [0,1]^4 -> [-1,1]^4
        X_norm = (
            2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        )

        return self.net(X_norm)


# ============================================================
# 5. Build deterministic slice grid
# ============================================================

print("=" * 70)
print(f"3D+time MMS absolute error comparison (Seed {SEED})")
print(f"Slice: p = {P_SLICE}, t = {T_SLICE}, grid {NX} x {NY}")
print("=" * 70)

x_plot = np.linspace(X_MIN, X_MAX, NX)
y_plot = np.linspace(Y_MIN, Y_MAX, NY)

X_GRID, Y_GRID = np.meshgrid(x_plot, y_plot, indexing="ij")

X_TEST = np.column_stack([
    X_GRID.reshape(-1),
    Y_GRID.reshape(-1),
    np.full(X_GRID.size, P_SLICE),
    np.full(X_GRID.size, T_SLICE),
])

EXACT_ALL = exact_solution(X_TEST)

EXACT_GRID = {
    variable: EXACT_ALL[:, i].reshape(NX, NY)
    for i, variable in enumerate(VARIABLES)
}

X_TEST_TENSOR = torch.tensor(
    X_TEST,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 6. Load models, predict, compute errors
# ============================================================

ERRORS = {}

REL_L2 = {}

for method in METHODS:

    strategy_dir = find_strategy_dir(FOLDER_PREFIX[method])

    model_path = os.path.join(
        RESULTS_DIR,
        strategy_dir,
        f"seed_{SEED}",
        "model_final.pth"
    )

    print(f"\n[{method}]")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\n找不到模型文件:\n{model_path}"
        )

    model = PINN(LAYERS).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    # 兼容裸 state_dict 或包裹格式
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        prediction = model(X_TEST_TENSOR).cpu().numpy()

    ERRORS[method] = {}
    REL_L2[method] = {}

    for i, variable in enumerate(VARIABLES):

        pred_grid = prediction[:, i].reshape(NX, NY)
        exact_grid = EXACT_GRID[variable]

        error_grid = np.abs(pred_grid - exact_grid)

        ERRORS[method][variable] = error_grid

        # 切片相对 L2（与训练代码同定义，+1e-12 防除零）
        rel_l2 = (
            np.linalg.norm(error_grid)
            / (np.linalg.norm(exact_grid) + 1e-12)
        )

        REL_L2[method][variable] = rel_l2

        print(
            f"  {variable}: max|err| = {error_grid.max():.3e}, "
            f"mean = {error_grid.mean():.3e}, rel L2 = {rel_l2:.3e}"
        )

# 共享 vmax：全部 20 格的最大误差
shared_vmax = max(
    ERRORS[m][v].max()
    for m in METHODS
    for v in VARIABLES
)

print(f"\nShared vmax (all panels) = {shared_vmax:.6e}")


# ============================================================
# 7. Draw error 5x4 figure
# ============================================================

def draw_error_figure(mode):
    """
    mode = "independent" -> each panel its own vmax
    mode = "shared"      -> all 20 panels share the global vmax
    """

    fig, axes = plt.subplots(
        nrows=5,
        ncols=4,
        figsize=(12, 15),
        constrained_layout=False
    )

    for i, method in enumerate(METHODS):

        for j, variable in enumerate(VARIABLES):

            ax = axes[i, j]

            error = ERRORS[method][variable]

            # ------------------------------------------------
            # Color scale
            # ------------------------------------------------

            error_vmin = 0.0

            if mode == "shared":
                error_vmax = shared_vmax
            else:
                error_vmax = error.max()

            if error_vmax <= error_vmin:
                error_vmax = 1e-12

            im = ax.pcolormesh(
                X_GRID,
                Y_GRID,
                error,
                shading="auto",
                cmap="jet",
                vmin=error_vmin,
                vmax=error_vmax
            )

            # ------------------------------------------------
            # Column titles (top row only)
            # ------------------------------------------------

            if i == 0:
                ax.set_title(
                    VARIABLE_TITLES[variable],
                    fontsize=15,
                    pad=8
                )

            # ------------------------------------------------
            # Axis labels
            # ------------------------------------------------

            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("y", fontsize=9)

            ax.tick_params(labelsize=7)

            # ------------------------------------------------
            # Colorbar (per panel, scientific ticks)
            # ------------------------------------------------

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))

            cbar.ax.yaxis.set_major_formatter(formatter)
            cbar.ax.tick_params(labelsize=7)

        # ------------------------------------------------
        # Row label
        # ------------------------------------------------

        axes[i, 0].text(
            -0.34,
            0.5,
            ROW_LABELS[method],
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

    # ------------------------------------------------
    # Figure spacing
    # ------------------------------------------------

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.965,
        bottom=0.035,
        wspace=0.34,
        hspace=0.62
    )

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    pdf_path = os.path.join(
        SAVE_DIR,
        f"3DMMS_seed{SEED}_error_5x4_{mode}.pdf"
    )

    png_path = os.path.join(
        SAVE_DIR,
        f"3DMMS_seed{SEED}_error_5x4_{mode}.png"
    )

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"\nSaved ({mode}):")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


draw_error_figure("independent")
draw_error_figure("shared")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
