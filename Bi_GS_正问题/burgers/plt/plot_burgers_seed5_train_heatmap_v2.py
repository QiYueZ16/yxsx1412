# ============================================================
# plot_burgers_seed5_train_heatmap_v2.py
# Burgers equation - training-grid heatmap (paper style)
# Seed = 5
# Output: burgers_seed{SEED}_train_heatmap.pdf / .png
# ============================================================


import os
import re
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


# ============================================================
# 1. Configuration
# ============================================================

SEED = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Burgers domain
X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0

# Network architecture (identical to training code)
LAYERS = [2, 64, 64, 64, 1]

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    CURRENT_DIR,
    "burgers_shock.mat"
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

# Row labels (same convention as the other paper figures)
ROW_LABELS = {
    "PINN": "(a) PINN",
    "PCGrad": "(b) PCGrad",
    "GradNorm": "(c) GradNorm",
    "MOO-VARI": "(d) MOO-VARI",
    "Bi-GS-PINN": "(e) Bi-GS-PINN",
}

# Column titles
COLUMN_TITLES = [
    "Reference",
    "Prediction",
    "Absolute Error",
]


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
# 3. PINN network (identical to training code)
# ============================================================

class PINN(nn.Module):

    def __init__(self, layers, inverse=False):

        super().__init__()

        self.lb = torch.tensor(
            [X_MIN, T_MIN],
            dtype=torch.float32,
            device=DEVICE
        )

        self.ub = torch.tensor(
            [X_MAX, T_MAX],
            dtype=torch.float32,
            device=DEVICE
        )

        self.inverse = inverse

        modules = []

        for i in range(len(layers) - 1):
            modules.append(
                nn.Linear(layers[i], layers[i + 1])
            )
            if i != len(layers) - 2:
                modules.append(nn.Tanh())

        self.net = nn.Sequential(*modules)

        if inverse:

            self.nu = nn.Parameter(
                torch.tensor(
                    0.03,
                    dtype=torch.float32
                )
            )

        else:

            self.nu = torch.tensor(
                0.01 / np.pi,
                dtype=torch.float32,
                device=DEVICE
            )

    def forward(self, x):

        # Same normalization as training: [-1,1]x[0,1] -> [-1,1]^2
        x_norm = (
            2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        )

        return self.net(x_norm)


# ============================================================
# 4. Load reference solution (RAW .mat grid, like training)
# ============================================================

print("=" * 70)
print("Loading Burgers reference solution (training grid)...")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n找不到 burgers_shock.mat:\n{DATA_PATH}"
    )

data = scipy.io.loadmat(DATA_PATH)

x_ref = data["x"].flatten()
t_ref = data["t"].flatten()
usol = np.real(data["usol"])

print(
    f"x shape: {x_ref.shape}, "
    f"t shape: {t_ref.shape}, "
    f"usol shape: {usol.shape}"
)

# Same grid construction as the training code:
# np.meshgrid(x_ref, t_ref, indexing="ij")  -> (256, 100)
X_MESH, T_MESH = np.meshgrid(x_ref, t_ref, indexing="ij")

# Raw exact values (no interpolation)
EXACT = usol

# Test points (same order as training: hstack of flattened x, t)
X_TEST_NP = np.hstack(
    [X_MESH.reshape(-1, 1), T_MESH.reshape(-1, 1)]
)

X_TEST_TENSOR = torch.tensor(
    X_TEST_NP,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 5. Load models, predict, compute errors
# ============================================================

PREDICTIONS = {}
ERRORS = {}

print("\n" + "=" * 70)
print("Loading models (same evaluation as training)...")
print("=" * 70)

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
        pred = model(X_TEST_TENSOR).cpu().numpy().reshape(EXACT.shape)

    error = np.abs(pred - EXACT)

    PREDICTIONS[method] = pred
    ERRORS[method] = error

    print(
        f"Prediction range: [{pred.min():.6f}, {pred.max():.6f}]"
    )

    print(
        f"Max abs error: {error.max():.6e} "
        f"| Mean abs error: {error.mean():.6e}"
    )


# ============================================================
# 6. Draw training-grid heatmap (5 x 3, independent scales)
# ============================================================

print("\n" + "=" * 70)
print("Drawing figure...")
print("=" * 70)

fig, axes = plt.subplots(
    nrows=5,
    ncols=3,
    figsize=(13, 19),
    constrained_layout=False
)

for i, method in enumerate(METHODS):

    pred = PREDICTIONS[method]
    error = ERRORS[method]

    # ------------------------------------------------
    # 1. Reference (same field in every row)
    # ------------------------------------------------

    ax_ref = axes[i, 0]

    ref_vmin = EXACT.min()
    ref_vmax = EXACT.max()

    im_ref = ax_ref.pcolormesh(
        T_MESH,
        X_MESH,
        EXACT,
        shading="auto",
        cmap="jet",
        vmin=ref_vmin,
        vmax=ref_vmax
    )

    ax_ref.set_title(
        COLUMN_TITLES[0],
        fontsize=12,
        pad=6
    )

    ax_ref.set_xlabel("t", fontsize=10)
    ax_ref.set_ylabel("x", fontsize=10)
    ax_ref.tick_params(labelsize=8)

    cbar_ref = fig.colorbar(
        im_ref,
        ax=ax_ref,
        fraction=0.046,
        pad=0.04
    )

    cbar_ref.ax.tick_params(labelsize=7)

    # ------------------------------------------------
    # 2. Prediction (own range, like training)
    # ------------------------------------------------

    ax_pred = axes[i, 1]

    pred_vmin = pred.min()
    pred_vmax = pred.max()

    if pred_vmax <= pred_vmin:
        pred_vmax = pred_vmin + 1e-12

    im_pred = ax_pred.pcolormesh(
        T_MESH,
        X_MESH,
        pred,
        shading="auto",
        cmap="jet",
        vmin=pred_vmin,
        vmax=pred_vmax
    )

    ax_pred.set_title(
        COLUMN_TITLES[1],
        fontsize=12,
        pad=6
    )

    ax_pred.set_xlabel("t", fontsize=10)
    ax_pred.set_ylabel("x", fontsize=10)
    ax_pred.tick_params(labelsize=8)

    cbar_pred = fig.colorbar(
        im_pred,
        ax=ax_pred,
        fraction=0.046,
        pad=0.04
    )

    cbar_pred.ax.tick_params(labelsize=7)

    # ------------------------------------------------
    # 3. Absolute Error (own range, like training)
    # ------------------------------------------------

    ax_error = axes[i, 2]

    error_vmin = 0.0
    error_vmax = error.max()

    if error_vmax <= error_vmin:
        error_vmax = 1e-12

    im_error = ax_error.pcolormesh(
        T_MESH,
        X_MESH,
        error,
        shading="auto",
        cmap="jet",
        vmin=error_vmin,
        vmax=error_vmax
    )

    ax_error.set_title(
        COLUMN_TITLES[2],
        fontsize=12,
        pad=6
    )

    ax_error.set_xlabel("t", fontsize=10)
    ax_error.set_ylabel("x", fontsize=10)
    ax_error.tick_params(labelsize=8)

    cbar_error = fig.colorbar(
        im_error,
        ax=ax_error,
        fraction=0.046,
        pad=0.04
    )

    # 误差色标：最后一行 (Bi-GS) 用纯小数格式（0.008 那样，
    # 与训练热图一致）；其余行沿用科学计数法自动格式
    if i == len(METHODS) - 1:
        cbar_error.ax.yaxis.set_major_formatter(
            FormatStrFormatter('%.3f')
        )
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        cbar_error.ax.yaxis.set_major_formatter(formatter)

    cbar_error.ax.tick_params(labelsize=7)

    # ------------------------------------------------
    # 4. Row label (LEFT side)
    # ------------------------------------------------

    axes[i, 0].text(
        -0.34,
        0.5,
        ROW_LABELS[method],
        transform=axes[i, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

# ------------------------------------------------
# Spacing
# ------------------------------------------------

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.965,
    bottom=0.035,
    wspace=0.5,
    hspace=0.5
)

# ------------------------------------------------
# Save
# ------------------------------------------------

pdf_path = os.path.join(
    SAVE_DIR,
    f"burgers_seed{SEED}_train_heatmap_v2.pdf"
)

png_path = os.path.join(
    SAVE_DIR,
    f"burgers_seed{SEED}_train_heatmap_v2.png"
)

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

plt.close(fig)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

print(f"\nSaved: {pdf_path}")
print(f"       {png_path}")
