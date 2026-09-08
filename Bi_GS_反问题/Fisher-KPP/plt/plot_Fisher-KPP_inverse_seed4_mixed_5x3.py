# -*- coding: utf-8 -*-
"""
plot_Fisher-KPP_inverse_seed4_mixed_5x3.py
反问题 2D Fisher-KPP 方程 - 论文用混合切片对比图 (Seed 4)
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 1. Configuration
# ============================================================

SEED = 4

DEVICE = torch.device("cpu")

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 0.4

LAYERS = [3, 64, 64, 64, 1]

NU_FISHER = 0.05  # 反问题真值 (解析解需要)
RHO = 20.0        # 反问题真值 (解析解需要)

# 第一列热图时刻: t = 0.20 (t_ref 节点)
T_HEAT = 0.2

# 第二/三列共享的切片值: c = 0.15 (fk_slice_l2_stats_seed4.py 选出)
SLICE_C = 0.15

PRED_BATCH_SIZE = 65536

# ============================================================
# 2. Directories 
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
SAVE_DIR = os.path.join(CURRENT_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 3. Fisher-KPP analytic solution 
# ============================================================

K_FISHER = np.sqrt(RHO / (6.0 * NU_FISHER))
C_WAVE = 5.0 * np.sqrt(NU_FISHER * RHO / 6.0)


def exact_solution_fisher(X_np):
    """X_np: (N, 3), 列顺序 (x, y, t) -> 返回 (N,)"""
    x = X_np[:, 0:1]
    y = X_np[:, 1:2]
    t = X_np[:, 2:3]
    z = (x + y) / np.sqrt(2.0)
    xi = K_FISHER * (z - C_WAVE * t)
    return (1.0 / (1.0 + np.exp(xi)) ** 2).reshape(-1)


# 网格与训练一致
x_ref = np.linspace(X_MIN, X_MAX, 65)
y_ref = np.linspace(Y_MIN, Y_MAX, 65)
t_ref = np.linspace(T_MIN, T_MAX, 21)

# 第一列: t = T_HEAT 时刻的 (x, y) 平面
X_MESH, Y_MESH = np.meshgrid(x_ref, y_ref, indexing="ij")
X_TEST_NP = np.hstack([
    X_MESH.reshape(-1, 1),
    Y_MESH.reshape(-1, 1),
    np.full(X_MESH.size, T_HEAT).reshape(-1, 1),
])
EXACT_HEAT = exact_solution_fisher(X_TEST_NP).reshape(X_MESH.shape)  # (65, 65)

# ============================================================
# 4. Methods and model paths
# ============================================================

METHODS = ["PINN", "PCGrad", "GradNorm", "MOO-VARI", "Bi-GS-PINN"]

FOLDER_PREFIX = {
    "PINN": "standard",
    "PCGrad": "pcgrad",
    "GradNorm": "gradnorm",
    "MOO-VARI": "moo_vari",
    "Bi-GS-PINN": "bi_gs",
}

ROW_LABELS = {
    "PINN": "(a) PINN",
    "PCGrad": "(b) PCGrad",
    "GradNorm": "(c) GradNorm",
    "MOO-VARI": "(d) MOO-VARI",
    "Bi-GS-PINN": "(e) Bi-GS-PINN",
}


PRED_COLOR = "#1f77b4"
EXACT_COLOR = "r"


def find_strategy_dir(prefix):
    """在 results/ 下按前缀找到实际策略文件夹 (目录名带时间戳)"""
    for name in sorted(os.listdir(RESULTS_DIR)):
        if re.match(rf"^inverse_{prefix}_", name) and os.path.isdir(
            os.path.join(RESULTS_DIR, name)
        ):
            return name
    raise FileNotFoundError(
        f"no strategy dir for prefix 'inverse_{prefix}_' in {RESULTS_DIR}"
    )


# ============================================================
# 5. PINN network 
# ============================================================

class PINN(nn.Module):
    def __init__(self, layers, inverse=True):
        super().__init__()
        self.lb = torch.tensor([X_MIN, Y_MIN, T_MIN], dtype=torch.float32)
        self.ub = torch.tensor([X_MAX, Y_MAX, T_MAX], dtype=torch.float32)
        self.inverse = inverse
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        if inverse:
            self.rho_param = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        else:
            self.rho_param = torch.tensor(RHO, dtype=torch.float32)

    def forward(self, x):
        x_norm = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)


# ============================================================
# 6. Prediction
# ============================================================

def predict_model(model, X_np, batch_size=PRED_BATCH_SIZE):
    outputs = []
    total = len(X_np)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            X_batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=DEVICE)
            y_batch = model(X_batch).detach().cpu().numpy().reshape(-1)
            outputs.append(y_batch)
    return np.concatenate(outputs)


def predict_tslice(model):
    """t = SLICE_C 时刻沿 y=0 的空间剖面 u(x_ref, 0, SLICE_C)"""
    X_slice = np.hstack([
        x_ref.reshape(-1, 1),
        np.zeros((len(x_ref), 1)),
        np.full((len(x_ref), 1), SLICE_C),
    ])
    return predict_model(model, X_slice)


def predict_xslice(model):
    """x = SLICE_C 位置沿 y=0 的时间演化 u(SLICE_C, 0, t_ref)"""
    X_slice = np.hstack([
        np.full((len(t_ref), 1), SLICE_C),
        np.zeros((len(t_ref), 1)),
        t_ref.reshape(-1, 1),
    ])
    return predict_model(model, X_slice)


# ============================================================
# 7. Evaluate all models
# ============================================================

print("=" * 70)
print(f"Inverse Fisher-KPP mixed-slice comparison (Seed {SEED})")
print(f"col1: t = {T_HEAT:.2f} (x,y) plane | col2: t = {SLICE_C:.2f}, y = 0 | "
      f"col3: x = {SLICE_C:.2f}, y = 0")
print("=" * 70)

errors_heat = {}   
l2_heat = {}       
pred_t = {}        
pred_x = {}     
maxerr_t = {}    
maxerr_x = {}   
exact_t_curve = exact_solution_fisher(np.hstack([
    x_ref.reshape(-1, 1),
    np.zeros((len(x_ref), 1)),
    np.full((len(x_ref), 1), SLICE_C),
]))  # 空间剖面参考解

exact_x_curve = exact_solution_fisher(np.hstack([
    np.full((len(t_ref), 1), SLICE_C),
    np.zeros((len(t_ref), 1)),
    t_ref.reshape(-1, 1),
]))  # 时间演化参考解

for method in METHODS:
    strategy_dir = find_strategy_dir(FOLDER_PREFIX[method])
    model_path = os.path.join(RESULTS_DIR, strategy_dir, f"seed_{SEED}", "model_final.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"missing model: {model_path}")

    model = PINN(LAYERS, inverse=True)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ---- 第一列: t = T_HEAT 平面误差云图 (65x65) ----
    pred_heat = predict_model(model, X_TEST_NP).reshape(X_MESH.shape)
    error_heat = np.abs(pred_heat - EXACT_HEAT)
    errors_heat[method] = error_heat
    l2_heat[method] = np.linalg.norm(pred_heat - EXACT_HEAT) / np.linalg.norm(EXACT_HEAT)

    # ---- t 切片 (空间剖面) ----
    pred_t[method] = predict_tslice(model)
    maxerr_t[method] = np.abs(pred_t[method] - exact_t_curve).max()

    # ---- x 切片 (时间演化) ----
    pred_x[method] = predict_xslice(model)
    maxerr_x[method] = np.abs(pred_x[method] - exact_x_curve).max()

    print(f"[{method}] L2(t={T_HEAT:.2f}) = {l2_heat[method]:.6e} | "
          f"max|err| = {error_heat.max():.6e} | "
          f"t-slice(t={SLICE_C:.2f}) max = {maxerr_t[method]:.6e} | "
          f"x-slice(x={SLICE_C:.2f}) max = {maxerr_x[method]:.6e}")

# ============================================================
# 8. Figure 5 x 3
# ============================================================

shared_vmax = max(errors_heat[m].max() for m in METHODS)
print(f"\nShared vmax (all heatmaps) = {shared_vmax:.6e}")


def draw_mixed_figure(mode):
    """
    mode = "independent" -> 每格热图自己的 vmax
    mode = "shared"      -> 5 格热图共享全局 vmax
    """
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15), constrained_layout=False)

    for i, method in enumerate(METHODS):

        # ------------------------------------------------
        # col 1: t = T_HEAT 时刻 (x, y) 平面绝对误差云图
        # ------------------------------------------------

        ax = axes[i, 0]
        error = errors_heat[method]

        error_vmin = 0.0
        if mode == "shared":
            error_vmax = shared_vmax
        else:
            error_vmax = error.max()
        if error_vmax <= error_vmin:
            error_vmax = 1e-12

        im = ax.pcolormesh(
            X_MESH, Y_MESH, error,
            shading="auto", cmap="jet",
            vmin=error_vmin, vmax=error_vmax,
        )

        ax.set_title(f"Absolute error (t = {T_HEAT:.2f})", fontsize=12, pad=6)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.tick_params(labelsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # ------------------------------------------------
        # col 2: t = SLICE_C 时刻沿 y=0 的空间剖面 u(x)
        #   pred(蓝实线) + exact(红虚线), 同轴
        # ------------------------------------------------

        ax = axes[i, 1]
        ax.plot(
            x_ref, pred_t[method],
            color=PRED_COLOR, lw=1.6,
            label="pred u(x,y,t)",
        )
        ax.plot(
            x_ref, exact_t_curve,
            color=EXACT_COLOR, linestyle="--", lw=2.0,
            label="exact u(x,y,t)",
        )
        ax.set_title(f"t = {SLICE_C:.2f} (y=0)", fontsize=12, pad=6)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("u(x,y,t)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=False, fontsize=7)

        # ------------------------------------------------
        # col 3: x = SLICE_C 位置沿 y=0 的时间演化 u(t), t in [0, 0.4]
        #   pred(蓝实线) + exact(红虚线), 同轴
        # ------------------------------------------------

        ax = axes[i, 2]
        ax.plot(
            t_ref, pred_x[method],
            color=PRED_COLOR, lw=1.6,
            label="pred u(x,y,t)",
        )
        ax.plot(
            t_ref, exact_x_curve,
            color=EXACT_COLOR, linestyle="--", lw=2.0,
            label="exact u(x,y,t)",
        )
        ax.set_title(f"x = {SLICE_C:.2f} (y=0)", fontsize=12, pad=6)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("u(x,y,t)", fontsize=10)
        ax.set_xlim(T_MIN, T_MAX)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=False, fontsize=7)

        # ------------------------------------------------
        # 行标签
        # ------------------------------------------------

        axes[i, 0].text(
            -0.34, 0.5, ROW_LABELS[method],
            transform=axes[i, 0].transAxes,
            rotation=90, va="center", ha="center",
            fontsize=11, fontweight="bold",
        )

    plt.subplots_adjust(
        left=0.08, right=0.98, top=0.965, bottom=0.035,
        wspace=0.50, hspace=0.62,
    )

    pdf_path = os.path.join(SAVE_DIR, f"Fisher-KPP_inverse_seed{SEED}_mixed_5x3_{mode}.pdf")
    png_path = os.path.join(SAVE_DIR, f"Fisher-KPP_inverse_seed{SEED}_mixed_5x3_{mode}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ({mode}):\n  {pdf_path}\n  {png_path}")


draw_mixed_figure("independent")
draw_mixed_figure("shared")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
