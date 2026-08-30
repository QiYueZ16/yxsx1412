# -*- coding: utf-8 -*-
"""
plot_burgers_inverse_seed9_mixed_5x3.py
反问题 1D Burgers 方程 - 论文用混合切片对比图 (Seed 9)
（cross-section: 空间剖面 + 时间演化, 两截面在 (t,x)=(0.75,0.75) 相交）

布局: 5 策略 x 3 列
  col 1: 整个求解域 (t, x) 上的绝对误差云图 |u_pred - u_exact| (与 error_5x3 一致)
  col 2: t = 0.75 时刻的空间剖面 u(x)   -- pred(蓝实线) + exact(红虚线), 同轴
  col 3: x = 0.75 位置的时间演化 u(t)   -- pred(蓝实线) + exact(红虚线), 同轴
         (t in [0, 2] 含训练域外推区; exact 用 Cole-Hopf 解析解,
          已验证与 .mat 参考解一致到 ~1e-12)

两种色标版本 (均输出 pdf + png):
  1) independent : 每格热图用各自的 vmax (= error.max())
  2) shared      : 5 格热图共享全局 vmax

网格: .mat 原始点, 256 (x) x 100 (t), 与训练一致
模型: results/inverse_{prefix}_A10000_L10000_*/seed_9/model_final.pth
  PINN(standard) / PCGrad(pcgrad) / GradNorm(gradnorm) / MOO-VARI(moo_vari) / Bi-GS-PINN(bi_gs)
"""

import os
import re
import glob
import numpy as np
import scipy.io
import scipy.interpolate as interp_lib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 1. Configuration
# ============================================================

SEED = 9

DEVICE = torch.device("cpu")

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

LAYERS = [2, 64, 64, 64, 1]

NU = 0.01 / np.pi  # 反问题真值 (Cole-Hopf 需要)

# 第二列: t 切片时刻 (空间剖面 u(x))
SLICE_T = 0.75

# 第三列: x 切片位置 (时间演化 u(t))
SLICE_X = 0.75

# 第三列 x 切片的 t 轴: t = 0.01 ~ 2.0 (t=0 处 Cole-Hopf 核奇异, 从 0.01 起画)
T_AXIS = np.linspace(0.01, 2.0, 200)

PRED_BATCH_SIZE = 65536

# ============================================================
# 2. Directories (路径一律用 ASCII 种子定位, 避免中文字面量)
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
SAVE_DIR = os.path.join(CURRENT_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# 脚本同目录下就有一份 .mat 副本 (与 Bi_GS_* 下两份 MD5 一致), 直接使用
DATA_PATH = os.path.join(CURRENT_DIR, "burgers_shock.mat")
if not os.path.exists(DATA_PATH):
    _mats = glob.glob(
        os.path.join(CURRENT_DIR, "..", "..", "Bi_GS_*", "burgers", "burgers_shock.mat")
    )
    if len(_mats) != 1:
        raise RuntimeError(f"expected exactly 1 mat file, found {_mats}")
    DATA_PATH = _mats[0]

# ============================================================
# 3. Reference solution (.mat grid, 与训练一致)
# ============================================================

data = scipy.io.loadmat(DATA_PATH)
x_ref = data["x"].flatten()
t_ref = data["t"].flatten()
usol = np.real(data["usol"])
if usol.shape[0] == len(t_ref) and usol.shape[1] == len(x_ref):
    usol = usol.T
elif usol.shape[0] == len(x_ref) and usol.shape[1] == len(t_ref):
    pass
else:
    raise ValueError(f"usol shape {usol.shape} vs x {len(x_ref)} t {len(t_ref)}")
if x_ref[0] > x_ref[-1]:
    x_ref = x_ref[::-1]
    usol = usol[::-1, :]
if t_ref[0] > t_ref[-1]:
    t_ref = t_ref[::-1]
    usol = usol[:, ::-1]

EXACT = usol  # (256, 100), 行 = x, 列 = t

X_MESH, T_MESH = np.meshgrid(x_ref, t_ref, indexing="ij")
X_TEST_NP = np.hstack([X_MESH.reshape(-1, 1), T_MESH.reshape(-1, 1)])

# 参考解在任意 (x, t) 处的取值 (与训练 exact_solution_burgers 同一套样条;
# SLICE_T = 0.75 在网格内, 样条即精确值)
_spline = interp_lib.RectBivariateSpline(x_ref, t_ref, EXACT)


def exact_at(x_np, t_value):
    return _spline.ev(x_np, np.full_like(x_np, t_value))


# ============================================================
# 4. Cole-Hopf analytic solution (x 切片 t > 0.99 段的真解)
#    参考 slice_l2_stats_beyond_t1_seed9.py, 已与 .mat 验证一致到 ~1e-12
# ============================================================

def cole_hopf_u(x_vals, t_val, xi_n=8001):
    """
    u(x,t) = [int (x-xi)/t * phi0(xi) * G(x,xi,t) dxi] / [int phi0(xi) * G(x,xi,t) dxi]
    phi0(xi) = exp(-cos(pi xi)/(2 pi nu)),  G = exp(-(x-xi)^2/(4 nu t))
    """
    xi = np.linspace(-2.0, 2.0, xi_n)
    phi0 = np.exp(-np.cos(np.pi * xi) / (2.0 * np.pi * NU))
    u_out = np.empty_like(x_vals)
    for i, xv in enumerate(x_vals):
        g = np.exp(-(xv - xi) ** 2 / (4.0 * NU * t_val))
        w = phi0 * g
        den = np.trapz(w, xi)
        num = np.trapz(w * (xv - xi) / t_val, xi)
        u_out[i] = num / den
    return u_out


# ============================================================
# 5. Methods and model paths
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

# 二三列 pred 曲线统一蓝色 (与快照图一致), exact 红色虚线
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
# 6. PINN network (inverse=True 注册 nu 参数以兼容 state_dict)
# ============================================================

class PINN(nn.Module):
    def __init__(self, layers, inverse=True):
        super().__init__()
        self.lb = torch.tensor([X_MIN, T_MIN], dtype=torch.float32)
        self.ub = torch.tensor([X_MAX, T_MAX], dtype=torch.float32)
        self.inverse = inverse
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        if inverse:
            self.nu = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))
        else:
            self.nu = torch.tensor(0.01 / np.pi, dtype=torch.float32)

    def forward(self, x):
        x_norm = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)


# ============================================================
# 7. Prediction
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
    """t = SLICE_T 时刻的空间剖面 u(x_ref, SLICE_T)"""
    X_slice = np.hstack([x_ref.reshape(-1, 1), np.full((len(x_ref), 1), SLICE_T)])
    return predict_model(model, X_slice)


def predict_xslice(model):
    """x = SLICE_X 位置的时间演化 u(SLICE_X, T_AXIS)"""
    X_slice = np.hstack([np.full((len(T_AXIS), 1), SLICE_X), T_AXIS.reshape(-1, 1)])
    return predict_model(model, X_slice)


# ============================================================
# 8. Evaluate all models
# ============================================================

print("=" * 70)
print(f"Inverse Burgers mixed-slice comparison (Seed {SEED})")
print(f"col2: t = {SLICE_T:.2f} slice | col3: x = {SLICE_X:.2f} slice, t in [0, 2]")
print("=" * 70)

errors_full = {}   # method -> (256, 100) 全网格绝对误差
l2_full = {}       # method -> 全网格相对 L2
pred_t = {}        # method -> u(x_ref, SLICE_T)
pred_x = {}        # method -> u(SLICE_X, T_AXIS)
maxerr_t = {}      # method -> t 切片最大绝对误差 (x 方向)
maxerr_x = {}      # method -> x 切片最大绝对误差 (t 方向)

exact_t_curve = exact_at(x_ref, SLICE_T)          # 空间剖面参考解
exact_x_curve = np.array([cole_hopf_u(np.array([SLICE_X]), t_val)[0] for t_val in T_AXIS])  # 时间演化参考解

for method in METHODS:
    strategy_dir = find_strategy_dir(FOLDER_PREFIX[method])
    model_path = os.path.join(RESULTS_DIR, strategy_dir, f"seed_{SEED}", "model_final.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"missing model: {model_path}")

    model = PINN(LAYERS, inverse=True)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ---- 全网格误差云图 (与训练 evaluate_full_mesh 同一网格) ----
    pred = predict_model(model, X_TEST_NP).reshape(EXACT.shape)
    error_grid = np.abs(pred - EXACT)
    errors_full[method] = error_grid
    l2_full[method] = np.linalg.norm(pred - EXACT) / np.linalg.norm(EXACT)

    # ---- t 切片 (空间剖面) ----
    pred_t[method] = predict_tslice(model)
    maxerr_t[method] = np.abs(pred_t[method] - exact_t_curve).max()

    # ---- x 切片 (时间演化) ----
    pred_x[method] = predict_xslice(model)
    maxerr_x[method] = np.abs(pred_x[method] - exact_x_curve).max()

    print(f"[{method}] L2 = {l2_full[method]:.6e} | "
          f"max|err| = {error_grid.max():.6e} | "
          f"t-slice(t={SLICE_T:.2f}) max = {maxerr_t[method]:.6e} | "
          f"x-slice(x={SLICE_X:.2f}) max = {maxerr_x[method]:.6e}")

# ============================================================
# 9. Figure 5 x 3
# ============================================================

shared_vmax = max(errors_full[m].max() for m in METHODS)
print(f"\nShared vmax (all heatmaps) = {shared_vmax:.6e}")


def draw_mixed_figure(mode):
    """
    mode = "independent" -> 每格热图自己的 vmax
    mode = "shared"      -> 5 格热图共享全局 vmax
    """
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15), constrained_layout=False)

    for i, method in enumerate(METHODS):

        # ------------------------------------------------
        # col 1: 全求解域绝对误差云图 (横轴 t, 纵轴 x, .mat 原始网格)
        # ------------------------------------------------

        ax = axes[i, 0]
        error = errors_full[method]

        error_vmin = 0.0
        if mode == "shared":
            error_vmax = shared_vmax
        else:
            error_vmax = error.max()
        if error_vmax <= error_vmin:
            error_vmax = 1e-12

        im = ax.pcolormesh(
            T_MESH, X_MESH, error,
            shading="auto", cmap="jet",
            vmin=error_vmin, vmax=error_vmax,
        )

        ax.set_title("Absolute error", fontsize=12, pad=6)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("x", fontsize=10)
        ax.tick_params(labelsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # ------------------------------------------------
        # col 2: t = SLICE_T 时刻空间剖面 u(x)
        #   pred(蓝实线) + exact(红虚线), 同轴
        # ------------------------------------------------

        ax = axes[i, 1]
        ax.plot(
            x_ref, pred_t[method],
            color=PRED_COLOR, lw=1.6,
            label="pred u(x,t)",
        )
        ax.plot(
            x_ref, exact_t_curve,
            color=EXACT_COLOR, linestyle="--", lw=2.0,
            label="exact u(x,t)",
        )
        ax.set_title(f"t = {SLICE_T:.2f}", fontsize=12, pad=6)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("u(x,t)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=False, fontsize=7)

        # ------------------------------------------------
        # col 3: x = SLICE_X 位置时间演化 u(t), t in [0, 2]
        #   pred(蓝实线) + exact(红虚线), 同轴
        # ------------------------------------------------

        ax = axes[i, 2]
        ax.plot(
            T_AXIS, pred_x[method],
            color=PRED_COLOR, lw=1.6,
            label="pred u(x,t)",
        )
        ax.plot(
            T_AXIS, exact_x_curve,
            color=EXACT_COLOR, linestyle="--", lw=2.0,
            label="exact u(x,t)",
        )
        ax.set_title(f"x = {SLICE_X:.2f}", fontsize=12, pad=6)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("u(x,t)", fontsize=10)
        ax.set_xlim(0.0, 2.0)
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

    pdf_path = os.path.join(SAVE_DIR, f"burgers_inverse_seed{SEED}_mixed_5x3_{mode}.pdf")
    png_path = os.path.join(SAVE_DIR, f"burgers_inverse_seed{SEED}_mixed_5x3_{mode}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ({mode}):\n  {pdf_path}\n  {png_path}")


draw_mixed_figure("independent")
draw_mixed_figure("shared")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
