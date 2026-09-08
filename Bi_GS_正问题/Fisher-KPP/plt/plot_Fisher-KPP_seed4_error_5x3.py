# ============================================================
# plot_Fisher-KPP_seed4_error_5x3.py
# 2D Fisher-KPP equation - error comparison for paper
# Seed = 4
# ============================================================


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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Fisher-KPP domain
# ------------------------------------------------------------

X_MIN = -1.0
X_MAX = 1.0

Y_MIN = -1.0
Y_MAX = 1.0

T_MIN = 0.0
T_MAX = 0.4

# ------------------------------------------------------------
# Fisher-KPP parameters
# ------------------------------------------------------------

NU_FISHER = 0.05
RHO = 20.0

# ------------------------------------------------------------
# Network architecture
# ------------------------------------------------------------

LAYERS = [
    3,
    64,
    64,
    64,
    1
]

# ------------------------------------------------------------
# Plotting settings
# ------------------------------------------------------------

N_X = 256
N_Y = 256

# Time snapshots used in the paper
PLOT_TIMES = [
    0.0,
    0.2,
    0.4
]

# Batch size for model prediction
PRED_BATCH_SIZE = 65536


# ============================================================
# 2. Directories
# ============================================================

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


# ============================================================
# 3. Methods and model paths
# ============================================================

methods = [
    "PINN",
    "PCGrad",
    "GradNorm",
    "MOO-VARI",
    "Bi-GS-PINN"
]

# Display name -> results folder prefix
FOLDER_PREFIX = {
    "PINN": "standard",
    "PCGrad": "pcgrad",
    "GradNorm": "gradnorm",
    "MOO-VARI": "moo_vari",
    "Bi-GS-PINN": "bi_gs",
}

# Row labels
error_row_labels = {
    "PINN": "(a) PINN",
    "PCGrad": "(b) PCGrad",
    "GradNorm": "(c) GradNorm",
    "MOO-VARI": "(d) MOO-VARI",
    "Bi-GS-PINN": "(e) Bi-GS-PINN",
}


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
# 4. Fisher-KPP exact solution
# ============================================================

print("=" * 70)
print("2D Fisher-KPP absolute error comparison (Seed 4)")
print("=" * 70)

print(
    f"Seed       : {SEED}"
)

print(
    f"Device     : {DEVICE}"
)

print(
    f"nu_fisher  : {NU_FISHER}"
)

print(
    f"rho        : {RHO}"
)

print(
    f"Domain     : "
    f"x=[{X_MIN}, {X_MAX}], "
    f"y=[{Y_MIN}, {Y_MAX}], "
    f"t=[{T_MIN}, {T_MAX}]"
)


# ------------------------------------------------------------
# Analytical traveling-wave parameters
# ------------------------------------------------------------

k_fisher = np.sqrt(
    RHO / (
        6.0 * NU_FISHER
    )
)

c_fisher = (
    5.0
    * np.sqrt(
        NU_FISHER * RHO / 6.0
    )
)

print(
    f"k          : {k_fisher:.6f}"
)

print(
    f"wave speed  : {c_fisher:.6f}"
)


# ------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------

def exact_solution_fisher(X_np):
    """
    2D Fisher-KPP planar traveling-wave exact solution.

    Input:
        X_np: numpy array, shape (N, 3)
              columns = (x, y, t)

    Output:
        u: numpy array, shape (N, 1)
    """

    x = X_np[:, 0:1]
    y = X_np[:, 1:2]
    t = X_np[:, 2:3]

    # Rotated coordinate
    z = (
        x + y
    ) / np.sqrt(2.0)

    # Traveling-wave variable
    xi = (
        k_fisher
        * (
            z
            - c_fisher * t
        )
    )

    # Exact solution
    #
    # u(z,t) = [1 + exp(xi)]^-2
    #
    return (
        1.0
        / (
            1.0 + np.exp(xi)
        ) ** 2
    )


# ============================================================
# 5. PINN network
# ============================================================

class PINN(nn.Module):

    def __init__(
        self,
        layers,
        inverse=False
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input normalization
        # ----------------------------------------------------

        self.lb = torch.tensor(
            [
                X_MIN,
                Y_MIN,
                T_MIN
            ],
            dtype=torch.float32,
            device=DEVICE
        )

        self.ub = torch.tensor(
            [
                X_MAX,
                Y_MAX,
                T_MAX
            ],
            dtype=torch.float32,
            device=DEVICE
        )

        self.inverse = inverse

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        modules = []

        for i in range(
            len(layers) - 1
        ):

            modules.append(
                nn.Linear(
                    layers[i],
                    layers[i + 1]
                )
            )

            if i != len(layers) - 2:

                modules.append(
                    nn.Tanh()
                )

        self.net = nn.Sequential(
            *modules
        )

        # ----------------------------------------------------
        # Inverse parameter placeholder
        # ----------------------------------------------------

        if inverse:

            self.rho_param = nn.Parameter(
                torch.tensor(
                    0.5,
                    dtype=torch.float32
                )
            )

        else:

            self.rho_param = torch.tensor(
                RHO,
                dtype=torch.float32,
                device=DEVICE
            )

    def forward(self, x):

        # ----------------------------------------------------
        # Same normalization as training
        # ----------------------------------------------------

        x_norm = (
            2.0
            * (
                x - self.lb
            )
            / (
                self.ub - self.lb
            )
            - 1.0
        )

        return self.net(
            x_norm
        )


# ============================================================
# 6. Create plotting grids
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Creating plotting grids..."
)

print(
    "=" * 70
)


x_plot = np.linspace(
    X_MIN,
    X_MAX,
    N_X
)

y_plot = np.linspace(
    Y_MIN,
    Y_MAX,
    N_Y
)


# ------------------------------------------------------------
# 2D spatial mesh
# ------------------------------------------------------------

X_GRID, Y_GRID = np.meshgrid(
    x_plot,
    y_plot,
    indexing="xy"
)

print(
    f"Spatial grid: "
    f"{N_X} x {N_Y}"
)


# ============================================================
# 7. Generate exact solutions at t=0, 0.2, 0.4
# ============================================================

exact_snapshots = {}

print(
    "\nGenerating exact solution snapshots..."
)


for t_value in PLOT_TIMES:

    X_snapshot = np.column_stack(
        [
            X_GRID.reshape(-1),
            Y_GRID.reshape(-1),
            np.full(
                X_GRID.size,
                t_value
            )
        ]
    )

    exact_flat = exact_solution_fisher(
        X_snapshot
    )

    exact_grid = (
        exact_flat
        .reshape(
            N_Y,
            N_X
        )
    )

    exact_snapshots[
        t_value
    ] = exact_grid

    print(
        f"t = {t_value:.3f} | "
        f"range = "
        f"[{exact_grid.min():.6f}, "
        f"{exact_grid.max():.6f}]"
    )


# ============================================================
# 8. Load all models
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Loading models..."
)

print(
    "=" * 70
)


models = {}


for method in methods:

    strategy_dir = find_strategy_dir(
        FOLDER_PREFIX[method]
    )

    model_path = os.path.join(
        RESULTS_DIR,
        strategy_dir,
        f"seed_{SEED}",
        "model_final.pth"
    )

    print(
        f"\n[{method}]"
    )

    print(
        f"Path: {model_path}"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"\n找不到模型文件:\n"
            f"{model_path}\n\n"
            f"请检查 results 目录。"
        )

    # --------------------------------------------------------
    # Create network
    # --------------------------------------------------------

    model = PINN(
        LAYERS,
        inverse=False
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load state dict
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    models[
        method
    ] = model

    print(
        "Loaded successfully."
    )


# ============================================================
# 9. Prediction function
# ============================================================

def predict_model(
    model,
    X_np,
    batch_size=PRED_BATCH_SIZE
):
    """
    Predict model output in batches.

    X_np:
        shape (N, 3)
        columns = x, y, t

    Return:
        numpy array shape (N,)
    """

    outputs = []

    total = len(X_np)

    with torch.no_grad():

        for start in range(
            0,
            total,
            batch_size
        ):

            end = min(
                start + batch_size,
                total
            )

            X_batch = torch.tensor(
                X_np[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            y_batch = model(
                X_batch
            )

            outputs.append(
                y_batch
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

    return np.concatenate(
        outputs
    )


# ============================================================
# 10. Predict all models at all time snapshots
# ============================================================

predictions = {}

errors = {}

relative_l2_errors = {}

max_abs_errors = {}

mean_abs_errors = {}


print(
    "\n" + "=" * 70
)

print(
    "Generating predictions..."
)

print(
    "=" * 70
)


for method in methods:

    print(
        f"\n[{method}]"
    )

    predictions[
        method
    ] = {}

    errors[
        method
    ] = {}

    relative_l2_errors[
        method
    ] = {}

    max_abs_errors[
        method
    ] = {}

    mean_abs_errors[
        method
    ] = {}

    model = models[
        method
    ]


    x_eval = np.linspace(
        X_MIN,
        X_MAX,
        65
    )

    y_eval = np.linspace(
        Y_MIN,
        Y_MAX,
        65
    )

    t_eval = np.linspace(
        T_MIN,
        T_MAX,
        21
    )

    Xe, Ye, Te = np.meshgrid(
        x_eval,
        y_eval,
        t_eval,
        indexing="ij"
    )

    X_eval_np = np.stack(
        [
            Xe.reshape(-1),
            Ye.reshape(-1),
            Te.reshape(-1)
        ],
        axis=1
    )

    exact_eval = exact_solution_fisher(
        X_eval_np
    ).reshape(-1)

    pred_eval = predict_model(
        model,
        X_eval_np
    )

    # --------------------------------------------------------
    # Relative L2 error
    # --------------------------------------------------------

    relative_l2 = (
        np.linalg.norm(
            pred_eval - exact_eval
        )
        /
        np.linalg.norm(
            exact_eval
        )
    )

    relative_l2_errors[
        method
    ]["full"] = relative_l2

    print(
        f"Full-grid relative L2 = "
        f"{relative_l2:.6e}"
    )

    # --------------------------------------------------------
    # Prediction and error at each plotting time
    # --------------------------------------------------------

    for t_value in PLOT_TIMES:

        X_snapshot = np.column_stack(
            [
                X_GRID.reshape(-1),
                Y_GRID.reshape(-1),
                np.full(
                    X_GRID.size,
                    t_value
                )
            ]
        )

        pred_flat = predict_model(
            model,
            X_snapshot
        )

        pred_grid = (
            pred_flat
            .reshape(
                N_Y,
                N_X
            )
        )

        predictions[
            method
        ][
            t_value
        ] = pred_grid

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        exact_grid = exact_snapshots[
            t_value
        ]

        error_grid = np.abs(
            pred_grid
            - exact_grid
        )

        errors[
            method
        ][
            t_value
        ] = error_grid

        max_error = error_grid.max()

        mean_error = error_grid.mean()

        max_abs_errors[
            method
        ][
            t_value
        ] = max_error

        mean_abs_errors[
            method
        ][
            t_value
        ] = mean_error

        print(
            f"  t={t_value:.3f} | "
            f"Prediction = "
            f"[{pred_grid.min():.6f}, "
            f"{pred_grid.max():.6f}] | "
            f"Max error = "
            f"{max_error:.6e} | "
            f"Mean error = "
            f"{mean_error:.6e}"
        )


# ============================================================
# 11. Print final summary
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL RELATIVE L2 ERRORS"
)

print(
    "=" * 70
)


for method in methods:

    print(
        f"{method:15s}: "
        f"{relative_l2_errors[method]['full']:.6e}"
    )


# ============================================================
# 12. Figure 5 x 3 (Absolute Error)
#
# Two colorbar variants:
#   1) independent : per-panel vmax = error.max()
#   2) shared      : vmax = global max over all 5 methods x 3 times
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "Creating error figures (5 x 3)..."
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# Shared vmax across all panels
# ------------------------------------------------------------

shared_vmax = max(
    errors[method][t_value].max()
    for method in methods
    for t_value in PLOT_TIMES
)

print(
    f"Shared vmax (all panels) = {shared_vmax:.6e}"
)


def draw_error_figure(
    mode
):
    """
    mode = "independent"  -> each panel its own vmax
    mode = "shared"       -> all panels share global vmax
    """

    fig, axes = plt.subplots(
        nrows=5,
        ncols=3,
        figsize=(11, 15),
        constrained_layout=False
    )

    for i, method in enumerate(
        methods
    ):

        for j, t_value in enumerate(
            PLOT_TIMES
        ):

            ax = axes[
                i,
                j
            ]

            error = errors[
                method
            ][
                t_value
            ]

            # ----------------------------------------------------
            # Color scale
            # ----------------------------------------------------

            error_vmin = 0.0

            if mode == "shared":
                error_vmax = shared_vmax
            else:
                error_vmax = error.max()

            if error_vmax <= error_vmin:
                error_vmax = 1e-12

            im = ax.imshow(
                error,
                extent=[
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX
                ],
                origin="lower",
                aspect="equal",
                cmap="jet",
                vmin=error_vmin,
                vmax=error_vmax
            )

            # ----------------------------------------------------
            # Column title
            # ----------------------------------------------------

            ax.set_title(
                f"t = {t_value:.3f}",
                fontsize=11,
                pad=5
            )

            ax.set_xlabel(
                "x",
                fontsize=9
            )

            ax.set_ylabel(
                "y",
                fontsize=9
            )

            ax.tick_params(
                labelsize=7
            )

            # ----------------------------------------------------
            # Colorbar
            # ----------------------------------------------------

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            cbar.ax.tick_params(
                labelsize=7
            )

        # --------------------------------------------------------
        # Row label
        # --------------------------------------------------------

        axes[
            i,
            0
        ].text(
            -0.34,
            0.5,
            error_row_labels[
                method
            ],
            transform=axes[
                i,
                0
            ].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

    # ------------------------------------------------------------
    # Figure spacing
    # ------------------------------------------------------------

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.965,
        bottom=0.035,
        wspace=0.34,
        hspace=0.62
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    pdf_path = os.path.join(
        SAVE_DIR,
        f"Fisher-KPP_seed{SEED}_error_5x3_{mode}.pdf"
    )

    png_path = os.path.join(
        SAVE_DIR,
        f"Fisher-KPP_seed{SEED}_error_5x3_{mode}.png"
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

    plt.close(
        fig
    )

    print(
        f"\nSaved ({mode}):"
    )

    print(
        f"  {pdf_path}"
    )

    print(
        f"  {png_path}"
    )


# ------------------------------------------------------------
# Draw both variants
# ------------------------------------------------------------

draw_error_figure(
    "independent"
)

draw_error_figure(
    "shared"
)


# ============================================================
# 13. Final output
# ============================================================

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
    "\nFinal relative L2 errors:"
)

for method in methods:

    print(
        f"  {method:15s}: "
        f"{relative_l2_errors[method]['full']:.6e}"
    )

print(
    "\nFigures saved to:"
)

print(
    f"  {SAVE_DIR}"
)

print(
    "\n" + "=" * 70
)
