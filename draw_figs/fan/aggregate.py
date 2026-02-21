import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# =====================================================
# Unified Style Configuration
# =====================================================
# All style parameters are centralized here for easy modification.
# Change these values to update all figures consistently.

# Font sizes
FONT_SCALE = 2.2  # Base font scale for seaborn
AXIS_LABEL_FONTSIZE = 24  # xlabel and ylabel font size
TICK_LABEL_FONTSIZE = 22  # Tick label font size
LEGEND_FONTSIZE = 20  # Legend font size
ANNOTATION_FONTSIZE = 24  # Text annotations font size

# Line widths
BASE_LINEWIDTH = 4.0  # Base line width for plots
AXIS_LINEWIDTH = 2.5  # Axis and tick line width
GRID_LINEWIDTH = 1.5  # Grid line width
BAR_EDGE_LINEWIDTH = 2.5  # Bar plot edge line width

# Colors (consistent across all figures)
COLOR_ORANGE = "#e67e22"  # Primary color for "Ours" methods
COLOR_GRAY = "#7f8c8d"  # Secondary color for baselines
COLOR_BLUE = "#3498db"  # Oracle
COLOR_RED = "#e74c3c"  # Progress
COLOR_GREEN = "#2ecc71"  # Uncertainty
COLOR_LIGHT_GRAY = "#95a5a6"  # Vanilla

# Marker sizes
MARKER_SIZE_LARGE = 14  # Large markers (line plots)
MARKER_SIZE_MEDIUM = 12  # Medium markers
MARKER_SIZE_SMALL = 10  # Small markers (legends)

# Grid and spine settings
GRID_ALPHA = 0.6  # Grid transparency
GRID_STYLE = ":"  # Grid line style
SPINE_LINEWIDTH = 2.5  # Spine line width

# =====================================================
# Shared style helper function
# =====================================================

def setup_icml_style():
    """Configure a unified ICML-like plotting style using global constants."""
    sns.set_theme(style="ticks", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "lines.linewidth": BASE_LINEWIDTH,
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.major.width": AXIS_LINEWIDTH,
        "ytick.major.width": AXIS_LINEWIDTH,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =====================================================
# Reusable Plotting Helper Functions
# =====================================================

def set_axis_labels(ax, xlabel, ylabel, xlabelpad=15, ylabelpad=15):
    """Set axis labels with unified styling."""
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=xlabelpad)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=ylabelpad)


def add_grid(ax, axis="y", alpha=None, linestyle=None, linewidth=None):
    """Add grid lines with unified styling."""
    if alpha is None:
        alpha = GRID_ALPHA
    if linestyle is None:
        linestyle = GRID_STYLE
    if linewidth is None:
        linewidth = GRID_LINEWIDTH
    ax.grid(axis=axis, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=0)


def create_legend(ax, handles=None, labels=None, loc="upper center", bbox_to_anchor=None, ncol=2, 
                  fontsize=None, columnspacing=1.0, handletextpad=0.5, 
                  borderaxespad=0.0, frameon=False):
    """Create legend with unified styling. Supports custom handles/labels."""
    if fontsize is None:
        fontsize = LEGEND_FONTSIZE
    if bbox_to_anchor is None:
        bbox_to_anchor = (0.5, 1.02)
    
    if handles is not None and labels is not None:
        ax.legend(
            handles=handles,
            labels=labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            fontsize=fontsize,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
            borderaxespad=borderaxespad,
            frameon=frameon,
        )
    else:
        ax.legend(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            fontsize=fontsize,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
            borderaxespad=borderaxespad,
            frameon=frameon,
        )
def create_figure_level_legend(fig, handles, ncol=None, fontsize=None,
                               bbox_to_anchor=(0.5, 1.08), columnspacing=1.0):
    """Create figure-level legend (for bar plots with hidden x-axis labels)."""
    if fontsize is None:
        fontsize = LEGEND_FONTSIZE - 4  # Slightly smaller for figure-level
    if ncol is None:
        ncol = 3 if len(handles) > 3 else len(handles)
    
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=fontsize,
        columnspacing=columnspacing,
        frameon=False,
    )


def apply_despine(ax, trim=False):
    """Apply despine with unified settings."""
    sns.despine(ax=ax, trim=trim)


# =====================================================
# Data for Figure 1 (fig1_50.py): Active learning vs. budget percentage
# =====================================================

fig1_loss_random = [7.86625, 5.77735, 4.04824, 3.43389, 3.06295]
fig1_loss_ours = [7.86625, 3.77809, 2.31340, 1.46498, 1.19249]

# Map rounds (0-4) to percentage (0-100)
fig1_percentages = [0, 25, 50, 75, 100]

fig1_df_random = pd.DataFrame({"pct": fig1_percentages, "loss": fig1_loss_random, "strategy": "Random"})
fig1_df_ours = pd.DataFrame({"pct": fig1_percentages, "loss": fig1_loss_ours, "strategy": "Ours"})
fig1_df = pd.concat([fig1_df_random, fig1_df_ours], ignore_index=True)

fig1_base_val = fig1_loss_random[0]
fig1_df["norm_loss"] = fig1_df["loss"] / fig1_base_val
fig1_scale = 1.0 / fig1_base_val


# =====================================================
# Data for Figure 3 (fig3_idm_comparison.py): IDM accuracy per dataset (4 methods)
# =====================================================

fig3_dataset_labels = [
    "Maze",
    "Wall",
    "Push-T",
    "Lift",
    "Can",
    "Square",
]

fig3_method_labels = ["TD-MPC2", "Dreamerv3", "DINO-WM", "TC-WM (Ours)"]
fig3_method_colors = [COLOR_LIGHT_GRAY, COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE]
fig3_method_values = {
    "TD-MPC2": [0.10, 0.28, 0.20, 0.24, 0.22, 0.16],
    "Dreamerv3": [1, 1, 0.3, 0.28, 0.22, 0.12],
    "DINO-WM": [0.98, 0.96, 0.90, 0.32, 0.30, 0.24],
    "TC-WM (Ours)": [1.00, 0.94, 0.92, 0.60, 0.62, 0.40],
}

fig3_dataset_count = len(fig3_dataset_labels)
for method_label in fig3_method_labels:
    values = fig3_method_values.get(method_label)
    if values is None:
        raise ValueError(f"Missing values for Figure 3 method: {method_label}")
    if len(values) != fig3_dataset_count:
        raise ValueError(
            f"Figure 3 values for {method_label} must have {fig3_dataset_count} entries, got {len(values)}."
        )

fig3_method_values_scaled = {}
for method_label, values in fig3_method_values.items():
    scaled = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            scaled.append(v)
        elif v <= 1.05:
            scaled.append(v * 100)
        else:
            scaled.append(v)
    fig3_method_values_scaled[method_label] = scaled


# =====================================================
# Data for Figure 6: PR/LP losses across datasets
# =====================================================

fig6_datasets = ["Maze", "Wall", "Push-T", "Lift", "Can", "Square", "Reacher", "Cheetah", "Hopper"]
fig6_datasets_raw_order = ["Lift", "Can", "Square", "Wall", "Maze", "PushT", "Reacher", "Cheetah", "Hopper"]
fig6_methods = ["TD-MPC2", "DreamerV3", "DINO-WM", "TC-WM (Ours)"]
fig6_method_colors = {
    "TD-MPC2": COLOR_LIGHT_GRAY,
    "DreamerV3": COLOR_BLUE,
    "DINO-WM": COLOR_GREEN,
    "TC-WM (Ours)": COLOR_ORANGE,
}

fig6_pr_values = {
    "TD-MPC2": [7.607, 3.192, 6.480, 1.533, 4.532, 5.321, 5.581, 65.300, 7.823],
    "DreamerV3": [8.924, 1.889, 8.067, 5.757, 0.942, 1.252, 8.156, 2.547, 2.192],
    "DINO-WM": [6.708, 3.500, 22.378, 2.644, 0.120, 3.204, 0.963, 14.815, 0.536],
    "TC-WM (Ours)": [4.050, 2.319, 5.790, 1.960, 0.093, 3.936, 1.492, 2.819, 3.838],
}

fig6_lp_values = {
    "TD-MPC2": [0.642, 0.741, 0.663, 1.340, 0.459, 18.907, 32.179, 47.250, 28.716],
    "DreamerV3": [43.213, 50.781, 70.801, 50.049, 47.607, 47.119, 70.801, 73.242, 32.227],
    "DINO-WM": [154.090, 83.130, 278.390, 4.252, 0.205, 25.045, 286.540, 6.918, 202.400],
    "TC-WM (Ours)": [0.333, 0.015, 0.127, 2.680, 0.117, 3.398, 6.300, 3.120, 12.701],
}

if fig6_datasets_raw_order != fig6_datasets:
    fig6_index_map = {name: idx for idx, name in enumerate(fig6_datasets_raw_order)}
    fig6_reorder = [fig6_index_map[name.replace("Push-T", "PushT")] for name in fig6_datasets]
    fig6_pr_values = {
        method: [values[i] for i in fig6_reorder] for method, values in fig6_pr_values.items()
    }
    fig6_lp_values = {
        method: [values[i] for i in fig6_reorder] for method, values in fig6_lp_values.items()
    }

for method_label in fig6_methods:
    if method_label not in fig6_pr_values or method_label not in fig6_lp_values:
        raise ValueError(f"Figure 6 missing values for method: {method_label}")


# =====================================================
# Data for Figure 4 (fig4_gap.py): OOS dynamics accuracy vs. samples
# =====================================================

fig4_x = [200, 400, 600, 800, 1000]

fig4_wm_acc = [0.5067, 0.6240, 0.7393, 0.7708, 0.8553]
fig4_wm_acc = [val * 100 for val in fig4_wm_acc]

fig4_im_acc = [0.7756, 0.9772, 0.9943, 1.000, 1.000]
fig4_im_acc = [val * 100 for val in fig4_im_acc]


# =====================================================
# Data for Figure 5 (fig_complexity.py): accuracy vs. state complexity
# =====================================================

complexity_levels = [6, 8, 10, 12, 14]

wm_acc_complexity = [0.9037, 0.8208, 0.7088, 0.6279, 0.4479]
wm_acc_complexity = [x * 100 for x in wm_acc_complexity]

idm_acc_complexity = [0.9495, 0.9466, 0.9270, 0.9337, 0.9136]
idm_acc_complexity = [x * 100 for x in idm_acc_complexity]


# =====================================================
# Shared JSON loader for Figures 5 and 6
# (fig5_average2.py and fig6_self_im.py)
# =====================================================


def robust_load_json(path: str):
    """Load possibly broken JSON logs by extracting individual objects."""
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return []
    all_records = []

    # Find all top-level JSON objects
    matches = re.finditer(r"\{[^{}]*\}", content)
    for match in matches:
        try:
            record = json.loads(match.group())
            all_records.append(record)
        except json.JSONDecodeError:
            continue

    # Fallback: try to repair a truncated list
    if not all_records:
        content = content.strip()
        if content.endswith(","):
            content = content[:-1]
        if not content.endswith("]"):
            content += "]"
        try:
            all_records = json.loads(content)
        except Exception:
            pass

    return all_records


# =====================================================
# Data for Figure 5 (fig5_average2.py): average performance across strategies
# =====================================================

_BASE_DIR = Path(__file__).resolve().parent
# Put your local logs next to this script (or change these paths as needed).
fig5_json_path = str(_BASE_DIR / "all_rounds_results_all.json")
fig5_target_round = 3

fig5_records = robust_load_json(fig5_json_path)
if not fig5_records:
    # Fallback: generate a small, schema-compatible sample so the script can run locally.
    fig5_records = [
        {"round": 3, "strategy": "Random", "macro_mse": 1.85},
        {"round": 3, "strategy": "Learning-Progress", "macro_mse": 1.55},
        {"round": 3, "strategy": "Uncertainty", "macro_mse": 1.42},
        {"round": 3, "strategy": "Hard-MSE", "macro_mse": 1.25},
        {"round": 3, "strategy": "Forward-Inverse-Oracle", "macro_mse": 1.05},
    ]
fig5_df = pd.DataFrame(fig5_records)

fig5_df = fig5_df[fig5_df["round"] == fig5_target_round].copy()

fig5_name_mapping = {
    "Random": "Vanilla",
    "Uncertainty": "Uncertainty",
    "Hard-MSE": "Oracle",
    "Learning-Progress": "Progress",
    "Forward-Inverse-Oracle": "Active Exploration (Ours)",
}

fig5_df["strategy"] = fig5_df["strategy"].replace(fig5_name_mapping)

fig5_strategy_order = [
    "Vanilla",
    "Progress",
    "Uncertainty",
    "Oracle",
    "Active Exploration (Ours)",
]

fig5_df = fig5_df[fig5_df["strategy"].isin(fig5_strategy_order)]


# =====================================================
# Data for fig6_self_im.py: self-improvement curves
# =====================================================

fig6_json_path = str(_BASE_DIR / "all_rounds_results_all_self-im.json")

fig6_records = robust_load_json(fig6_json_path)
if not fig6_records:
    # Fallback sample: rounds -> macro_mse decreasing with budget, two strategies.
    fig6_records = [
        {"round": 0, "strategy": "self-improvement(vanilla)", "macro_mse": 1.90},
        {"round": 1, "strategy": "self-improvement(vanilla)", "macro_mse": 1.55},
        {"round": 2, "strategy": "self-improvement(vanilla)", "macro_mse": 1.35},
        {"round": 3, "strategy": "self-improvement(vanilla)", "macro_mse": 1.22},
        {"round": 4, "strategy": "self-improvement(vanilla)", "macro_mse": 1.15},
        {"round": 0, "strategy": "self-improvement(ours)", "macro_mse": 1.90},
        {"round": 1, "strategy": "self-improvement(ours)", "macro_mse": 1.35},
        {"round": 2, "strategy": "self-improvement(ours)", "macro_mse": 1.10},
        {"round": 3, "strategy": "self-improvement(ours)", "macro_mse": 0.98},
        {"round": 4, "strategy": "self-improvement(ours)", "macro_mse": 0.92},
    ]
fig6_df = pd.DataFrame(fig6_records)

if not fig6_df.empty:
    fig6_df["macro_mse"] = pd.to_numeric(fig6_df["macro_mse"], errors="coerce")
    fig6_df["round"] = pd.to_numeric(fig6_df["round"], errors="coerce")
    fig6_df = fig6_df.dropna(subset=["macro_mse", "round"])

    fig6_df["data_budget"] = fig6_df["round"] * 100

    fig6_strategy_mapping = {
        "self-improvement(vanilla)": "Self-Improvement (Vanilla)",
        "self-improvement(ours)": "Self-Improvement (Ours)",
    }
    fig6_df["strategy"] = fig6_df["strategy"].replace(fig6_strategy_mapping)

print("Shared imports, styles, and data for all figures are ready.")

# Figure 3: IDM accuracy per dataset (4 methods)

setup_icml_style()

def plot_fig3_group(ax, dataset_labels, dataset_indices, ylim, yticks, title=None, show_ylabel=True):

    x = np.arange(len(dataset_labels))
    total_width = 0.92
    bar_width = total_width / len(fig3_method_labels)
    offsets = (np.arange(len(fig3_method_labels)) - (len(fig3_method_labels) - 1) / 2) * bar_width

    for idx, (method_label, color) in enumerate(zip(fig3_method_labels, fig3_method_colors)):
        values = [fig3_method_values_scaled[method_label][i] for i in dataset_indices]
        bars = ax.bar(
            x + offsets[idx],
            values,
            bar_width,
            label=method_label,
            color=color,
            zorder=3,
        )
        for bar in bars:
            height = bar.get_height()
            if height is None or (isinstance(height, float) and np.isnan(height)):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(height - 1.2, 0),
                f"{height:.0f}",
                ha="center",
                va="top",
                rotation=90,
                fontsize=TICK_LABEL_FONTSIZE - 2,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontweight="bold", rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE + 2)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    set_axis_labels(ax, "", "Success Rate (%)" if show_ylabel else "", ylabelpad=15)

    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    add_grid(ax, axis="y", linestyle="--", alpha=0.5)
    apply_despine(ax, trim=False)

    if title:
        ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", pad=10)


fig3_common_size = (10.5, 6.2)
fig3_total_size = (fig3_common_size[0] * 2.0, fig3_common_size[1])
fig, axes = plt.subplots(1, 2, figsize=fig3_total_size, sharey=False)

fig3_group_a_labels = ["Maze", "Wall", "Push-T"]
fig3_group_a_indices = [fig3_dataset_labels.index(name) for name in fig3_group_a_labels]
plot_fig3_group(
    axes[0],
    fig3_group_a_labels,
    fig3_group_a_indices,
    ylim=(0, 100),
    yticks=[0, 25, 50, 75, 100],
    title="CEM",
    show_ylabel=True,
)

fig3_group_b_labels = ["Lift", "Can", "Square"]
fig3_group_b_indices = [fig3_dataset_labels.index(name) for name in fig3_group_b_labels]
plot_fig3_group(
    axes[1],
    fig3_group_b_labels,
    fig3_group_b_indices,
    ylim=(0, 75),
    yticks=[0, 25, 50, 75],
    title="LDP",
    show_ylabel=False,
)

legend_handles = [
    mpatches.Patch(color=color, label=label)
    for label, color in zip(fig3_method_labels, fig3_method_colors)
]
fig.legend(
    legend_handles,
    fig3_method_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    ncol=len(fig3_method_labels),
    fontsize=LEGEND_FONTSIZE + 2,
    columnspacing=1.2,
    handletextpad=0.6,
    frameon=False,
)

plt.tight_layout()
plt.subplots_adjust(top=0.86, wspace=0.15)
fig3_output_path = "plan_compare_cem_ldp.pdf"
plt.savefig(fig3_output_path, bbox_inches="tight", transparent=True)
print(f"Figure 3 saved to: {fig3_output_path}")

################################################################################
# Figure 6: PR/LP losses across datasets (two panels, shared legend)

setup_icml_style()

fig6_fig_width = max(26.0, 3.2 * len(fig6_datasets))
fig, axes = plt.subplots(1, 2, figsize=(fig6_fig_width, 7))

fig6_total_width = 0.65
fig6_bar_width = fig6_total_width / len(fig6_methods)
fig6_offsets = (
    (np.arange(len(fig6_methods)) - (len(fig6_methods) - 1) / 2) * fig6_bar_width
)

for ax, values, title in [
    (axes[0], fig6_pr_values, "Image Prediction (↓)"),
    (axes[1], fig6_lp_values, "Latent Prediction (↓)"),
]:
    x = np.arange(len(fig6_datasets))
    for idx, method_label in enumerate(fig6_methods):
        ax.bar(
            x + fig6_offsets[idx],
            values[method_label],
            fig6_bar_width,
            label=method_label,
            color=fig6_method_colors[method_label],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(fig6_datasets, fontweight="bold", rotation=20, ha="right")
    ax.tick_params(axis="x", pad=2)
    ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", pad=12)
    set_axis_labels(ax, "", "")
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE + 2)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.2, base=10)
    max_val = max(max(v) for v in values.values())
    max_ylim = max(2, max_val * 1.05)
    ax.set_ylim(0, max_ylim)
    tick_candidates = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ticks = [t for t in tick_candidates if t <= max_ylim]
    if len(ticks) > 8:
        ticks = [0, 0.2, 0.5, 1, 2, 5, 10] + [t for t in ticks if t >= 50]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" if t < 100 else f"{int(t)}" for t in ticks])

    add_grid(ax, axis="y", linestyle="--", alpha=0.5)
    apply_despine(ax, trim=False)

legend_handles = [
    mpatches.Patch(color=fig6_method_colors[name], label=name) for name in fig6_methods
]
create_figure_level_legend(
    fig,
    legend_handles,
    ncol=len(fig6_methods),
    fontsize=LEGEND_FONTSIZE + 2,
    bbox_to_anchor=(0.5, -0.004),
)

fig.supylabel(r"MSE Loss (x $10^{-3}$)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

plt.tight_layout()
plt.subplots_adjust(top=0.86, bottom=0.12, wspace=0.12)

fig6_output_path = "pred_loss.pdf"
plt.savefig(fig6_output_path, bbox_inches="tight", transparent=True)
print(f"Figure 6 saved to: {fig6_output_path}")

################################################################################
# Linear probing scatter: R^2 (proprio vs object), size indicates goal

setup_icml_style()

# Placeholder data: 4 methods, 1 point each (R^2 in [0, 1]).
# Each point is (proprio_r2, object_r2, goal).
linear_probing_data_lift = {
    "TD-MPC2": {
        "color": COLOR_GRAY,
        "marker": "o",
        "points": [(0.38, 0.32, 0.49)],
    },
    "Dreamerv3": {
        "color": "#9cc9f5",
        "marker": "o",
        "points": [(0.64, 0.44, 0.53)],
    },
    "DINO-WM": {
        "color": COLOR_BLUE,
        "marker": "o",
        "points": [(0.75, 0.83, 0.82)],
    },
    "TC-WM (Ours)": {
        "color": COLOR_ORANGE,
        "marker": "o",
        "points": [(0.83, 0.89, 0.95)],
    },
}

linear_probing_data_can = {
    "TD-MPC2": {
        "color": COLOR_GRAY,
        "marker": "o",
        "points": [(0.45, 0.63, 0.40)],
    },
    "Dreamerv3": {
        "color": "#9cc9f5",
        "marker": "o",
        "points": [(0.34, 0.54, 0.32)],
    },
    "DINO-WM": {
        "color": COLOR_BLUE,
        "marker": "o",
        "points": [(0.72, 0.73, 0.72)],
    },
    "TC-WM (Ours)": {
        "color": COLOR_ORANGE,
        "marker": "o",
        "points": [(0.87, 0.91, 0.92)],
    },
}

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

def _goal_size(goal, base=60, scale=1600):
    # Larger goal => larger size (manual input from experiment).
    return base + scale * goal

def _plot_linear_probing(ax, data, title, show_ylabel=True):
    for method, spec in data.items():
        xs = [p[0] for p in spec["points"]]
        ys = [p[1] for p in spec["points"]]
        sizes = [_goal_size(p[2]) for p in spec["points"]]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            label=method,
            color=spec["color"],
            marker=spec["marker"],
            alpha=0.9,
            zorder=5,
        )

    set_axis_labels(ax, "proprio states", "object states" if show_ylabel else "")
    ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", pad=10)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.25, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0.25, 1.01, 0.25))
    ax.set_yticks(np.arange(0.50, 1.01, 0.25))
    ax.minorticks_off()

    add_grid(ax, axis="y")
    add_grid(ax, axis="x", alpha=0.3)
    apply_despine(ax, trim=False)

    # Size legend with three circles inside a boxed note
    note_x, note_y = 0.04, 0.62  # axes fraction
    box = mpatches.FancyBboxPatch(
        (note_x, note_y),
        0.46,
        0.32,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor="lightgray",
        linewidth=1.2,
        zorder=6,
    )
    ax.add_patch(box)

    size_levels = [0.4, 0.7, 1.0]
    row_y = [note_y + 0.29, note_y + 0.18, note_y + 0.05]

    ax.text(
        note_x + 0.05,
        note_y + 0.17,
        "goal\n states",
        transform=ax.transAxes,
        va="center",
        ha="center",
        rotation=90,
        fontsize=LEGEND_FONTSIZE - 6,
        zorder=7,
    )
    for v, y in zip(size_levels, row_y):
        ax.scatter(
            [note_x + 0.19],
            [y],
            s=_goal_size(v),
            color="lightgray",
            alpha=0.9,
            transform=ax.transAxes,
            zorder=7,
        )
        ax.text(
            note_x + 0.27,
            y,
            f"{v:.2f}",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=LEGEND_FONTSIZE - 4,
            zorder=7,
        )

_plot_linear_probing(axes[0], linear_probing_data_lift, "Lift", show_ylabel=True)
_plot_linear_probing(axes[1], linear_probing_data_can, "Can", show_ylabel=False)

# Create uniform-size legend markers (independent of data sizes).
legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=12,
           markerfacecolor=linear_probing_data_lift[name]["color"],
           markeredgewidth=0, label=name)
    for name in linear_probing_data_lift.keys()
]
fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    columnspacing=1.5,
    handletextpad=0.5,
    borderaxespad=0.0,
    frameon=False,
    fontsize=LEGEND_FONTSIZE,
)

plt.tight_layout()
plt.subplots_adjust(top=0.86, wspace=0.15)

linear_probe_output_path = "linear_probing_scatter.pdf"
plt.savefig(linear_probe_output_path, bbox_inches="tight", transparent=True)
print(f"linear_probing_scatter saved to: {linear_probe_output_path}")

################################################################################
# Loss compare: DINO-z vs Ours-z (aligned steps, fig4 style)

setup_icml_style()

loss_compare_dir = _BASE_DIR.parent / "loss_compare"
dino_csv_path = loss_compare_dir / "dino_z.csv"
ours_csv_path = loss_compare_dir / "ours_z.csv"

# Alignment mode: "intersection" uses exact shared steps, "interpolate" fills missing steps.
LOSS_COMPARE_ALIGN_MODE = "intersection"


def _load_loss_curve(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    step_col = "Step" if "Step" in df.columns else df.columns[0]
    value_cols = [
        c for c in df.columns
        if c != step_col and not c.endswith("__MIN") and not c.endswith("__MAX")
    ]
    if not value_cols:
        value_cols = [c for c in df.columns if c != step_col]
    if not value_cols:
        return None
    value_col = value_cols[0]
    out = df[[step_col, value_col]].rename(columns={step_col: "step", value_col: "value"})
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["step", "value"]).sort_values("step")
    return out, value_col


def _align_curves(dino_df, ours_df):
    if LOSS_COMPARE_ALIGN_MODE == "interpolate":
        steps = np.array(sorted(set(dino_df["step"]) | set(ours_df["step"])))
        dino_i = dino_df.set_index("step").reindex(steps).interpolate(method="index")
        ours_i = ours_df.set_index("step").reindex(steps).interpolate(method="index")
        aligned = pd.DataFrame({
            "step": steps,
            "dino": dino_i["value"].values,
            "ours": ours_i["value"].values,
        }).dropna()
        return aligned
    aligned = pd.merge(dino_df, ours_df, on="step", how="inner", suffixes=("_dino", "_ours"))
    aligned = aligned.rename(columns={"value_dino": "dino", "value_ours": "ours"})
    return aligned


dino_loaded = _load_loss_curve(dino_csv_path)
ours_loaded = _load_loss_curve(ours_csv_path)

if dino_loaded is None or ours_loaded is None:
    print("loss_compare_z skipped: missing or empty dino_z.csv / ours_z.csv.")
else:
    dino_df, dino_label = dino_loaded
    ours_df, ours_label = ours_loaded
    aligned_df = _align_curves(dino_df, ours_df)

    if aligned_df.empty:
        print("loss_compare_z skipped: no aligned steps found.")
    else:
        fig, ax = plt.subplots(figsize=(9, 7))

        ax.plot(
            aligned_df["step"],
            aligned_df["ours"],
            label="Ours (z)",
            color=COLOR_ORANGE,
            linestyle="-",
            marker="o",
            markersize=MARKER_SIZE_LARGE,
            markerfacecolor="white",
            markeredgewidth=BAR_EDGE_LINEWIDTH,
            zorder=5,
        )

        ax.plot(
            aligned_df["step"],
            aligned_df["dino"],
            label="DINO-WM (z)",
            color=COLOR_GRAY,
            linestyle="--",
            dashes=(4, 2),
            marker="s",
            markersize=MARKER_SIZE_LARGE,
            markerfacecolor="white",
            markeredgewidth=BAR_EDGE_LINEWIDTH,
            zorder=4,
        )

        y_label = "Z Loss"
        for raw_label in (ours_label, dino_label):
            if isinstance(raw_label, str) and " - " in raw_label:
                y_label = raw_label.split(" - ", 1)[-1].strip()
                break

        set_axis_labels(ax, "Training Step", y_label)
        add_grid(ax, axis="y")
        add_grid(ax, axis="x", alpha=0.3)
        apply_despine(ax, trim=False)

        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0]
        create_legend(
            ax,
            handles=[handles[i] for i in order],
            labels=[labels[i] for i in order],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            columnspacing=1.5,
            handletextpad=0.5,
            borderaxespad=0.0,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        loss_compare_output_path = "loss_compare_z.pdf"
        plt.savefig(loss_compare_output_path, bbox_inches="tight", transparent=True)
        print(f"loss_compare_z saved to: {loss_compare_output_path}")
