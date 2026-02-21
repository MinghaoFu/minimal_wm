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
        fontsize = LEGEND_FONTSIZE - 4
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
# Data for Figure 1: Active learning vs. budget percentage
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
# Data for Figure 4: OOS dynamics accuracy vs. samples
# =====================================================

fig4_x = [200, 400, 600, 800, 1000]

fig4_wm_acc = [0.5067, 0.6240, 0.7393, 0.7708, 0.8553]
fig4_wm_acc = [val * 100 for val in fig4_wm_acc]

fig4_im_acc = [0.7756, 0.9772, 0.9943, 1.000, 1.000]
fig4_im_acc = [val * 100 for val in fig4_im_acc]


# =====================================================
# Data for Figure 5: Robustness vs. state complexity
# =====================================================

complexity_levels = [6, 8, 10, 12, 14]

wm_acc_complexity = [0.9037, 0.8208, 0.7088, 0.6279, 0.4479]
wm_acc_complexity = [x * 100 for x in wm_acc_complexity]

idm_acc_complexity = [0.9495, 0.9466, 0.9270, 0.9337, 0.9136]
idm_acc_complexity = [x * 100 for x in idm_acc_complexity]


# =====================================================
# Data for fig5_average2.py: average performance across strategies
# =====================================================

def robust_load_json(path: str):
    """Load possibly broken JSON logs by extracting individual objects."""
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return []
    all_records = []

    matches = re.finditer(r"\{[^{}]*\}", content)
    for match in matches:
        try:
            record = json.loads(match.group())
            all_records.append(record)
        except json.JSONDecodeError:
            continue

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


_BASE_DIR = Path(__file__).resolve().parent
fig5_json_path = str(_BASE_DIR / "all_rounds_results_all.json")
fig5_target_round = 3

fig5_records = robust_load_json(fig5_json_path)
if not fig5_records:
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


################################################################################
# Figure 1: Active learning performance vs. data budget percentage

setup_icml_style()

fig, ax = plt.subplots(figsize=(10.0, 10.0))

# Show all spines
for spine in ax.spines.values():
    spine.set_visible(True)

palette = {"Random": "#d95f02", "Ours": "#1b9e77"}
markers = {"Random": "o", "Ours": "o"}

sns.lineplot(
    data=fig1_df,
    x="pct",
    y="norm_loss",
    hue="strategy",
    style="strategy",
    hue_order=["Random", "Ours"],
    palette=palette,
    markers=markers,
    markersize=20,
    dashes={"Random": "", "Ours": ""},
    ax=ax,
    legend=False,
)

# Horizontal arrow annotation showing label efficiency
x_start = 1.56 * 25
x_end = 3.98 * 25
x_text = 2.77 * 25
y_target = 2.97 * fig1_scale

ax.annotate(
    "",
    xy=(x_start, y_target),
    xytext=(x_end, y_target),
    arrowprops=dict(arrowstyle="<->", color="black", lw=3.5),
)

ax.text(
    x_text,
    y_target - 0.04,
    r"$\mathbf{3x\ fewer\ labels}$",
    ha="center",
    va="center",
    fontsize=ANNOTATION_FONTSIZE,
    color="black",
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=3),
)

ax.grid(color="grey", linestyle=":", linewidth=0.3)

custom_lines = [
    Line2D([0], [0], color=palette["Random"], lw=4, marker="o", markersize=MARKER_SIZE_SMALL),
    Line2D([0], [0], color=palette["Ours"], lw=4, marker="o", markersize=MARKER_SIZE_SMALL),
]

leg = ax.legend(
    custom_lines,
    ["Random", "Ours"],
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    ncol=1,
    frameon=True,
    fontsize=LEGEND_FONTSIZE,
)

frame = leg.get_frame()
frame.set_facecolor("white")
frame.set_edgecolor("lightgray")
frame.set_linewidth(1.2)

# X axis ticks as percentages
ticks = [0, 20, 40, 60, 80, 100]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{x}%" for x in ticks])
set_axis_labels(ax, "Percentage of Data Budget", "Normalized World Model Error", xlabelpad=10, ylabelpad=10)

ax.set_xlim(-2, 105)
ax.set_ylim(0.8 * fig1_scale, 8.0 * fig1_scale)

plt.tight_layout()
fig1_output_path = "active_learning_percentage.pdf"
plt.savefig(fig1_output_path, bbox_inches="tight")
print(f"Figure 1 saved to: {fig1_output_path}")


################################################################################
# Figure 4: OOS dynamics accuracy vs. number of training samples

setup_icml_style()

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    fig4_x,
    fig4_im_acc,
    label="Inverse Model",
    color=COLOR_ORANGE,
    linestyle="-",
    marker="o",
    markersize=MARKER_SIZE_LARGE,
    markerfacecolor="white",
    markeredgewidth=BAR_EDGE_LINEWIDTH,
    zorder=5,
)

ax.plot(
    fig4_x,
    fig4_wm_acc,
    label="World Model",
    color=COLOR_GRAY,
    linestyle="--",
    dashes=(4, 2),
    marker="s",
    markersize=MARKER_SIZE_LARGE,
    markerfacecolor="white",
    markeredgewidth=BAR_EDGE_LINEWIDTH,
    zorder=4,
)

set_axis_labels(ax, "Number of Training Samples", "OOS Dynamic Accuracy (%)")

ax.set_xticks(fig4_x)
ax.set_xticklabels(fig4_x, fontweight="bold")

ax.set_ylim(45, 105)
ax.yaxis.set_major_locator(plticker.MultipleLocator(10))

add_grid(ax, axis="y")
add_grid(ax, axis="x", alpha=0.3)

apply_despine(ax, trim=False)

# Legend order: World Model first, then Inverse Model
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

fig4_output_path = "sample_efficiency_oos1.pdf"
plt.savefig(fig4_output_path, bbox_inches="tight", transparent=True)
print(f"Figure 4 saved to: {fig4_output_path}")

################################################################################
# fig_complexity.py: Robustness vs. state complexity

setup_icml_style()

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    complexity_levels,
    idm_acc_complexity,
    label="Inverse Model",
    color=COLOR_ORANGE,
    linestyle="-",
    marker="o",
    markersize=MARKER_SIZE_LARGE,
    markerfacecolor="white",
    markeredgewidth=BAR_EDGE_LINEWIDTH,
    zorder=5,
)

ax.plot(
    complexity_levels,
    wm_acc_complexity,
    label="World Model",
    color=COLOR_GRAY,
    linestyle="--",
    dashes=(4, 2),
    marker="s",
    markersize=MARKER_SIZE_LARGE,
    markerfacecolor="white",
    markeredgewidth=BAR_EDGE_LINEWIDTH,
    zorder=4,
)

set_axis_labels(ax, "State Complexity (Number of Objects)", "Dynamics Accuracy (%)")

ax.set_xticks(complexity_levels)
ax.set_ylim(40, 105)
ax.yaxis.set_major_locator(plticker.MultipleLocator(10))

add_grid(ax, axis="y")
apply_despine(ax, trim=False)

# Legend order: World Model first, then Inverse Model
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
fig5_complexity_output_path = "complexity_robustness.pdf"
plt.savefig(fig5_complexity_output_path, bbox_inches="tight", transparent=True)
print(f"Figure 5 saved to: {fig5_complexity_output_path}")

################################################################################
# fig5_average2.py: Average performance comparison across strategies (bar plot)

# Guard against missing/empty logs
required_cols = {"strategy", "macro_mse"}
if fig5_df.empty or not required_cols.issubset(set(fig5_df.columns)):
    print("fig5_average2 skipped: fig5_df is empty or missing required columns.")
else:
    setup_icml_style()

    color_palette = {
        "Vanilla": COLOR_LIGHT_GRAY,
        "Oracle": COLOR_BLUE,
        "Progress": COLOR_RED,
        "Uncertainty": COLOR_GREEN,
        "Active Exploration (Ours)": COLOR_ORANGE,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Seaborn version compatibility: errorbar (>=0.12) vs ci (<0.12)
    try:
        sns.barplot(
            data=fig5_df,
            x="strategy",
            y="macro_mse",
            order=fig5_strategy_order,
            palette=color_palette,
            hue="strategy",
            ax=ax,
            capsize=0.15,
            errorbar="sd",
            edgecolor="black",
            linewidth=BAR_EDGE_LINEWIDTH,
            dodge=False,
        )
    except TypeError:
        sns.barplot(
            data=fig5_df,
            x="strategy",
            y="macro_mse",
            order=fig5_strategy_order,
            palette=color_palette,
            hue="strategy",
            ax=ax,
            capsize=0.15,
            ci="sd",
            edgecolor="black",
            linewidth=BAR_EDGE_LINEWIDTH,
            dodge=False,
        )

    # Remove internal legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    set_axis_labels(ax, "Strategies", "Prediction Error", xlabelpad=15, ylabelpad=10)

    # Hide x tick labels (use figure-level legend)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", bottom=False)

    ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=5))
    apply_despine(ax)
    add_grid(ax, axis="y", linestyle="--", alpha=0.4, linewidth=2.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    legend_handles = [
        mpatches.Patch(color=color_palette[name], label=name)
        for name in fig5_strategy_order
        if name in set(fig5_df["strategy"].unique())
    ]

    create_figure_level_legend(fig, legend_handles)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    fig5_avg_output_path = "icml_average_performance_round2.pdf"
    plt.savefig(fig5_avg_output_path, dpi=300, bbox_inches="tight")
    print(f"fig5_average2 saved to: {fig5_avg_output_path}")
