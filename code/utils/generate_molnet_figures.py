#!/usr/bin/env python3
"""
Generate publication-quality figures from MoleculeNet benchmark results.
Reads results/molnet_benchmark.json and outputs PDF figures to paper/figures/.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ── Nature-style settings ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})

COLORS = {
    "MAGNet":   "#D62728",  # red
    "GCN":      "#1F77B4",  # blue
    "GAT":      "#FF7F0E",  # orange
    "GIN":      "#2CA02C",  # green
    "GraphSAGE": "#9467BD", # purple
    "MPNN":     "#8C564B",  # brown
    "RF-Morgan": "#7F7F7F", # grey
}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(ROOT, "results", "molnet_benchmark.json")
FIGURES_DIR = os.path.join(ROOT, "paper", "figures")


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ================================================================
# Figure 1: Main results – grouped bar chart across datasets
# ================================================================
def fig_main_results(results):
    benchmark = results["benchmark"]
    datasets = list(benchmark.keys())
    models = ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "MAGNet", "RF-Morgan"]
    models = [m for m in models if m in benchmark[datasets[0]]]

    n_ds = len(datasets)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(7, 3.2))

    for i, model in enumerate(models):
        means = [benchmark[ds][model]["mean_auc"] for ds in datasets]
        stds  = [benchmark[ds][model]["std_auc"]  for ds in datasets]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      label=model, color=COLORS.get(model, "#333"),
                      capsize=2, edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 0.8})
        # annotate value on top
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=5.5,
                    rotation=90 if n_models > 5 else 0)

    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right", ncol=2, frameon=False)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "molnet_main_results.pdf"))
    plt.close(fig)
    print("  ✓ molnet_main_results.pdf")


# ================================================================
# Figure 2: Per-dataset detail with seed scatter
# ================================================================
def fig_seed_scatter(results):
    benchmark = results["benchmark"]
    for ds_name, ds_res in benchmark.items():
        models = [m for m in ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "MAGNet"]
                  if m in ds_res]
        fig, ax = plt.subplots(figsize=(5, 2.8))
        for i, model in enumerate(models):
            aucs = ds_res[model]["seed_aucs"]
            ax.scatter([i] * len(aucs), aucs, color=COLORS.get(model, "#333"),
                       alpha=0.5, s=20, zorder=3)
            mean = ds_res[model]["mean_auc"]
            std  = ds_res[model]["std_auc"]
            ax.errorbar(i, mean, yerr=std, fmt="D", color=COLORS.get(model, "#333"),
                        markersize=6, capsize=4, elinewidth=1.2, zorder=4)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{ds_name} — per-seed results")
        ax.set_ylim(0.5, 1.05)
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"molnet_scatter_{ds_name}.pdf"))
        plt.close(fig)
        print(f"  ✓ molnet_scatter_{ds_name}.pdf")


# ================================================================
# Figure 3: Cross-validation results
# ================================================================
def fig_cross_validation(results):
    if "cross_validation" not in results:
        return
    cv = results["cross_validation"]
    datasets = list(cv.keys())
    models = ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "MAGNet"]
    models = [m for m in models if m in cv[datasets[0]]]

    fig, axes = plt.subplots(1, len(datasets), figsize=(2.5 * len(datasets), 3),
                              sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        means = [cv[ds][m]["mean_auc"] for m in models]
        stds  = [cv[ds][m]["std_auc"]  for m in models]
        colors = [COLORS.get(m, "#333") for m in models]
        bars = ax.barh(range(len(models)), means, xerr=stds,
                       color=colors, edgecolor="white", linewidth=0.5,
                       capsize=3, error_kw={"linewidth": 0.8})
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel("ROC-AUC")
        ax.set_title(ds)
        ax.set_xlim(0.5, 1.05)
        ax.axvline(x=1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{m:.3f}", va="center", fontsize=7)

    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "molnet_cross_validation.pdf"))
    plt.close(fig)
    print("  ✓ molnet_cross_validation.pdf")


# ================================================================
# Figure 4: Significance heatmap
# ================================================================
def fig_significance(results):
    if "significance" not in results:
        return
    sig = results["significance"]
    datasets = list(sig.keys())
    baselines = list(sig[datasets[0]].keys()) if datasets else []
    if not baselines:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(2.5 * len(datasets), 2.5),
                              sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        pvals = [sig[ds][b]["t_pvalue"] for b in baselines]
        colors = ["#2ca02c" if p < 0.05 else "#d62728" for p in pvals]
        bars = ax.barh(range(len(baselines)), [-np.log10(max(p, 1e-10)) for p in pvals],
                       color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(baselines)))
        ax.set_yticklabels(baselines)
        ax.set_xlabel("-log₁₀(p-value)")
        ax.set_title(ds)
        ax.axvline(x=-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8,
                   label="p=0.05")
        ax.legend(fontsize=6, loc="lower right")
        for bar, p in zip(bars, pvals):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"p={p:.4f}", va="center", fontsize=6)

    axes[0].invert_yaxis()
    fig.suptitle("MAGNet vs baselines (paired t-test)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "molnet_significance.pdf"))
    plt.close(fig)
    print("  ✓ molnet_significance.pdf")


# ================================================================
# Figure 5: Ablation – scales & heads
# ================================================================
def fig_ablation(results):
    for key, xlabel, fname in [
        ("ablation_scales", "Number of scales", "molnet_ablation_scales.pdf"),
        ("ablation_heads",  "Number of attention heads", "molnet_ablation_heads.pdf"),
    ]:
        if key not in results:
            continue
        data = results[key]
        xs = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)
        means = [data[x]["mean"] for x in xs]
        stds  = [data[x]["std"]  for x in xs]

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ax.errorbar(range(len(xs)), means, yerr=stds, fmt="-o", color=COLORS["MAGNet"],
                    capsize=4, markersize=7, elinewidth=1.2, markerfacecolor="white",
                    markeredgewidth=1.5)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ROC-AUC (BBBP)")
        ax.set_ylim(0.7, 1.05)
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.annotate(f"{m:.3f}±{s:.3f}", (i, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        plt.close(fig)
        print(f"  ✓ {fname}")


# ================================================================
# Figure 6: Hyperparameter sensitivity
# ================================================================
def fig_hyperparameter(results):
    if "hyperparameter_search" not in results:
        return
    hp = results["hyperparameter_search"]
    params = list(hp.keys())
    fig, axes = plt.subplots(1, len(params), figsize=(2.5 * len(params), 3))
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        vals = sorted(hp[param].keys(), key=lambda x: float(x))
        means = [hp[param][v]["mean"] for v in vals]
        stds  = [hp[param][v]["std"]  for v in vals]
        ax.errorbar(range(len(vals)), means, yerr=stds, fmt="-o",
                    color=COLORS["MAGNet"], capsize=4, markersize=6,
                    elinewidth=1.2, markerfacecolor="white", markeredgewidth=1.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals, fontsize=7)
        ax.set_xlabel(param)
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.7, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "molnet_hyperparameter.pdf"))
    plt.close(fig)
    print("  ✓ molnet_hyperparameter.pdf")


# ================================================================
# Figure 7: Parameter-matched comparison
# ================================================================
def fig_parameter_matched(results):
    if "parameter_matched" not in results:
        return
    pm = results["parameter_matched"]
    models = list(pm.keys())
    fig, ax1 = plt.subplots(figsize=(5, 3))

    x = range(len(models))
    means = [pm[m]["mean_auc"] for m in models]
    stds  = [pm[m]["std_auc"]  for m in models]
    params = [pm[m]["params"] for m in models]
    colors = [COLORS.get(m, "#333") for m in models]

    bars = ax1.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                   linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
    ax1.set_ylabel("ROC-AUC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.set_ylim(0.5, 1.05)

    # annotate params
    for bar, p in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width() / 2, 0.52,
                 f"{p/1000:.0f}K", ha="center", fontsize=6, color="white",
                 fontweight="bold")

    ax1.set_title("Parameter-matched comparison (BBBP)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "molnet_param_matched.pdf"))
    plt.close(fig)
    print("  ✓ molnet_param_matched.pdf")


# ================================================================
# Main
# ================================================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results = load_results()

    print("Generating MoleculeNet figures...")
    fig_main_results(results)
    fig_seed_scatter(results)
    fig_cross_validation(results)
    fig_significance(results)
    fig_ablation(results)
    fig_hyperparameter(results)
    fig_parameter_matched(results)
    print("Done!")


if __name__ == "__main__":
    main()
