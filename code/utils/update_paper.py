#!/usr/bin/env python3
"""
Auto-update paper/main.tex with MoleculeNet benchmark results.
Reads results/molnet_benchmark.json and generates LaTeX tables/sections.
"""

import os
import json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(ROOT, "results", "molnet_benchmark.json")
PAPER_PATH = os.path.join(ROOT, "paper", "main.tex")


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def fmt(val, prec=4):
    return f"{val:.{prec}f}"


def fmt_pm(mean, std, prec=4):
    return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"


# ================================================================
# Generate LaTeX tables
# ================================================================
def gen_main_table(results):
    """Table 1: Main results on MoleculeNet benchmarks."""
    bench = results["benchmark"]
    datasets = list(bench.keys())
    models = ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "RF-Morgan", "MAGNet"]

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance comparison on MoleculeNet benchmarks (ROC-AUC $\pm$ std over 10 seeds with scaffold splitting). Best results in \textbf{bold}, second-best \underline{underlined}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & " + " & ".join(r"\textbf{" + d + "}" for d in datasets) + r" \\")
    lines.append(r"\midrule")

    # Find best and second-best per dataset
    best = {}; second = {}
    for ds in datasets:
        vals = [(m, bench[ds][m]["mean_auc"]) for m in models if m in bench[ds]]
        vals.sort(key=lambda x: -x[1])
        best[ds] = vals[0][0]
        second[ds] = vals[1][0] if len(vals) > 1 else None

    for model in models:
        cells = []
        for ds in datasets:
            if model not in bench[ds]:
                cells.append("--")
                continue
            r = bench[ds][model]
            s = fmt_pm(r["mean_auc"], r["std_auc"])
            if model == best[ds]:
                cells.append(r"\textbf{" + s + "}")
            elif model == second[ds]:
                cells.append(r"\underline{" + s + "}")
            else:
                cells.append(s)
        lines.append(model + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_efficiency_table(results):
    """Table 2: Parameter counts and training time."""
    bench = results["benchmark"]
    # Use first dataset for params/time
    ds = list(bench.keys())[0]
    models = ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "MAGNet"]

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Model complexity and training efficiency.}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Parameters} & \textbf{Time/seed (s)} & \textbf{Hidden dim} \\")
    lines.append(r"\midrule")

    for model in models:
        if model not in bench[ds]:
            continue
        r = bench[ds][model]
        lines.append(f"{model} & {r['params']:,} & {r['mean_time']:.1f} & 128 \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_cv_table(results):
    """Table 3: 5-fold cross-validation results."""
    if "cross_validation" not in results:
        return "% Cross-validation results not available"
    cv = results["cross_validation"]
    datasets = list(cv.keys())
    models = ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "MAGNet"]

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{5-fold cross-validation results (ROC-AUC $\pm$ std).}")
    lines.append(r"\label{tab:cv_results}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & " + " & ".join(r"\textbf{" + d + "}" for d in datasets) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        cells = []
        for ds in datasets:
            if model not in cv[ds]:
                cells.append("--")
                continue
            r = cv[ds][model]
            cells.append(fmt_pm(r["mean_auc"], r["std_auc"]))
        lines.append(model + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_significance_table(results):
    """Table 4: Significance tests."""
    if "significance" not in results:
        return "% Significance results not available"
    sig = results["significance"]
    datasets = list(sig.keys())
    baselines = list(sig[datasets[0]].keys()) if datasets else []

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical significance of MAGNet vs baselines (paired $t$-test, $p < 0.05$ marked with $^*$).}")
    lines.append(r"\label{tab:significance}")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Baseline} & " + " & ".join(r"\textbf{" + d + "}" for d in datasets) + r" \\")
    lines.append(r"\midrule")

    for baseline in baselines:
        cells = []
        for ds in datasets:
            if baseline not in sig[ds]:
                cells.append("--")
                continue
            p = sig[ds][baseline]["t_pvalue"]
            s = f"{p:.4f}"
            if p < 0.05:
                s += r"$^*$"
            cells.append(s)
        lines.append(baseline + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_ablation_tables(results):
    """Tables for ablation studies."""
    tables = []
    for key, label, caption in [
        ("ablation_scales", "Scales", "Effect of the number of scales on BBBP."),
        ("ablation_heads", "Heads", "Effect of the number of attention heads on BBBP."),
    ]:
        if key not in results:
            continue
        data = results[key]
        xs = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)

        lines = []
        lines.append(r"\begin{table}[!t]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + caption + r"}")
        lines.append(r"\label{tab:" + key + r"}")
        lines.append(r"\begin{tabular}{lcc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{" + label + r"} & \textbf{ROC-AUC} & \textbf{Std} \\")
        lines.append(r"\midrule")
        for x in xs:
            lines.append(f"{x} & {data[x]['mean']:.4f} & {data[x]['std']:.4f} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


# ================================================================
# Main
# ================================================================
def main():
    results = load_results()

    print("=" * 60)
    print("Generated LaTeX for paper/main.tex")
    print("=" * 60)

    print("\n% --- Table 1: Main Results ---")
    print(gen_main_table(results))

    print("\n% --- Table 2: Efficiency ---")
    print(gen_efficiency_table(results))

    print("\n% --- Table 3: Cross-Validation ---")
    print(gen_cv_table(results))

    print("\n% --- Table 4: Significance ---")
    print(gen_significance_table(results))

    print("\n% --- Ablation Tables ---")
    print(gen_ablation_tables(results))


if __name__ == "__main__":
    main()
