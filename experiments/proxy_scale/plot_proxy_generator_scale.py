# -*- coding: utf-8 -*-
"""Publication plot for Qwen2.5 DP-SPD proxy-generator scale results.

The earlier exploratory plot used per-example min--max bands. Those bands are
not informative for heavy-tailed PPL or binary MIA outcomes. This version uses
group-level files to construct interpretable uncertainty intervals:

* ROUGE-L: non-parametric bootstrap 95% CI of the mean.
* PPL: token-weighted corpus PPL reconstructed from per-group suffix PPL and
  suffix length, with a bootstrap 95% CI.
* MIA: Wilson 95% binomial confidence intervals.
* Time: one measured full sweep per scale, shown without an invented error bar.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ScaleResult:
    label: str
    size_b: float
    summary: Dict[str, str]
    rouge_mean: float
    rouge_ci: Tuple[float, float]
    ppl: float
    ppl_ci: Tuple[float, float]
    mia_ppl: float
    mia_ppl_ci: Tuple[float, float]
    mia_min_k: float
    mia_min_k_ci: Tuple[float, float]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create publication-ready Qwen2.5 proxy-scale plots.")
    parser.add_argument(
        "--summary_glob",
        type=str,
        default="outputs/proxy_generator_scale/*_summary.csv",
        help="Glob containing one summary CSV per proxy scale.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/proxy_generator_scale/qwen25_proxy_generator_scale_final.png",
    )
    parser.add_argument(
        "--pdf_output",
        type=str,
        default=None,
        help="Optional PDF. Defaults to --output with a .pdf suffix.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--title_prefix", type=str, default="Effect of proxy generator (Qwen2.5 family) scale")
    return parser


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def summary_to_groups_path(summary_path: Path) -> Path:
    suffix = "_summary.csv"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Unexpected summary filename: {summary_path}")
    return summary_path.with_name(summary_path.name[: -len(suffix)] + "_groups.jsonl")


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    if values.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    distribution = values[indices].mean(axis=1)
    return point, (float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975)))


def bootstrap_token_weighted_ppl_ci(
    ppls: np.ndarray,
    token_counts: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[float, Tuple[float, float]]:
    """Recover corpus PPL from group PPL values and private-suffix lengths."""
    if ppls.size == 0 or token_counts.size != ppls.size:
        return float("nan"), (float("nan"), float("nan"))
    log_ppl = np.log(np.clip(ppls, 1e-12, None))
    point = float(np.exp(np.average(log_ppl, weights=token_counts)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, ppls.size, size=(samples, ppls.size))
    sampled_nll = log_ppl[indices]
    sampled_counts = token_counts[indices]
    distribution = np.exp((sampled_nll * sampled_counts).sum(axis=1) / sampled_counts.sum(axis=1))
    return point, (float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975)))


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def load_scale_result(summary_path: Path, args, index: int) -> ScaleResult:
    with open(summary_path, encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle), None)
    if summary is None:
        raise ValueError(f"Summary is empty: {summary_path}")
    groups_path = summary_to_groups_path(summary_path)
    if not groups_path.exists():
        raise FileNotFoundError(f"Group metrics required for confidence intervals are missing: {groups_path}")
    with open(groups_path, encoding="utf-8") as handle:
        groups = [json.loads(line) for line in handle if line.strip()]
    if not groups:
        raise ValueError(f"No group records found in {groups_path}")

    rouge = np.asarray([float(group["surface_dummy_rouge_l"]) for group in groups], dtype=np.float64)
    ppls = np.asarray([float(group["proxy_oracle_suffix_ppl"]) for group in groups], dtype=np.float64)
    suffix_lengths = np.asarray([float(group["suffix_len"]) for group in groups], dtype=np.float64)
    mia_ppl = np.asarray([float(group["mia_ppl_correct"]) for group in groups], dtype=np.float64)
    mia_min_k = np.asarray([float(group["mia_min_k_correct"]) for group in groups], dtype=np.float64)

    rouge_mean, rouge_ci = bootstrap_mean_ci(rouge, args.bootstrap_samples, args.bootstrap_seed + index * 17)
    ppl, ppl_ci = bootstrap_token_weighted_ppl_ci(
        ppls, suffix_lengths, args.bootstrap_samples, args.bootstrap_seed + index * 17 + 1
    )
    trials = len(groups)
    mia_ppl_rate = float(mia_ppl.mean())
    mia_min_k_rate = float(mia_min_k.mean())
    return ScaleResult(
        label=str(summary.get("proxy_label", "?")),
        size_b=as_float(summary, "proxy_size_b"),
        summary=summary,
        rouge_mean=rouge_mean,
        rouge_ci=rouge_ci,
        ppl=ppl,
        ppl_ci=ppl_ci,
        mia_ppl=mia_ppl_rate,
        mia_ppl_ci=wilson_ci(int(mia_ppl.sum()), trials),
        mia_min_k=mia_min_k_rate,
        mia_min_k_ci=wilson_ci(int(mia_min_k.sum()), trials),
    )


def error_yerr(points: Sequence[float], intervals: Sequence[Tuple[float, float]]) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    lower = np.asarray([interval[0] for interval in intervals], dtype=np.float64)
    upper = np.asarray([interval[1] for interval in intervals], dtype=np.float64)
    return np.vstack([point_array - lower, upper - point_array])


def plot_line_with_ci(ax, x, points, intervals, *, color: str, marker: str, label: str, linestyle: str = "-") -> None:
    ax.errorbar(
        x,
        points,
        yerr=error_yerr(points, intervals),
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.35,
        markersize=7,
        capsize=4,
        elinewidth=1.5,
        label=label,
    )


def plot_publication_figure(results: Sequence[ScaleResult], args) -> plt.Figure:
    labels = [result.label for result in results]
    x = np.arange(len(results))
    plt.rcParams.update({"font.size": 12, "axes.titleweight": "regular"})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=True)
    fig.suptitle(args.title_prefix, fontsize=16)

    ax = axes[0, 0]
    rouge = [result.rouge_mean for result in results]
    rouge_ci = [result.rouge_ci for result in results]
    plot_line_with_ci(ax, x, rouge, rouge_ci, color="#2b6cb0", marker="s", label="Mean ± bootstrap 95% CI")
    ax.set_title("(a) Server-visible dummy continuation quality")
    ax.set_ylabel("ROUGE-L (higher is better)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", linestyle=":", alpha=0.65)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[0, 1]
    ppls = [result.ppl for result in results]
    ppl_ci = [result.ppl_ci for result in results]
    plot_line_with_ci(ax, x, ppls, ppl_ci, color="#805ad5", marker="s", label="Token-weighted PPL ± bootstrap 95% CI")
    ax.set_title("(b) Held-out suffix modeling quality")
    ax.set_ylabel("Token-weighted PPL (lower is better; log scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", alpha=0.65, which="both")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 0]
    seconds = np.asarray([as_float(result.summary, "client_proxy_seconds_per_group") for result in results])
    throughput = np.asarray([as_float(result.summary, "client_proxy_dummy_tokens_per_second") for result in results])
    ax.plot(x, seconds, color="#dd6b20", marker="s", linewidth=2.35, markersize=7, label="Seconds / group (one run)")
    ax.set_title("(c) Client continuation-generation cost")
    ax.set_ylabel("Seconds per candidate group (lower is better)")
    ax.grid(axis="y", linestyle=":", alpha=0.65)
    secondary = ax.twinx()
    secondary.plot(x, throughput, color="#38a169", marker="o", linestyle="--", linewidth=2.1, markersize=7, label="Dummy tokens / s (one run)")
    secondary.set_ylabel("Dummy tokens / second (higher is better)")
    handles, labels_legend = ax.get_legend_handles_labels()
    handles2, labels_legend2 = secondary.get_legend_handles_labels()
    ax.legend(handles + handles2, labels_legend + labels_legend2, loc="upper left", frameon=False)

    ax = axes[1, 1]
    min_k = [result.mia_min_k for result in results]
    min_k_ci = [result.mia_min_k_ci for result in results]
    ppl_asr = [result.mia_ppl for result in results]
    ppl_asr_ci = [result.mia_ppl_ci for result in results]
    plot_line_with_ci(ax, x, min_k, min_k_ci, color="#c53030", marker="s", label="MIA Min-K ASR ± Wilson 95% CI")
    plot_line_with_ci(ax, x, ppl_asr, ppl_asr_ci, color="#dd6b20", marker="o", linestyle="--", label="MIA PPL ASR ± Wilson 95% CI")
    random_baseline = as_float(results[0].summary, "mia_random_baseline")
    if np.isfinite(random_baseline):
        ax.axhline(random_baseline, color="black", linestyle=":", linewidth=1.6, label=f"Random ({random_baseline:.2f})")
    max_upper = max([interval[1] for interval in min_k_ci + ppl_asr_ci] + [random_baseline])
    ax.set_ylim(0.0, max(0.14, max_upper * 1.22))
    ax.set_title("(d) Candidate-identification privacy attack")
    ax.set_ylabel("MIA ASR (lower is better)")
    ax.grid(axis="y", linestyle=":", alpha=0.65)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Proxy generator scale")
    return fig


def write_plot_summary(path: Path, results: Sequence[ScaleResult]) -> None:
    fields = [
        "proxy_label", "proxy_size_b", "groups",
        "surface_dummy_rouge_l_mean", "surface_dummy_rouge_l_ci95_low", "surface_dummy_rouge_l_ci95_high",
        "proxy_token_weighted_ppl", "proxy_token_weighted_ppl_ci95_low", "proxy_token_weighted_ppl_ci95_high",
        "mia_ppl_asr", "mia_ppl_asr_wilson95_low", "mia_ppl_asr_wilson95_high",
        "mia_min_k_asr", "mia_min_k_asr_wilson95_low", "mia_min_k_asr_wilson95_high",
        "seconds_per_group_one_run", "dummy_tokens_per_second_one_run",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "proxy_label": result.label,
                "proxy_size_b": result.size_b,
                "groups": result.summary.get("groups", ""),
                "surface_dummy_rouge_l_mean": f"{result.rouge_mean:.8f}",
                "surface_dummy_rouge_l_ci95_low": f"{result.rouge_ci[0]:.8f}",
                "surface_dummy_rouge_l_ci95_high": f"{result.rouge_ci[1]:.8f}",
                "proxy_token_weighted_ppl": f"{result.ppl:.8f}",
                "proxy_token_weighted_ppl_ci95_low": f"{result.ppl_ci[0]:.8f}",
                "proxy_token_weighted_ppl_ci95_high": f"{result.ppl_ci[1]:.8f}",
                "mia_ppl_asr": f"{result.mia_ppl:.8f}",
                "mia_ppl_asr_wilson95_low": f"{result.mia_ppl_ci[0]:.8f}",
                "mia_ppl_asr_wilson95_high": f"{result.mia_ppl_ci[1]:.8f}",
                "mia_min_k_asr": f"{result.mia_min_k:.8f}",
                "mia_min_k_asr_wilson95_low": f"{result.mia_min_k_ci[0]:.8f}",
                "mia_min_k_asr_wilson95_high": f"{result.mia_min_k_ci[1]:.8f}",
                "seconds_per_group_one_run": result.summary.get("client_proxy_seconds_per_group", ""),
                "dummy_tokens_per_second_one_run": result.summary.get("client_proxy_dummy_tokens_per_second", ""),
            })


def run() -> None:
    args = get_parser().parse_args()
    summary_paths = [Path(path) for path in sorted(glob.glob(args.summary_glob))]
    if not summary_paths:
        raise FileNotFoundError(f"No summary CSV matched: {args.summary_glob}")
    results = [load_scale_result(path, args, index) for index, path in enumerate(summary_paths)]
    results.sort(key=lambda result: (result.size_b, result.label))
    figure = plot_publication_figure(results, args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    pdf_output = Path(args.pdf_output) if args.pdf_output else output.with_suffix(".pdf")
    figure.savefig(pdf_output, bbox_inches="tight")
    plot_summary = output.with_name(output.stem + "_metrics.csv")
    write_plot_summary(plot_summary, results)
    print(f"[Saved] {output}")
    print(f"[Saved] {pdf_output}")
    print(f"[Saved] {plot_summary}")


if __name__ == "__main__":
    run()
