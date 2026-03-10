from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


EXTERNAL_EXPERIMENTS = [
    ("完整模型", "checkpoints/ch5_main_strict_poseaux_{version}/cross_fold_summary.json"),
    ("AvgPool", "checkpoints/ch5_baseline_avgpool_pure_{version}/cross_fold_summary.json"),
    ("BiGRU", "checkpoints/ch5_baseline_bigru_{version}/cross_fold_summary.json"),
    ("BiLSTM", "checkpoints/ch5_baseline_bilstm_{version}/cross_fold_summary.json"),
    ("Temporal Transformer", "checkpoints/ch5_baseline_temporal_transformer_{version}/cross_fold_summary.json"),
    ("Single-stream Mamba", "checkpoints/ch5_baseline_single_mamba_{version}/cross_fold_summary.json"),
]

INTERNAL_EXPERIMENTS = [
    ("完整模型", "checkpoints/ch5_main_strict_poseaux_{version}/cross_fold_summary.json"),
    ("去除质量门控与不确定性加权", "checkpoints/ch5_ablation_no_quality_uncertainty_{version}/cross_fold_summary.json"),
    ("单向时序建模", "checkpoints/ch5_ablation_unidirectional_{version}/cross_fold_summary.json"),
    ("去除非身份流", "checkpoints/ch5_ablation_no_pose_stream_{version}/cross_fold_summary.json"),
    ("关闭 MINE", "checkpoints/ch5_ablation_no_mi_loss_{version}/cross_fold_summary.json"),
    ("去除时序顺序辅助监督", "checkpoints/ch5_ablation_no_pose_aux_{version}/cross_fold_summary.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Chapter 5 result tables")
    parser.add_argument("--version", type=str, default="v2", help="experiment version suffix")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory used to save summary artifacts",
    )
    parser.add_argument(
        "--uncertainty-dir",
        type=Path,
        default=None,
        help="uncertainty analysis directory; defaults to runs/ch5_uncertainty_<version>",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table_rows(specs: List[tuple[str, str]], version: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, pattern in specs:
        path = REPO_ROOT / pattern.format(version=version)
        payload = load_summary(path)
        if payload is None:
            continue
        summary = payload["summary"]
        rows.append(
            {
                "method": label,
                "source_path": str(path.relative_to(REPO_ROOT)),
                "best_val_mAP_mean": float(summary["best_val_mAP"]["mean"]),
                "best_val_mAP_std": float(summary["best_val_mAP"]["std"]),
                "test_mAP_mean": float(summary["test_mAP"]["mean"]),
                "test_mAP_std": float(summary["test_mAP"]["std"]),
                "test_rank1_mean": float(summary["test_rank-1"]["mean"]),
                "test_rank1_std": float(summary["test_rank-1"]["std"]),
            }
        )
    return rows


def render_markdown_table(title: str, rows: List[Dict[str, object]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| 方法 | best val mAP | test mAP | test Rank-1 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | "
            f"{row['best_val_mAP_mean'] * 100:.2f} ± {row['best_val_mAP_std'] * 100:.2f} | "
            f"{row['test_mAP_mean'] * 100:.2f} ± {row['test_mAP_std'] * 100:.2f} | "
            f"{row['test_rank1_mean'] * 100:.2f} ± {row['test_rank1_std'] * 100:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_latex_table(caption: str, label: str, rows: List[Dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    方法 & best val mAP(\%) & test mAP(\%) & Rank-1(\%) \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            f"{row['method']} & "
            f"{row['best_val_mAP_mean'] * 100:.2f}$\\pm${row['best_val_mAP_std'] * 100:.2f} & "
            f"{row['test_mAP_mean'] * 100:.2f}$\\pm${row['test_mAP_std'] * 100:.2f} & "
            f"{row['test_rank1_mean'] * 100:.2f}$\\pm${row['test_rank1_std'] * 100:.2f} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_uncertainty_latex(bucket_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{按查询片段平均不确定性分桶后的检索统计}",
        r"  \label{tab:ch5_sigma_bucket}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    分桶 & 样本数 & 平均$\sigma^2$ & Rank-1(\%) & mAP(\%) \\",
        r"    \midrule",
    ]
    for _, row in bucket_df.iterrows():
        lines.append(
            "    "
            f"{row['bucket']} & {int(row['num_queries'])} & {row['sigma2_mean']:.4f} & "
            f"{row['rank1_rate'] * 100.0:.2f} & {row['mAP_mean'] * 100.0:.2f} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_bucket_markdown(bucket_df: pd.DataFrame) -> str:
    lines = [
        "| 分桶 | 样本数 | 平均 sigma2 | Rank-1 | mAP |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in bucket_df.iterrows():
        lines.append(
            f"| {row['bucket']} | {int(row['num_queries'])} | {row['sigma2_mean']:.4f} | "
            f"{row['rank1_rate'] * 100.0:.2f} | {row['mAP_mean'] * 100.0:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / (args.output_dir or Path(f"runs/ch5_summary_{args.version}"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    external_rows = build_table_rows(EXTERNAL_EXPERIMENTS, args.version)
    internal_rows = build_table_rows(INTERNAL_EXPERIMENTS, args.version)

    uncertainty_dir = (REPO_ROOT / (args.uncertainty_dir or Path(f"runs/ch5_uncertainty_{args.version}"))).resolve()
    bucket_path = uncertainty_dir / "bucket_summary.csv"
    bucket_df = pd.read_csv(bucket_path) if bucket_path.exists() else pd.DataFrame()

    payload = {
        "version": args.version,
        "external_baselines": external_rows,
        "internal_ablations": internal_rows,
        "uncertainty_bucket_path": str(bucket_path.relative_to(REPO_ROOT)) if bucket_path.exists() else None,
        "uncertainty_buckets": bucket_df.to_dict(orient="records") if not bucket_df.empty else [],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    markdown_parts = ["# Chapter 5 Result Summary", ""]
    if external_rows:
        markdown_parts.append(render_markdown_table("外部基线对比", external_rows))
    if internal_rows:
        markdown_parts.append(render_markdown_table("关键消融", internal_rows))
    if not bucket_df.empty:
        markdown_parts.extend(
            [
                "## 不确定性分桶统计",
                "",
                render_bucket_markdown(bucket_df),
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(markdown_parts), encoding="utf-8")

    if external_rows:
        (output_dir / "external_baselines.tex").write_text(
            render_latex_table("第五章外部基线对比结果", "tab:ch5_external_baselines", external_rows),
            encoding="utf-8",
        )
        pd.DataFrame(external_rows).to_csv(
            output_dir / "external_baselines.csv", index=False, encoding="utf-8-sig"
        )

    if internal_rows:
        (output_dir / "internal_ablations.tex").write_text(
            render_latex_table("第五章关键消融实验结果", "tab:ch5_internal_ablation", internal_rows),
            encoding="utf-8",
        )
        pd.DataFrame(internal_rows).to_csv(
            output_dir / "internal_ablations.csv", index=False, encoding="utf-8-sig"
        )

    if not bucket_df.empty:
        (output_dir / "uncertainty_bucket_table.tex").write_text(
            render_uncertainty_latex(bucket_df), encoding="utf-8"
        )
        bucket_df.to_csv(output_dir / "uncertainty_bucket_table.csv", index=False, encoding="utf-8-sig")

    print(f"Saved Chapter 5 summaries to {output_dir}")


if __name__ == "__main__":
    main()
