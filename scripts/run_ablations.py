"""
消融实验自动运行脚本 (TODO-13)

自动运行以下消融组合并汇总到一张表：
1. 仅聚合加权 vs 仅状态门控 vs 两者同时
2. 单向 Mamba vs 双向 Mamba
3. 无 pose stream vs 有 pose stream
4. 无 MI loss vs 无 orth loss vs 两者都有
5. num_blocks=2 vs 4 vs 6
6. baseline: AvgPool 替代不确定性加权聚合
7. baseline: 无辅助姿态任务 vs 有辅助姿态任务

用法：
    python scripts/run_ablations.py --strict-manifest dataset_ch5_strict/strict_reid_splits.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AblationConfig:
    name: str
    extra_args: List[str]


def build_ablation_configs() -> List[AblationConfig]:
    """构建所有消融实验配置。"""
    configs = []

    # ==== Full model (baseline) ====
    configs.append(AblationConfig(
        name="full_model",
        extra_args=[],
    ))

    # ==== 1. 聚合加权 vs 状态门控 ====
    configs.append(AblationConfig(
        name="no_quality_gate",
        extra_args=["--disable-quality-gate"],
    ))
    configs.append(AblationConfig(
        name="no_uncertainty_weighting",
        extra_args=["--disable-uncertainty-weighting"],
    ))
    configs.append(AblationConfig(
        name="no_gate_no_weight",
        extra_args=["--disable-quality-gate", "--disable-uncertainty-weighting"],
    ))

    # ==== 2. 单向 vs 双向 Mamba ====
    configs.append(AblationConfig(
        name="unidirectional",
        extra_args=["--disable-bidirectional"],
    ))

    # ==== 3. 无 pose stream vs 有 ====
    configs.append(AblationConfig(
        name="no_pose_stream",
        extra_args=["--disable-pose-stream"],
    ))

    # ==== 4. MI/orth loss 消融 ====
    configs.append(AblationConfig(
        name="no_mi_loss",
        extra_args=["--disable-mi-loss"],
    ))
    configs.append(AblationConfig(
        name="no_orth_loss",
        extra_args=["--disable-orth-loss"],
    ))
    configs.append(AblationConfig(
        name="no_mi_no_orth",
        extra_args=["--disable-mi-loss", "--disable-orth-loss"],
    ))

    # ==== 5. num_blocks 消融 ====
    configs.append(AblationConfig(
        name="blocks_2",
        extra_args=["--num-blocks", "2"],
    ))
    configs.append(AblationConfig(
        name="blocks_6",
        extra_args=["--num-blocks", "6"],
    ))

    # ==== 6. AvgPool baseline (禁用不确定性加权 + 禁用质量门控) ====
    configs.append(AblationConfig(
        name="avgpool_baseline",
        extra_args=[
            "--disable-uncertainty-weighting",
            "--disable-quality-gate",
            "--disable-kl-loss",
        ],
    ))

    # ==== 7. 无辅助姿态任务 ====
    configs.append(AblationConfig(
        name="no_pose_aux",
        extra_args=["--disable-pose-aux"],
    ))

    return configs


def run_single_ablation(
    config: AblationConfig,
    base_args: List[str],
    output_dir: Path,
) -> Optional[Dict]:
    """运行单个消融实验。"""
    ckpt_dir = output_dir / config.name
    log_dir = output_dir / "logs" / config.name

    cmd = [
        sys.executable, "-m", "src.train_cowclips",
        "--ckpt-dir", str(ckpt_dir),
        "--log-dir", str(log_dir),
        "--run-all-folds",
    ] + base_args + config.extra_args

    print(f"\n{'=' * 60}")
    print(f"Running ablation: {config.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600 * 4,
        )
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] {config.name} failed with return code {result.returncode}")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {config.name} timed out")
        return None

    # 读取结果
    summary_path = ckpt_dir / "cross_fold_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {"name": config.name, "summary": summary.get("summary", {})}
    else:
        print(f"[WARN] No summary found for {config.name}")
        return None


def generate_latex_table(results: List[Dict]) -> str:
    """生成消融结果 LaTeX 表格。"""
    if not results:
        return "% No results to display"

    # 收集所有指标
    all_keys = set()
    for r in results:
        all_keys.update(r.get("summary", {}).keys())

    display_keys = []
    for key in ["test_mAP", "test_rank-1", "test_rank-5", "best_val_mAP"]:
        if key in all_keys:
            display_keys.append(key)

    if not display_keys:
        display_keys = sorted(all_keys)[:4]

    # 生成表头
    header = "Method & " + " & ".join(
        k.replace("_", r"\_") for k in display_keys
    ) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{消融实验结果}",
        r"  \label{tab:ablation}",
        f"  \\begin{{tabular}}{{l{'c' * len(display_keys)}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]

    for r in results:
        name = r["name"].replace("_", r"\_")
        summary = r.get("summary", {})
        vals = []
        for key in display_keys:
            if key in summary:
                mean = summary[key].get("mean", 0.0)
                std = summary[key].get("std", 0.0)
                vals.append(f"{mean:.4f}$\\pm${std:.4f}")
            else:
                vals.append("-")
        lines.append(f"    {name} & {' & '.join(vals)} " + r"\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--strict-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="ablation_results")
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print commands without running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--strict-manifest", args.strict_manifest,
        "--protocol", "strict_reid",
        "--num-epochs", str(args.num_epochs),
        "--device", args.device,
    ]

    configs = build_ablation_configs()
    print(f"Total ablation configs: {len(configs)}")
    for c in configs:
        print(f"  - {c.name}: {c.extra_args}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    results: List[Dict] = []
    for config in configs:
        result = run_single_ablation(config, base_args, output_dir)
        if result is not None:
            results.append(result)

    # 保存汇总结果
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved ablation results -> {results_path}")

    # 生成 LaTeX 表格
    latex = generate_latex_table(results)
    latex_path = output_dir / "ablation_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX table -> {latex_path}")

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("Ablation Summary")
    print(f"{'=' * 60}")
    for r in results:
        name = r["name"]
        summary = r.get("summary", {})
        mAP = summary.get("test_mAP", {}).get("mean", -1)
        rank1 = summary.get("test_rank-1", {}).get("mean", -1)
        print(f"  {name:30s}  mAP={mAP:.4f}  rank-1={rank1:.4f}")


if __name__ == "__main__":
    main()
