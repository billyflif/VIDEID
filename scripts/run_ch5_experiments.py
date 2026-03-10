from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_MANIFEST = Path("dataset_ch5_strict/strict_reid_splits.json")


@dataclass
class CommandSpec:
    name: str
    kind: str
    command: List[str]
    log_dir: Optional[str] = None
    ckpt_dir: Optional[str] = None
    output_dir: Optional[str] = None
    summary_required: bool = True


def build_train_base_args(manifest_path: Path, device: str) -> List[str]:
    return [
        "-m",
        "src.train_cowclips",
        "--strict-manifest",
        str(manifest_path),
        "--protocol",
        "strict_reid",
        "--device",
        device,
        "--patience",
        "15",
        "--frames-per-clip",
        "8",
        "--batch-size",
        "4",
        "--feat-dim",
        "512",
        "--num-blocks",
        "4",
        "--lr",
        "3e-4",
        "--mine-lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--margin",
        "0.3",
        "--lambda-mi",
        "0.1",
        "--lambda-orth",
        "0.01",
        "--lambda-temp",
        "0.1",
        "--lambda-kl",
        "0.01",
        "--lambda-pose-aux",
        "0.2",
        "--num-workers",
        "0",
    ]


def build_train_command(
    manifest_path: Path,
    device: str,
    log_dir: str,
    ckpt_dir: str,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    extra_args = extra_args or []
    return [
        *build_train_base_args(manifest_path, device),
        "--log-dir",
        log_dir,
        "--ckpt-dir",
        ckpt_dir,
        *extra_args,
    ]


def build_specs(version: str, manifest_path: Path, device: str) -> Dict[str, CommandSpec]:
    main_log = f"runs/ch5_main_strict_poseaux_{version}"
    main_ckpt = f"checkpoints/ch5_main_strict_poseaux_{version}"

    specs = {
        "ch5_main_strict_poseaux": CommandSpec(
            name="ch5_main_strict_poseaux",
            kind="train",
            log_dir=main_log,
            ckpt_dir=main_ckpt,
            command=build_train_command(
                manifest_path,
                device,
                log_dir=main_log,
                ckpt_dir=main_ckpt,
                extra_args=["--run-all-folds", "--num-epochs", "80", "--model-arch", "dual_mamba"],
            ),
        ),
        "ch5_baseline_avgpool_pure": CommandSpec(
            name="ch5_baseline_avgpool_pure",
            kind="train",
            log_dir=f"runs/ch5_baseline_avgpool_pure_{version}",
            ckpt_dir=f"checkpoints/ch5_baseline_avgpool_pure_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_baseline_avgpool_pure_{version}",
                ckpt_dir=f"checkpoints/ch5_baseline_avgpool_pure_{version}",
                extra_args=["--run-all-folds", "--num-epochs", "80", "--model-arch", "avgpool"],
            ),
        ),
        "ch5_baseline_bigru": CommandSpec(
            name="ch5_baseline_bigru",
            kind="train",
            log_dir=f"runs/ch5_baseline_bigru_{version}",
            ckpt_dir=f"checkpoints/ch5_baseline_bigru_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_baseline_bigru_{version}",
                ckpt_dir=f"checkpoints/ch5_baseline_bigru_{version}",
                extra_args=["--run-all-folds", "--num-epochs", "80", "--model-arch", "gru"],
            ),
        ),
        "ch5_baseline_bilstm": CommandSpec(
            name="ch5_baseline_bilstm",
            kind="train",
            log_dir=f"runs/ch5_baseline_bilstm_{version}",
            ckpt_dir=f"checkpoints/ch5_baseline_bilstm_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_baseline_bilstm_{version}",
                ckpt_dir=f"checkpoints/ch5_baseline_bilstm_{version}",
                extra_args=["--run-all-folds", "--num-epochs", "80", "--model-arch", "lstm"],
            ),
        ),
        "ch5_baseline_temporal_transformer": CommandSpec(
            name="ch5_baseline_temporal_transformer",
            kind="train",
            log_dir=f"runs/ch5_baseline_temporal_transformer_{version}",
            ckpt_dir=f"checkpoints/ch5_baseline_temporal_transformer_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_baseline_temporal_transformer_{version}",
                ckpt_dir=f"checkpoints/ch5_baseline_temporal_transformer_{version}",
                extra_args=["--run-all-folds", "--num-epochs", "80", "--model-arch", "transformer"],
            ),
        ),
        "ch5_baseline_single_mamba": CommandSpec(
            name="ch5_baseline_single_mamba",
            kind="train",
            log_dir=f"runs/ch5_baseline_single_mamba_{version}",
            ckpt_dir=f"checkpoints/ch5_baseline_single_mamba_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_baseline_single_mamba_{version}",
                ckpt_dir=f"checkpoints/ch5_baseline_single_mamba_{version}",
                extra_args=["--run-all-folds", "--num-epochs", "80", "--model-arch", "single_mamba"],
            ),
        ),
        "ch5_ablation_no_quality_uncertainty": CommandSpec(
            name="ch5_ablation_no_quality_uncertainty",
            kind="train",
            log_dir=f"runs/ch5_ablation_no_quality_uncertainty_{version}",
            ckpt_dir=f"checkpoints/ch5_ablation_no_quality_uncertainty_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_ablation_no_quality_uncertainty_{version}",
                ckpt_dir=f"checkpoints/ch5_ablation_no_quality_uncertainty_{version}",
                extra_args=[
                    "--run-all-folds",
                    "--num-epochs",
                    "80",
                    "--model-arch",
                    "dual_mamba",
                    "--disable-uncertainty-weighting",
                    "--disable-quality-gate",
                    "--disable-kl-loss",
                ],
            ),
        ),
        "ch5_ablation_unidirectional": CommandSpec(
            name="ch5_ablation_unidirectional",
            kind="train",
            log_dir=f"runs/ch5_ablation_unidirectional_{version}",
            ckpt_dir=f"checkpoints/ch5_ablation_unidirectional_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_ablation_unidirectional_{version}",
                ckpt_dir=f"checkpoints/ch5_ablation_unidirectional_{version}",
                extra_args=[
                    "--run-all-folds",
                    "--num-epochs",
                    "80",
                    "--model-arch",
                    "dual_mamba",
                    "--disable-bidirectional",
                ],
            ),
        ),
        "ch5_ablation_no_pose_stream": CommandSpec(
            name="ch5_ablation_no_pose_stream",
            kind="train",
            log_dir=f"runs/ch5_ablation_no_pose_stream_{version}",
            ckpt_dir=f"checkpoints/ch5_ablation_no_pose_stream_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_ablation_no_pose_stream_{version}",
                ckpt_dir=f"checkpoints/ch5_ablation_no_pose_stream_{version}",
                extra_args=[
                    "--run-all-folds",
                    "--num-epochs",
                    "80",
                    "--model-arch",
                    "dual_mamba",
                    "--disable-pose-stream",
                ],
            ),
        ),
        "ch5_ablation_no_mi_loss": CommandSpec(
            name="ch5_ablation_no_mi_loss",
            kind="train",
            log_dir=f"runs/ch5_ablation_no_mi_loss_{version}",
            ckpt_dir=f"checkpoints/ch5_ablation_no_mi_loss_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_ablation_no_mi_loss_{version}",
                ckpt_dir=f"checkpoints/ch5_ablation_no_mi_loss_{version}",
                extra_args=[
                    "--run-all-folds",
                    "--num-epochs",
                    "80",
                    "--model-arch",
                    "dual_mamba",
                    "--disable-mi-loss",
                ],
            ),
        ),
        "ch5_ablation_no_pose_aux": CommandSpec(
            name="ch5_ablation_no_pose_aux",
            kind="train",
            log_dir=f"runs/ch5_ablation_no_pose_aux_{version}",
            ckpt_dir=f"checkpoints/ch5_ablation_no_pose_aux_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/ch5_ablation_no_pose_aux_{version}",
                ckpt_dir=f"checkpoints/ch5_ablation_no_pose_aux_{version}",
                extra_args=[
                    "--run-all-folds",
                    "--num-epochs",
                    "80",
                    "--model-arch",
                    "dual_mamba",
                    "--disable-pose-aux",
                ],
            ),
        ),
        "ch5_uncertainty_analysis": CommandSpec(
            name="ch5_uncertainty_analysis",
            kind="analysis",
            output_dir=f"runs/ch5_uncertainty_{version}",
            command=[
                "scripts/analyze_ch5_uncertainty.py",
                "--ckpt-dir",
                main_ckpt,
                "--strict-manifest",
                str(manifest_path),
                "--device",
                device,
                "--frames-per-clip",
                "8",
                "--output-dir",
                f"runs/ch5_uncertainty_{version}",
            ],
        ),
    }

    for arch in ["dual_mamba", "avgpool", "gru", "lstm", "transformer", "single_mamba"]:
        smoke_name = f"ch5_smoke_{arch}"
        specs[smoke_name] = CommandSpec(
            name=smoke_name,
            kind="train",
            log_dir=f"runs/{smoke_name}_{version}",
            ckpt_dir=f"checkpoints/{smoke_name}_{version}",
            command=build_train_command(
                manifest_path,
                device,
                log_dir=f"runs/{smoke_name}_{version}",
                ckpt_dir=f"checkpoints/{smoke_name}_{version}",
                extra_args=[
                    "--fold-index",
                    "1",
                    "--num-epochs",
                    "2",
                    "--model-arch",
                    arch,
                    "--pose-aux-warmup-steps",
                    "0",
                    "--pose-aux-ramp-steps",
                    "1",
                    "--mi-warmup-steps",
                    "0",
                    "--mi-ramp-steps",
                    "1",
                    "--kl-warmup-steps",
                    "0",
                    "--kl-ramp-steps",
                    "1",
                ],
            ),
        )

    return specs


def load_train_summary(ckpt_dir: Path) -> Optional[Dict[str, object]]:
    summary_path = ckpt_dir / "cross_fold_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_analysis_summary(output_dir: Path) -> Optional[Dict[str, object]]:
    summary_path = output_dir / "analysis_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(
    python_exe: Path,
    spec: CommandSpec,
    record: Dict[str, object],
    dry_run: bool,
) -> None:
    if spec.kind == "train":
        command = [str(python_exe), *spec.command]
    elif spec.kind == "analysis":
        command = [str(python_exe), *spec.command]
    else:
        raise ValueError(f"Unsupported spec.kind={spec.kind}")

    entry = {
        "kind": spec.kind,
        "command": command,
        "log_dir": spec.log_dir,
        "ckpt_dir": spec.ckpt_dir,
        "output_dir": spec.output_dir,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "planned" if dry_run else "running",
    }
    record["experiments"][spec.name] = entry

    print(f"\n{'=' * 88}")
    print(f"[{spec.name}] {' '.join(command)}")
    print(f"{'=' * 88}")

    if dry_run:
        return

    start = time.time()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    duration = time.time() - start
    entry["duration_sec"] = round(duration, 2)
    entry["returncode"] = completed.returncode

    if completed.returncode != 0:
        entry["status"] = "failed"
        raise RuntimeError(f"{spec.name} failed with return code {completed.returncode}")

    if spec.kind == "train" and spec.ckpt_dir is not None:
        summary = load_train_summary(REPO_ROOT / spec.ckpt_dir)
        if summary is not None:
            entry["summary"] = summary.get("summary", {})
            entry["metrics_per_fold"] = summary.get("metrics_per_fold", [])
        elif spec.summary_required:
            entry["status"] = "failed_no_summary"
            raise RuntimeError(f"{spec.name} finished without cross_fold_summary.json")
    elif spec.kind == "analysis" and spec.output_dir is not None:
        summary = load_analysis_summary(REPO_ROOT / spec.output_dir)
        if summary is not None:
            entry["summary"] = summary
        elif spec.summary_required:
            entry["status"] = "failed_no_summary"
            raise RuntimeError(f"{spec.name} finished without analysis_summary.json")

    entry["status"] = "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 5 experiment suite")
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python executable used to launch scripts",
    )
    parser.add_argument(
        "--strict-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="strict_reid manifest path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to training/analysis scripts",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "smoke",
            "main",
            "external_baselines",
            "internal_ablations",
            "uncertainty_analysis",
            "all",
        ],
        default="all",
        help="Experiment stage to run",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="Suffix used in log/checkpoint directory names",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Optional JSON record path",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated experiment names to run within the selected stage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the planned record without executing them",
    )
    return parser.parse_args()


def write_record(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    manifest_path = (REPO_ROOT / args.strict_manifest).resolve()
    python_exe = args.python_exe.resolve()

    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Strict manifest not found: {manifest_path}")

    specs = build_specs(args.version, manifest_path, args.device)
    stage_map = {
        "smoke": [
            "ch5_smoke_dual_mamba",
            "ch5_smoke_avgpool",
            "ch5_smoke_gru",
            "ch5_smoke_lstm",
            "ch5_smoke_transformer",
            "ch5_smoke_single_mamba",
        ],
        "main": ["ch5_main_strict_poseaux"],
        "external_baselines": [
            "ch5_baseline_avgpool_pure",
            "ch5_baseline_bigru",
            "ch5_baseline_bilstm",
            "ch5_baseline_temporal_transformer",
            "ch5_baseline_single_mamba",
        ],
        "internal_ablations": [
            "ch5_ablation_no_quality_uncertainty",
            "ch5_ablation_unidirectional",
            "ch5_ablation_no_pose_stream",
            "ch5_ablation_no_mi_loss",
            "ch5_ablation_no_pose_aux",
        ],
        "uncertainty_analysis": ["ch5_uncertainty_analysis"],
        "all": [
            "ch5_main_strict_poseaux",
            "ch5_baseline_avgpool_pure",
            "ch5_baseline_bigru",
            "ch5_baseline_bilstm",
            "ch5_baseline_temporal_transformer",
            "ch5_baseline_single_mamba",
            "ch5_ablation_no_quality_uncertainty",
            "ch5_ablation_unidirectional",
            "ch5_ablation_no_pose_stream",
            "ch5_ablation_no_mi_loss",
            "ch5_ablation_no_pose_aux",
            "ch5_uncertainty_analysis",
        ],
    }
    selected = list(stage_map[args.stage])

    if args.only:
        requested = [name.strip() for name in args.only.split(",") if name.strip()]
        invalid = [name for name in requested if name not in specs]
        if invalid:
            raise ValueError(f"Unknown experiment names in --only: {invalid}")
        selected = [name for name in selected if name in requested]
        if not selected:
            raise ValueError(
                f"--only={requested} does not match any experiment in stage={args.stage}"
            )

    record_path = args.record_path or Path(f"runs/ch5_experiment_record_{args.version}.json")
    record = {
        "python_exe": str(python_exe),
        "strict_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "device": args.device,
        "stage": args.stage,
        "version": args.version,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {},
    }

    print(f"Repository root: {REPO_ROOT}")
    print(f"Stage: {args.stage}")
    print(f"Experiments: {selected}")
    write_record(REPO_ROOT / record_path, record)

    for name in selected:
        run_one(python_exe, specs[name], record, dry_run=args.dry_run)
        write_record(REPO_ROOT / record_path, record)

    print(f"Record saved to {REPO_ROOT / record_path}")


if __name__ == "__main__":
    main()

