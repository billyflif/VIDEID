from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.models.reid_model import VideoReIDModel
    from src.train_cowclips import (
        build_strict_eval_loaders,
        compute_reid_metrics,
        parse_strict_manifest,
        set_seed,
    )
except Exception:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.reid_model import VideoReIDModel
    from src.train_cowclips import (
        build_strict_eval_loaders,
        compute_reid_metrics,
        parse_strict_manifest,
        set_seed,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze uncertainty quality cues for Chapter 5")
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--strict-manifest", type=Path, required=True, help="strict_reid manifest")
    parser.add_argument("--device", type=str, default="cuda", help="inference device")
    parser.add_argument("--frames-per-clip", type=int, default=8, help="frames per clip for evaluation")
    parser.add_argument("--batch-size", type=int, default=4, help="evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--output-dir", type=Path, required=True, help="analysis output directory")
    parser.add_argument(
        "--consistency-tol",
        type=float,
        default=1e-6,
        help="tolerance used when comparing detailed metrics with compute_reid_metrics",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def sanitize_filename(name: str, limit: int = 80) -> str:
    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    clean = clean.strip("._")
    if not clean:
        clean = "sample"
    return clean[:limit]


def denormalize_video(video: torch.Tensor) -> np.ndarray:
    video = video.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    video = video.clamp(0.0, 1.0)
    video = (video.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    return video


def frame_quality_proxy(frame_rgb: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    return blur_score, brightness_mean


def scalarize_sigma2(sigma2: torch.Tensor) -> torch.Tensor:
    if sigma2.dim() == 2 and sigma2.size(-1) > 1:
        return sigma2.norm(dim=-1)
    if sigma2.dim() == 2 and sigma2.size(-1) == 1:
        return sigma2.squeeze(-1)
    raise ValueError(f"Unsupported sigma2 shape: {tuple(sigma2.shape)}")


def scalarize_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.dim() == 2 and weights.size(-1) == 1:
        return weights.squeeze(-1)
    if weights.dim() == 1:
        return weights
    raise ValueError(f"Unsupported weights shape: {tuple(weights.shape)}")


def save_frame_pair_visualization(
    output_path: Path,
    low_frame: np.ndarray,
    high_frame: np.ndarray,
    low_index: int,
    high_index: int,
    low_sigma: float,
    high_sigma: float,
    low_blur: float,
    high_blur: float,
    low_brightness: float,
    high_brightness: float,
    identity_name: str,
    relative_path: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_identity = sanitize_filename(identity_name)
    safe_stem = sanitize_filename(Path(relative_path).stem)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=180)
    axes[0].imshow(low_frame)
    axes[0].set_title(
        f"Low $\\sigma^2$ #{low_index}\n$\\sigma^2$={low_sigma:.3f}\nblur={low_blur:.1f}, bright={low_brightness:.1f}"
    )
    axes[0].axis("off")

    axes[1].imshow(high_frame)
    axes[1].set_title(
        f"High $\\sigma^2$ #{high_index}\n$\\sigma^2$={high_sigma:.3f}\nblur={high_blur:.1f}, bright={high_brightness:.1f}"
    )
    axes[1].axis("off")

    fig.suptitle(f"{safe_identity} | {safe_stem}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def extract_split_records(
    model: VideoReIDModel,
    loader,
    device: torch.device,
    split_name: str,
    visualization_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, object]], torch.Tensor, torch.Tensor]:
    model.eval()
    dataset = loader.dataset
    label_to_id = {label: id_name for id_name, label in dataset.id2label.items()}

    records: List[Dict[str, object]] = []
    all_feats: List[torch.Tensor] = []
    all_labels: List[int] = []
    sample_index = 0

    for videos, labels in loader:
        videos = videos.to(device)
        outputs = model(videos)
        sigma2 = outputs["sigma2"].detach().cpu()
        weights = outputs["weights"].detach().cpu()
        feats = outputs["vid_id"].detach().cpu()
        videos_cpu = videos.detach().cpu()

        batch_size = labels.size(0)
        for batch_offset in range(batch_size):
            video_path, expected_label = dataset.samples[sample_index]
            label = int(labels[batch_offset].item())
            if int(expected_label) != label:
                raise RuntimeError(
                    f"Label mismatch at index {sample_index}: dataset={expected_label}, batch={label}"
                )

            sigma_values = scalarize_sigma2(sigma2[batch_offset])
            weight_values = scalarize_weights(weights[batch_offset])
            if sigma_values.numel() != weight_values.numel():
                raise RuntimeError("sigma2 and weight lengths do not match")

            sigma_np = sigma_values.numpy()
            weight_np = weight_values.numpy()
            low_index = int(np.argmin(sigma_np))
            high_index = int(np.argmax(sigma_np))
            weight_entropy = float(-(weight_np * np.log(np.clip(weight_np, 1e-8, None))).sum())
            weight_max = float(weight_np.max())

            vis_path = None
            low_blur = high_blur = 0.0
            low_brightness = high_brightness = 0.0
            if visualization_dir is not None:
                frames = denormalize_video(videos_cpu[batch_offset])
                low_frame = frames[low_index]
                high_frame = frames[high_index]
                low_blur, low_brightness = frame_quality_proxy(low_frame)
                high_blur, high_brightness = frame_quality_proxy(high_frame)
                vis_name = (
                    f"{sample_index:04d}_{sanitize_filename(label_to_id[label])}_"
                    f"{sanitize_filename(video_path.stem)}.png"
                )
                vis_path = visualization_dir / vis_name
                save_frame_pair_visualization(
                    output_path=vis_path,
                    low_frame=low_frame,
                    high_frame=high_frame,
                    low_index=low_index,
                    high_index=high_index,
                    low_sigma=float(sigma_np[low_index]),
                    high_sigma=float(sigma_np[high_index]),
                    low_blur=low_blur,
                    high_blur=high_blur,
                    low_brightness=low_brightness,
                    high_brightness=high_brightness,
                    identity_name=label_to_id[label],
                    relative_path=str(video_path),
                )

            record = {
                "sample_index": sample_index,
                "split": split_name,
                "identity": label,
                "identity_name": label_to_id[label],
                "path": str(video_path),
                "relative_path": video_path.name,
                "sigma2_mean": float(sigma_np.mean()),
                "sigma2_std": float(sigma_np.std()),
                "sigma2_min": float(sigma_np.min()),
                "sigma2_max": float(sigma_np.max()),
                "weight_entropy": weight_entropy,
                "weight_max": weight_max,
                "low_sigma_frame_index": low_index,
                "high_sigma_frame_index": high_index,
                "low_sigma_value": float(sigma_np[low_index]),
                "high_sigma_value": float(sigma_np[high_index]),
                "low_sigma_blur_score": low_blur,
                "high_sigma_blur_score": high_blur,
                "low_sigma_brightness_mean": low_brightness,
                "high_sigma_brightness_mean": high_brightness,
                "visualization_path": (
                    str(vis_path.relative_to(REPO_ROOT)) if vis_path is not None else None
                ),
                "sigma2_per_frame": [float(v) for v in sigma_np.tolist()],
                "weight_per_frame": [float(v) for v in weight_np.tolist()],
            }
            records.append(record)
            all_feats.append(feats[batch_offset])
            all_labels.append(label)
            sample_index += 1

    feature_tensor = torch.stack(all_feats, dim=0) if all_feats else torch.empty(0, 512)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    return records, feature_tensor, label_tensor


def compute_query_details(
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    query_records: Sequence[Dict[str, object]],
    gallery_feats: torch.Tensor,
    gallery_labels: torch.Tensor,
    gallery_records: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    q = F.normalize(query_feats, dim=1)
    g = F.normalize(gallery_feats, dim=1)
    sim = torch.matmul(q, g.t())

    max_rank = min(10, g.size(0)) if g.numel() > 0 else 0
    cmc = torch.zeros(max_rank, dtype=torch.float32)
    all_ap: List[float] = []
    valid_queries = 0
    detailed_records: List[Dict[str, object]] = []

    for idx, base_record in enumerate(query_records):
        q_label = int(query_labels[idx].item())
        order = torch.argsort(sim[idx], descending=True)
        matches = (gallery_labels[order] == q_label).float()
        num_rel = int(matches.sum().item())

        record = dict(base_record)
        if num_rel == 0:
            record.update(
                {
                    "rank1_hit": False,
                    "ap": 0.0,
                    "first_correct_rank": None,
                    "top1_gallery_path": None,
                    "top1_gallery_identity": None,
                    "first_correct_gallery_path": None,
                    "top1_similarity": None,
                    "first_correct_similarity": None,
                }
            )
            detailed_records.append(record)
            continue

        valid_queries += 1
        if max_rank > 0:
            first_correct = int(torch.nonzero(matches, as_tuple=False).view(-1)[0].item())
            if first_correct < max_rank:
                cmc[first_correct:] += 1
        else:
            first_correct = 0

        ranks = torch.arange(1, matches.numel() + 1, dtype=torch.float32)
        cum_hits = torch.cumsum(matches, dim=0)
        precision = cum_hits / ranks
        ap = float(((precision * matches).sum() / max(num_rel, 1)).item())
        all_ap.append(ap)

        top1_index = int(order[0].item())
        first_correct_gallery_index = int(order[first_correct].item())
        record.update(
            {
                "rank1_hit": bool(matches[0].item() > 0),
                "ap": ap,
                "first_correct_rank": first_correct + 1,
                "top1_gallery_path": gallery_records[top1_index]["path"],
                "top1_gallery_identity": gallery_records[top1_index]["identity_name"],
                "first_correct_gallery_path": gallery_records[first_correct_gallery_index]["path"],
                "top1_similarity": float(sim[idx, top1_index].item()),
                "first_correct_similarity": float(sim[idx, first_correct_gallery_index].item()),
            }
        )
        detailed_records.append(record)

    metrics = {
        "rank-1": float(cmc[0].item() / valid_queries) if valid_queries > 0 and max_rank > 0 else 0.0,
        "mAP": float(np.mean(all_ap)) if all_ap else 0.0,
    }
    return detailed_records, metrics


def build_bucket_summary(query_df: pd.DataFrame) -> pd.DataFrame:
    if query_df.empty:
        return pd.DataFrame(
            columns=["bucket", "num_queries", "sigma2_mean", "rank1_rate", "mAP_mean"]
        )

    ranks = query_df["sigma2_mean"].rank(method="first")
    bucket_ids = pd.qcut(ranks, q=min(4, len(query_df)), labels=False, duplicates="drop")
    label_map = {
        0: "Q1 lowest sigma2",
        1: "Q2",
        2: "Q3",
        3: "Q4 highest sigma2",
    }
    query_df = query_df.copy()
    query_df["sigma_bucket_id"] = bucket_ids.astype(int)
    query_df["bucket"] = query_df["sigma_bucket_id"].map(label_map)

    summary = (
        query_df.groupby(["sigma_bucket_id", "bucket"], as_index=False)
        .agg(
            num_queries=("path", "count"),
            sigma2_mean=("sigma2_mean", "mean"),
            rank1_rate=("rank1_hit", "mean"),
            mAP_mean=("ap", "mean"),
        )
        .sort_values("sigma_bucket_id")
        .reset_index(drop=True)
    )
    return summary


def render_bucket_plot(bucket_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    x = np.arange(len(bucket_df))
    width = 0.36
    ax1.bar(x - width / 2, bucket_df["rank1_rate"] * 100.0, width=width, label="Rank-1")
    ax1.bar(x + width / 2, bucket_df["mAP_mean"] * 100.0, width=width, label="mAP")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_df["bucket"], rotation=0)
    ax1.set_ylim(0, 100)
    ax1.set_title("Sigma2 bucketed retrieval performance")
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(x, bucket_df["sigma2_mean"], color="black", marker="o", linewidth=1.5, label="mean sigma2")
    ax2.set_ylabel("Mean sigma2")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_sigma_scatter(query_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    colors = np.where(query_df["rank1_hit"], "#2c7c54", "#c44536")
    ax.scatter(query_df["sigma2_mean"], query_df["ap"], c=colors, alpha=0.8, edgecolors="none")
    ax.set_xlabel("Mean sigma2")
    ax.set_ylabel("AP")
    ax.set_title("Mean sigma2 vs AP")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_table_tex(bucket_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{基于查询片段平均不确定性的分桶统计结果}",
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
            "% 图注建议",
            "% 图~\\ref{fig:ch5_sigma_frame_pairs} 展示了同一查询片段中最低与最高 $\\sigma^2$ 帧的对比，",
            "% 可用于说明遮挡、模糊或背光帧通常对应更高的不确定性。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint format is invalid: {ckpt_path}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: Dict[str, object], device: torch.device) -> VideoReIDModel:
    args = checkpoint.get("args", {}) or {}
    model_config = checkpoint.get("model_config", {}) or {}

    model_arch = str(model_config.get("model_arch", args.get("model_arch", "dual_mamba")))
    feat_dim = int(args.get("feat_dim", 512))
    num_blocks = int(args.get("num_blocks", 4))
    model = VideoReIDModel(
        feat_dim=feat_dim,
        num_blocks=num_blocks,
        model_arch=model_arch,
        use_quality_gating=bool(model_config.get("use_quality_gating", True)),
        bidirectional=bool(model_config.get("bidirectional", True)),
        use_pose_stream=bool(model_config.get("use_pose_stream", model_arch == "dual_mamba")),
        use_pose_to_id=bool(model_config.get("use_pose_to_id", model_arch == "dual_mamba")),
        use_uncertainty_weighting=bool(model_config.get("use_uncertainty_weighting", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def summarize_folds(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not fold_metrics:
        return {}
    keys = sorted(fold_metrics[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([fold[key] for fold in fold_metrics], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_dir = (REPO_ROOT / args.ckpt_dir).resolve()
    manifest_path = (REPO_ROOT / args.strict_manifest).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Strict manifest not found: {manifest_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    fold_specs, data_root = parse_strict_manifest(manifest_path)
    fold_metrics: List[Dict[str, float]] = []
    query_tables: List[pd.DataFrame] = []
    gallery_tables: List[pd.DataFrame] = []

    for fold_spec in fold_specs:
        ckpt_path = ckpt_dir / f"{fold_spec.fold_name}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fold checkpoint not found: {ckpt_path}")

        checkpoint = load_checkpoint(ckpt_path, device)
        model = build_model_from_checkpoint(checkpoint, device)
        fold_output_dir = output_dir / fold_spec.fold_name
        vis_dir = fold_output_dir / "frame_pairs"

        gallery_loader, query_loader = build_strict_eval_loaders(
            fold_spec.test_gallery_entries,
            fold_spec.test_query_entries,
            data_root=data_root,
            frames_per_clip=args.frames_per_clip,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        gallery_records, gallery_feats, gallery_labels = extract_split_records(
            model=model,
            loader=gallery_loader,
            device=device,
            split_name="test_gallery",
            visualization_dir=None,
        )
        query_records, query_feats, query_labels = extract_split_records(
            model=model,
            loader=query_loader,
            device=device,
            split_name="test_query",
            visualization_dir=vis_dir,
        )

        detailed_queries, metrics = compute_query_details(
            query_feats=query_feats,
            query_labels=query_labels,
            query_records=query_records,
            gallery_feats=gallery_feats,
            gallery_labels=gallery_labels,
            gallery_records=gallery_records,
        )
        reference_metrics = compute_reid_metrics(
            query_feats=query_feats,
            query_labels=query_labels,
            gallery_feats=gallery_feats,
            gallery_labels=gallery_labels,
        )

        for key in ("rank-1", "mAP"):
            if not math.isclose(metrics[key], reference_metrics[key], rel_tol=0.0, abs_tol=args.consistency_tol):
                raise RuntimeError(
                    f"Metric mismatch on {fold_spec.fold_name} for {key}: "
                    f"detailed={metrics[key]:.8f}, reference={reference_metrics[key]:.8f}"
                )

        fold_metrics.append(
            {
                "test_rank-1": metrics["rank-1"],
                "test_mAP": metrics["mAP"],
            }
        )

        query_df = pd.DataFrame(detailed_queries)
        gallery_df = pd.DataFrame(gallery_records)
        bucket_df = build_bucket_summary(query_df)

        fold_output_dir.mkdir(parents=True, exist_ok=True)
        query_df.to_csv(fold_output_dir / "query_records.csv", index=False, encoding="utf-8-sig")
        gallery_df.to_csv(fold_output_dir / "gallery_records.csv", index=False, encoding="utf-8-sig")
        save_json(
            fold_output_dir / "query_records.json",
            query_df.to_dict(orient="records"),
        )
        save_json(
            fold_output_dir / "gallery_records.json",
            gallery_df.to_dict(orient="records"),
        )
        bucket_df.to_csv(fold_output_dir / "bucket_summary.csv", index=False, encoding="utf-8-sig")
        render_bucket_plot(bucket_df, fold_output_dir / "sigma_bucket_metrics.png")
        render_sigma_scatter(query_df, fold_output_dir / "sigma_vs_ap_scatter.png")
        write_table_tex(bucket_df, fold_output_dir / "uncertainty_bucket_table.tex")

        query_df.insert(0, "fold_name", fold_spec.fold_name)
        gallery_df.insert(0, "fold_name", fold_spec.fold_name)
        bucket_df.insert(0, "fold_name", fold_spec.fold_name)
        query_tables.append(query_df)
        gallery_tables.append(gallery_df)

    all_query_df = pd.concat(query_tables, ignore_index=True) if query_tables else pd.DataFrame()
    all_gallery_df = pd.concat(gallery_tables, ignore_index=True) if gallery_tables else pd.DataFrame()
    overall_bucket_df = build_bucket_summary(all_query_df)

    all_query_df.to_csv(output_dir / "query_records_all_folds.csv", index=False, encoding="utf-8-sig")
    all_gallery_df.to_csv(output_dir / "gallery_records_all_folds.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "query_records_all_folds.json", all_query_df.to_dict(orient="records"))
    save_json(output_dir / "gallery_records_all_folds.json", all_gallery_df.to_dict(orient="records"))
    overall_bucket_df.to_csv(output_dir / "bucket_summary.csv", index=False, encoding="utf-8-sig")
    render_bucket_plot(overall_bucket_df, output_dir / "sigma_bucket_metrics.png")
    render_sigma_scatter(all_query_df, output_dir / "sigma_vs_ap_scatter.png")
    write_table_tex(overall_bucket_df, output_dir / "uncertainty_bucket_table.tex")

    analysis_summary = {
        "ckpt_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
        "strict_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "num_folds": len(fold_metrics),
        "metrics_per_fold": fold_metrics,
        "summary": summarize_folds(fold_metrics),
        "num_query_samples": int(len(all_query_df)),
        "num_gallery_samples": int(len(all_gallery_df)),
        "bucket_summary_path": str((output_dir / "bucket_summary.csv").relative_to(REPO_ROOT)),
        "bucket_table_tex_path": str((output_dir / "uncertainty_bucket_table.tex").relative_to(REPO_ROOT)),
        "bucket_plot_path": str((output_dir / "sigma_bucket_metrics.png").relative_to(REPO_ROOT)),
        "scatter_plot_path": str((output_dir / "sigma_vs_ap_scatter.png").relative_to(REPO_ROOT)),
    }
    save_json(output_dir / "analysis_summary.json", analysis_summary)
    print(f"Saved uncertainty analysis to {output_dir}")


if __name__ == "__main__":
    main()
