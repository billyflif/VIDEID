from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

import cv2


def collect_raw_video_stats(raw_root: Path) -> Dict[str, object]:
    video_paths = sorted(raw_root.rglob("*.mp4"))
    id_counts: Dict[str, int] = {}
    durations: List[float] = []

    for path in video_paths:
        id_name = path.parent.name
        id_counts[id_name] = id_counts.get(id_name, 0) + 1

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps > 0.0 and frame_count > 0.0:
            durations.append(frame_count / fps)

    if not video_paths:
        raise RuntimeError(f"No .mp4 files found under {raw_root}")

    sorted_counts = sorted(id_counts.items(), key=lambda item: (-item[1], item[0]))
    stats = {
        "raw_root": str(raw_root),
        "raw_videos": len(video_paths),
        "num_ids": len(id_counts),
        "avg_videos_per_id": sum(id_counts.values()) / max(len(id_counts), 1),
        "videos_per_id_min": min(id_counts.values()),
        "videos_per_id_max": max(id_counts.values()),
        "avg_duration_sec": statistics.mean(durations) if durations else 0.0,
        "median_duration_sec": statistics.median(durations) if durations else 0.0,
        "duration_min_sec": min(durations) if durations else 0.0,
        "duration_max_sec": max(durations) if durations else 0.0,
        "top_ids_by_video_count": [
            {"id": id_name, "video_count": count}
            for id_name, count in sorted_counts[:5]
        ],
        "bottom_ids_by_video_count": [
            {"id": id_name, "video_count": count}
            for id_name, count in sorted(id_counts.items(), key=lambda item: (item[1], item[0]))[:5]
        ],
    }
    return stats


def collect_strict_split_stats(manifest_path: Path) -> Dict[str, object]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("protocol") != "strict_reid":
        raise RuntimeError(
            f"Expected protocol='strict_reid', got {manifest.get('protocol')!r}"
        )

    fold_stats = []
    for fold in manifest["folds"]:
        fold_stats.append(
            {
                "fold_name": fold["fold_name"],
                "train_ids": len(fold["train_ids"]),
                "val_ids": len(fold["val_ids"]),
                "test_ids": len(fold["test_ids"]),
                "train_videos": len(fold["train_videos"]),
                "val_gallery": len(fold["val_gallery"]),
                "val_query": len(fold["val_query"]),
                "test_gallery": len(fold["test_gallery"]),
                "test_query": len(fold["test_query"]),
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "num_folds": manifest["num_folds"],
        "eval_ids": manifest["eval_ids"],
        "excluded_ids": manifest["excluded_ids"],
        "train_only_ids": manifest.get("train_only_ids", []),
        "min_videos_per_eval_id": manifest.get("min_videos_per_eval_id"),
        "singleton_policy": manifest.get("singleton_policy"),
        "id_video_counts": manifest.get("id_video_counts", {}),
        "folds": fold_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Chapter 5 dataset statistics")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("兽医院-标注"),
        help="Root directory of raw annotated videos",
    )
    parser.add_argument(
        "--strict-manifest",
        type=Path,
        default=Path("dataset_ch5_strict/strict_reid_splits.json"),
        help="strict_reid manifest path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ch5_dataset_stats.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "raw_dataset": collect_raw_video_stats(args.raw_root),
        "strict_protocol": collect_strict_split_stats(args.strict_manifest),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    raw_stats = payload["raw_dataset"]
    strict_stats = payload["strict_protocol"]
    print(f"Saved dataset stats -> {args.output}")
    print(
        "Raw dataset: "
        f"{raw_stats['raw_videos']} videos, {raw_stats['num_ids']} IDs, "
        f"avg {raw_stats['avg_videos_per_id']:.3f} videos/ID, "
        f"avg duration {raw_stats['avg_duration_sec']:.3f}s"
    )
    print(
        "Strict protocol: "
        f"{len(strict_stats['eval_ids'])} eval IDs, "
        f"{len(strict_stats['excluded_ids'])} excluded singleton IDs, "
        f"{strict_stats['num_folds']} folds"
    )
    for fold in strict_stats["folds"]:
        print(
            f"  {fold['fold_name']}: train={fold['train_videos']} "
            f"val(g/q)={fold['val_gallery']}/{fold['val_query']} "
            f"test(g/q)={fold['test_gallery']}/{fold['test_query']}"
        )


if __name__ == "__main__":
    main()
