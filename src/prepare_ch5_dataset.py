from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from .video_clip import split_video_into_two_clips
except Exception:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from src.video_clip import split_video_into_two_clips


@dataclass
class FoldSpec:
    fold_index: int
    fold_name: str
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


def iter_raw_videos(raw_root: Path) -> Iterable[Tuple[str, str, Path]]:
    for hospital_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        for id_dir in sorted([p for p in hospital_dir.iterdir() if p.is_dir()]):
            for video_path in sorted(id_dir.glob("*.mp4")):
                if video_path.stem.endswith("_clip1") or video_path.stem.endswith("_clip2"):
                    continue
                yield hospital_dir.name, id_dir.name, video_path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_mode: str, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    ensure_dir(dst.parent)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_flat_ids(raw_root: Path, flat_root: Path, copy_mode: str, overwrite: bool) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for hospital_name, id_name, src in iter_raw_videos(raw_root):
        id_dir = flat_root / id_name
        ensure_dir(id_dir)
        prefixed_name = f"{hospital_name}__{src.name}"
        dst = id_dir / prefixed_name
        link_or_copy(src, dst, copy_mode=copy_mode, overwrite=overwrite)
        stats[id_name] = stats.get(id_name, 0) + 1
    return stats


def split_all_flat_videos(flat_root: Path, overwrite: bool) -> Tuple[int, int, int]:
    raw = 0
    clip1 = 0
    clip2 = 0
    for id_dir in sorted([p for p in flat_root.iterdir() if p.is_dir()]):
        for video_path in sorted(id_dir.glob("*.mp4")):
            if video_path.stem.endswith("_clip1") or video_path.stem.endswith("_clip2"):
                continue
            raw += 1
            split_video_into_two_clips(video_path, id_dir, overwrite=overwrite)
        clip1 += len(list(id_dir.glob("*_clip1.mp4")))
        clip2 += len(list(id_dir.glob("*_clip2.mp4")))
    return raw, clip1, clip2


def split_into_folds(ids: Sequence[str], num_folds: int) -> List[List[str]]:
    fold_size = len(ids) // num_folds
    remainder = len(ids) % num_folds
    folds: List[List[str]] = []
    start = 0
    for idx in range(num_folds):
        end = start + fold_size + (1 if idx < remainder else 0)
        folds.append(list(ids[start:end]))
        start = end
    return folds


def build_tracklet_fold_specs(ids: Sequence[str], num_folds: int, val_ratio: float, seed: int) -> List[FoldSpec]:
    if num_folds > len(ids):
        raise RuntimeError(f"num_folds={num_folds} > num_ids={len(ids)}")
    if not (0.0 < val_ratio < 0.8):
        raise RuntimeError(f"val_ratio must be in (0, 0.8), got {val_ratio}")

    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    test_folds = split_into_folds(shuffled, num_folds)

    specs: List[FoldSpec] = []
    for fold_idx, test_ids in enumerate(test_folds, start=1):
        remain = [x for x in shuffled if x not in test_ids]
        rng_fold = random.Random(seed + 10000 + fold_idx)
        rng_fold.shuffle(remain)

        val_n = max(1, int(round(len(remain) * val_ratio)))
        val_n = min(val_n, len(remain) - 1)
        val_ids = sorted(remain[:val_n])
        train_ids = sorted(remain[val_n:])
        if not train_ids or not val_ids or not test_ids:
            raise RuntimeError(
                f"Invalid split in fold {fold_idx}: "
                f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
            )

        specs.append(
            FoldSpec(
                fold_index=fold_idx,
                fold_name=f"fold{fold_idx:02d}",
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=sorted(list(test_ids)),
            )
        )
    return specs


def list_tracklet_clip_files(flat_root: Path, id_name: str) -> List[Path]:
    id_dir = flat_root / id_name
    return sorted(id_dir.glob("*_clip1.mp4")) + sorted(id_dir.glob("*_clip2.mp4"))


def clear_generated_fold_dir(fold_dir: Path) -> None:
    for name in ("train", "val", "test", "split.json"):
        path = fold_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def prune_stale_fold_dirs(folds_root: Path, valid_fold_names: Sequence[str]) -> None:
    valid = set(valid_fold_names)
    for child in folds_root.iterdir():
        if child.is_dir() and child.name.startswith("fold") and child.name not in valid:
            shutil.rmtree(child)


def materialize_tracklet_folds(
    flat_root: Path,
    folds_root: Path,
    fold_specs: Sequence[FoldSpec],
    copy_mode: str,
    overwrite: bool,
    seed: int,
    clean_folds: bool,
) -> None:
    ensure_dir(folds_root)
    expected_fold_names = [spec.fold_name for spec in fold_specs]
    if clean_folds:
        prune_stale_fold_dirs(folds_root, expected_fold_names)

    all_manifest = []
    for spec in fold_specs:
        fold_dir = folds_root / spec.fold_name
        ensure_dir(fold_dir)
        if clean_folds:
            clear_generated_fold_dir(fold_dir)
        for split_name, ids in (("train", spec.train_ids), ("val", spec.val_ids), ("test", spec.test_ids)):
            split_root = fold_dir / split_name
            ensure_dir(split_root)
            for id_name in ids:
                dst_id_dir = split_root / id_name
                ensure_dir(dst_id_dir)
                for clip_path in list_tracklet_clip_files(flat_root, id_name):
                    link_or_copy(clip_path, dst_id_dir / clip_path.name, copy_mode=copy_mode, overwrite=overwrite)

        manifest = {
            "fold_index": spec.fold_index,
            "fold_name": spec.fold_name,
            "train_ids": spec.train_ids,
            "val_ids": spec.val_ids,
            "test_ids": spec.test_ids,
            "seed": seed,
        }
        with open(fold_dir / "split.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        all_manifest.append(manifest)

    with open(folds_root / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "tracklet_halves", "folds": all_manifest}, f, ensure_ascii=False, indent=2)


def list_raw_videos(flat_root: Path, id_name: str) -> List[Path]:
    id_dir = flat_root / id_name
    return sorted(
        p
        for p in id_dir.glob("*.mp4")
        if not p.stem.endswith("_clip1") and not p.stem.endswith("_clip2")
    )


def build_id_video_map(flat_root: Path) -> Dict[str, List[Path]]:
    return {
        id_dir.name: list_raw_videos(flat_root, id_dir.name)
        for id_dir in sorted([p for p in flat_root.iterdir() if p.is_dir()])
    }


def build_balanced_outer_folds(id_to_count: Dict[str, int], num_folds: int) -> List[List[str]]:
    if num_folds <= 1:
        return [sorted(id_to_count.keys())]
    if num_folds > len(id_to_count):
        raise RuntimeError(f"num_folds={num_folds} > num_ids={len(id_to_count)}")

    ordered_ids = sorted(id_to_count.items(), key=lambda item: (-item[1], item[0]))
    folds: List[List[str]] = [[] for _ in range(num_folds)]
    fold_sums = [0 for _ in range(num_folds)]
    for id_name, count in ordered_ids:
        target_idx = min(range(num_folds), key=lambda idx: (fold_sums[idx], len(folds[idx]), idx))
        folds[target_idx].append(id_name)
        fold_sums[target_idx] += count
    return [sorted(fold) for fold in folds]


def choose_val_ids(candidate_ids: Sequence[str], id_to_count: Dict[str, int], val_ratio: float) -> List[str]:
    candidate_ids = sorted(candidate_ids)
    if len(candidate_ids) < 2:
        raise RuntimeError("At least two candidate IDs are required to build a validation split.")
    target_count = sum(id_to_count[id_name] for id_name in candidate_ids) * val_ratio
    max_r = min(4, len(candidate_ids) - 1)
    min_r = 2 if len(candidate_ids) >= 3 else 1

    best_score = None
    best_combo = None
    for r in range(min_r, max_r + 1):
        for combo in combinations(candidate_ids, r):
            combo_sum = sum(id_to_count[id_name] for id_name in combo)
            score = (abs(combo_sum - target_count), r, -combo_sum, combo)
            if best_score is None or score < best_score:
                best_score = score
                best_combo = combo
        if best_score is not None and best_score[0] <= 1:
            break
    if best_combo is None:
        raise RuntimeError("Failed to choose validation IDs.")
    return sorted(list(best_combo))


def split_gallery_query(video_paths: Sequence[Path]) -> Tuple[List[Path], List[Path]]:
    ordered = sorted(video_paths)
    if len(ordered) < 2:
        raise RuntimeError("Strict ReID evaluation requires at least two raw videos per ID.")
    gallery = ordered[::2]
    query = ordered[1::2]
    if not query:
        query = [gallery.pop()]
    if not gallery or not query:
        raise RuntimeError("Failed to build non-empty gallery/query split.")
    return gallery, query


def entry_from_path(output_root: Path, path: Path, id_name: str) -> Dict[str, str]:
    return {"id": id_name, "path": path.relative_to(output_root).as_posix()}


def build_strict_reid_manifest(
    raw_root: Path,
    flat_root: Path,
    output_root: Path,
    num_folds: int,
    val_ratio: float,
    seed: int,
    min_videos_per_eval_id: int,
    singleton_policy: str,
    balanced_by: str,
) -> Dict[str, object]:
    if balanced_by != "video_count":
        raise RuntimeError(f"Unsupported balanced-by={balanced_by}")

    id_to_videos = build_id_video_map(flat_root)
    id_to_count = {id_name: len(videos) for id_name, videos in id_to_videos.items()}
    eval_ids = sorted([id_name for id_name, count in id_to_count.items() if count >= min_videos_per_eval_id])
    excluded_ids = sorted([id_name for id_name, count in id_to_count.items() if count < min_videos_per_eval_id])
    if len(eval_ids) < num_folds:
        raise RuntimeError(f"Not enough IDs for strict ReID folds: {len(eval_ids)} < {num_folds}")

    train_only_ids = excluded_ids if singleton_policy == "train_only" else []
    dropped_ids = excluded_ids if singleton_policy == "drop" else []

    outer_folds = build_balanced_outer_folds({id_name: id_to_count[id_name] for id_name in eval_ids}, num_folds)

    fold_payloads: List[Dict[str, object]] = []
    for fold_index, test_ids in enumerate(outer_folds, start=1):
        remain_ids = sorted([id_name for id_name in eval_ids if id_name not in test_ids])
        val_ids = choose_val_ids(remain_ids, id_to_count, val_ratio)
        train_ids = sorted([id_name for id_name in remain_ids if id_name not in val_ids] + train_only_ids)

        train_videos = [
            entry_from_path(output_root, path, id_name)
            for id_name in train_ids
            for path in id_to_videos[id_name]
        ]

        val_gallery: List[Dict[str, str]] = []
        val_query: List[Dict[str, str]] = []
        val_per_id: Dict[str, Dict[str, int]] = {}
        for id_name in val_ids:
            gallery_paths, query_paths = split_gallery_query(id_to_videos[id_name])
            val_gallery.extend(entry_from_path(output_root, path, id_name) for path in gallery_paths)
            val_query.extend(entry_from_path(output_root, path, id_name) for path in query_paths)
            val_per_id[id_name] = {"gallery": len(gallery_paths), "query": len(query_paths)}

        test_gallery: List[Dict[str, str]] = []
        test_query: List[Dict[str, str]] = []
        test_per_id: Dict[str, Dict[str, int]] = {}
        for id_name in test_ids:
            gallery_paths, query_paths = split_gallery_query(id_to_videos[id_name])
            test_gallery.extend(entry_from_path(output_root, path, id_name) for path in gallery_paths)
            test_query.extend(entry_from_path(output_root, path, id_name) for path in query_paths)
            test_per_id[id_name] = {"gallery": len(gallery_paths), "query": len(query_paths)}

        fold_payloads.append(
            {
                "fold_index": fold_index,
                "fold_name": f"fold{fold_index:02d}",
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": sorted(test_ids),
                "train_videos": train_videos,
                "val_gallery": val_gallery,
                "val_query": val_query,
                "test_gallery": test_gallery,
                "test_query": test_query,
                "stats": {
                    "train_video_count": len(train_videos),
                    "val_gallery_count": len(val_gallery),
                    "val_query_count": len(val_query),
                    "test_gallery_count": len(test_gallery),
                    "test_query_count": len(test_query),
                    "val_per_id": val_per_id,
                    "test_per_id": test_per_id,
                },
            }
        )

    return {
        "protocol": "strict_reid",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "flat_root": str((output_root / "flat_ids").relative_to(output_root).as_posix()),
        "balanced_by": balanced_by,
        "num_folds": num_folds,
        "seed": seed,
        "val_ratio": val_ratio,
        "min_videos_per_eval_id": min_videos_per_eval_id,
        "singleton_policy": singleton_policy,
        "eval_ids": eval_ids,
        "train_only_ids": train_only_ids,
        "excluded_ids": dropped_ids,
        "id_video_counts": id_to_count,
        "folds": fold_payloads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chapter-5 dataset and splits")
    parser.add_argument("--raw-root", type=str, default="兽医院-标注")
    parser.add_argument("--output-root", type=str, default="dataset_ch5")
    parser.add_argument("--protocol", type=str, choices=["strict_reid", "tracklet_halves"], default="strict_reid")
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", type=str, choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-videos-per-eval-id", type=int, default=2)
    parser.add_argument("--singleton-policy", type=str, choices=["drop", "train_only"], default="train_only")
    parser.add_argument("--balanced-by", type=str, choices=["video_count"], default="video_count")
    parser.add_argument(
        "--no-clean-folds",
        action="store_true",
        help="Keep existing fold directories instead of rebuilding split/train/val/test folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    flat_root = output_root / "flat_ids"
    folds_root = output_root / "folds"

    if not raw_root.exists():
        raise FileNotFoundError(f"raw-root does not exist: {raw_root}")
    ensure_dir(output_root)
    ensure_dir(flat_root)

    stats = build_flat_ids(raw_root, flat_root, copy_mode=args.copy_mode, overwrite=args.overwrite)
    num_ids = len(stats)
    num_raw = sum(stats.values())
    print(f"[1/3] Flatten done: ids={num_ids}, raw_videos={num_raw}")

    if args.protocol == "strict_reid":
        manifest = build_strict_reid_manifest(
            raw_root=raw_root,
            flat_root=flat_root,
            output_root=output_root,
            num_folds=args.num_folds,
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_videos_per_eval_id=args.min_videos_per_eval_id,
            singleton_policy=args.singleton_policy,
            balanced_by=args.balanced_by,
        )
        manifest_path = output_root / "strict_reid_splits.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(
            "[2/3] Strict ReID manifest done: "
            f"eval_ids={len(manifest['eval_ids'])}, "
            f"excluded_ids={len(manifest['excluded_ids'])}, "
            f"train_only_ids={len(manifest['train_only_ids'])}"
        )
        print(f"[3/3] Saved strict manifest -> {manifest_path}")
        return

    raw_count, clip1_count, clip2_count = split_all_flat_videos(flat_root, overwrite=args.overwrite)
    print(f"[2/3] Clip split done: raw={raw_count}, clip1={clip1_count}, clip2={clip2_count}")
    if clip1_count != raw_count or clip2_count != raw_count:
        print("[WARN] clip counts do not match raw count; please check problematic files.")

    id_list = sorted([p.name for p in flat_root.iterdir() if p.is_dir()])
    fold_specs = build_tracklet_fold_specs(id_list, args.num_folds, args.val_ratio, args.seed)
    materialize_tracklet_folds(
        flat_root=flat_root,
        folds_root=folds_root,
        fold_specs=fold_specs,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        seed=args.seed,
        clean_folds=not args.no_clean_folds,
    )
    print(f"[3/3] Tracklet-half fold materialization done: {len(fold_specs)} folds at {folds_root}")


if __name__ == "__main__":
    main()
