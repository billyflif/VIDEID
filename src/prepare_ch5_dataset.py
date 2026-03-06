from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
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


def build_flat_ids(raw_root: Path, flat_root: Path, overwrite: bool) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for hospital_name, id_name, src in iter_raw_videos(raw_root):
        id_dir = flat_root / id_name
        ensure_dir(id_dir)
        prefixed_name = f"{hospital_name}__{src.name}"
        dst = id_dir / prefixed_name
        if dst.exists() and not overwrite:
            stats[id_name] = stats.get(id_name, 0) + 1
            continue
        shutil.copy2(src, dst)
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


def build_fold_specs(ids: Sequence[str], num_folds: int, val_ratio: float, seed: int) -> List[FoldSpec]:
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
        val_ids = remain[:val_n]
        train_ids = remain[val_n:]
        if not train_ids or not val_ids or not test_ids:
            raise RuntimeError(
                f"Invalid split in fold {fold_idx}: "
                f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
            )

        specs.append(
            FoldSpec(
                fold_index=fold_idx,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=list(test_ids),
            )
        )
    return specs


def list_id_clip_files(flat_root: Path, id_name: str) -> List[Path]:
    id_dir = flat_root / id_name
    clips = sorted(id_dir.glob("*_clip1.mp4")) + sorted(id_dir.glob("*_clip2.mp4"))
    return clips


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


def materialize_folds(
    flat_root: Path,
    folds_root: Path,
    fold_specs: Sequence[FoldSpec],
    copy_mode: str,
    overwrite: bool,
    seed: int,
    clean_folds: bool,
) -> None:
    ensure_dir(folds_root)
    expected_fold_names = [f"fold{spec.fold_index:02d}" for spec in fold_specs]
    if clean_folds:
        prune_stale_fold_dirs(folds_root, expected_fold_names)
    all_manifest = []
    for spec in fold_specs:
        fold_name = f"fold{spec.fold_index:02d}"
        fold_dir = folds_root / fold_name
        ensure_dir(fold_dir)
        if clean_folds:
            clear_generated_fold_dir(fold_dir)
        for split_name, ids in (
            ("train", spec.train_ids),
            ("val", spec.val_ids),
            ("test", spec.test_ids),
        ):
            split_root = fold_dir / split_name
            ensure_dir(split_root)
            for id_name in ids:
                dst_id_dir = split_root / id_name
                ensure_dir(dst_id_dir)
                for clip_path in list_id_clip_files(flat_root, id_name):
                    link_or_copy(clip_path, dst_id_dir / clip_path.name, copy_mode=copy_mode, overwrite=overwrite)

        manifest = {
            "fold_index": spec.fold_index,
            "train_ids": spec.train_ids,
            "val_ids": spec.val_ids,
            "test_ids": spec.test_ids,
            "seed": seed,
        }
        with open(fold_dir / "split.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        all_manifest.append(manifest)

    with open(folds_root / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"folds": all_manifest}, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chapter-5 dataset and folds")
    parser.add_argument("--raw-root", type=str, default="兽医院-标注")
    parser.add_argument("--output-root", type=str, default="dataset_ch5")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", type=str, choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
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

    stats = build_flat_ids(raw_root, flat_root, overwrite=args.overwrite)
    num_ids = len(stats)
    num_raw = sum(stats.values())
    print(f"[1/3] Flatten done: ids={num_ids}, raw_videos={num_raw}")

    raw_count, clip1_count, clip2_count = split_all_flat_videos(flat_root, overwrite=args.overwrite)
    print(f"[2/3] Clip split done: raw={raw_count}, clip1={clip1_count}, clip2={clip2_count}")
    if clip1_count != raw_count or clip2_count != raw_count:
        print("[WARN] clip counts do not match raw count; please check problematic files.")

    id_list = sorted([p.name for p in flat_root.iterdir() if p.is_dir()])
    fold_specs = build_fold_specs(id_list, args.num_folds, args.val_ratio, args.seed)
    materialize_folds(
        flat_root=flat_root,
        folds_root=folds_root,
        fold_specs=fold_specs,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        seed=args.seed,
        clean_folds=not args.no_clean_folds,
    )
    print(f"[3/3] Fold materialization done: {len(fold_specs)} folds at {folds_root}")


if __name__ == "__main__":
    main()
