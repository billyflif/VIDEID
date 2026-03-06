import argparse
from pathlib import Path

import cv2


def split_video_into_two_clips(video_path: Path, output_dir: Path, overwrite: bool = False) -> None:
    """Split one mp4 video into clip1 (first half) and clip2 (second half)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 1:
        print(f"[WARN] Too few frames: {video_path} (frames={total_frames})")
        cap.release()
        return

    mid_frame = total_frames // 2
    clip1_path = output_dir / f"{video_path.stem}_clip1.mp4"
    clip2_path = output_dir / f"{video_path.stem}_clip2.mp4"

    if not overwrite and clip1_path.exists() and clip2_path.exists():
        print(f"[INFO] Skip existing clips for: {video_path}")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer1 = cv2.VideoWriter(str(clip1_path), fourcc, fps, (width, height))
    writer2 = cv2.VideoWriter(str(clip2_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < mid_frame:
            writer1.write(frame)
        else:
            writer2.write(frame)
        frame_idx += 1

    cap.release()
    writer1.release()
    writer2.release()
    print(f"[OK] {video_path} -> {clip1_path.name}, {clip2_path.name}")


def process_dataset_root(dataset_root: Path, overwrite: bool = False, recursive: bool = False) -> None:
    """Split videos under dataset_root.

    - recursive=False: scans dataset_root/<id>/*.mp4
    - recursive=True: scans dataset_root/**/*.mp4
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    if recursive:
        video_paths = sorted(dataset_root.rglob("*.mp4"))
        for video_path in video_paths:
            if video_path.stem.endswith("_clip1") or video_path.stem.endswith("_clip2"):
                continue
            split_video_into_two_clips(video_path, video_path.parent, overwrite=overwrite)
        return

    for id_dir in sorted(dataset_root.iterdir()):
        if not id_dir.is_dir():
            continue
        for video_path in sorted(id_dir.glob("*.mp4")):
            if video_path.stem.endswith("_clip1") or video_path.stem.endswith("_clip2"):
                continue
            split_video_into_two_clips(video_path, id_dir, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split mp4 videos into clip1/clip2")
    parser.add_argument("--dataset-root", type=str, required=True, help="dataset root path")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing clip files")
    parser.add_argument("--recursive", action="store_true", help="scan nested subdirectories recursively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset_root(Path(args.dataset_root), overwrite=args.overwrite, recursive=args.recursive)


if __name__ == "__main__":
    main()

