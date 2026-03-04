import argparse
from pathlib import Path

import cv2


def split_video_into_two_clips(video_path: Path, output_dir: Path, overwrite: bool = False) -> None:
    """将单个 mp4 视频切分为前半段和后半段两个片段。

    输入结构假设为：dataset_root/idXXXX/0001.mp4
    输出为同一目录下：0001_clip1.mp4 和 0001_clip2.mp4
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 1:
        print(f"[WARN] 视频帧数过少: {video_path} (frames={total_frames})")
        cap.release()
        return

    mid_frame = total_frames // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    stem = video_path.stem  # 例如 0001
    clip1_path = output_dir / f"{stem}_clip1.mp4"
    clip2_path = output_dir / f"{stem}_clip2.mp4"

    if not overwrite and clip1_path.exists() and clip2_path.exists():
        print(f"[INFO] 已存在切分结果，跳过: {video_path}")
        cap.release()
        return

    writer1 = cv2.VideoWriter(str(clip1_path), fourcc, fps, (width, height))
    writer2 = cv2.VideoWriter(str(clip2_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
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


def process_dataset_root(dataset_root: Path, overwrite: bool = False) -> None:
    """遍历 dataset_root/idXXXX/ 结构，对其中的 mp4 进行切分。"""
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 不存在: {dataset_root}")

    for id_dir in sorted(dataset_root.iterdir()):
        if not id_dir.is_dir():
            continue
        for video_path in sorted(id_dir.glob("*.mp4")):
            # 只处理未切分的原始视频：排除 *_clip1.mp4 / *_clip2.mp4
            if video_path.stem.endswith("_clip1") or video_path.stem.endswith("_clip2"):
                continue
            split_video_into_two_clips(video_path, id_dir, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 dataset_root 下的每个 mp4 切成两个 clip")
    parser.add_argument("--dataset-root", type=str, required=True, help="数据根目录，例如 ./dataset_root")
    parser.add_argument("--overwrite", action="store_true", help="若已存在 *_clip1/_clip2 则覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    process_dataset_root(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
