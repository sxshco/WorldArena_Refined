import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset structure preprocessing script")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--gen_video_dir", type=str, required=True, help="Path to generated videos")
    parser.add_argument("--output_base", type=str, default="./data", help="Output base directory")
    return parser.parse_args()


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, output_dir: Path):
    """Extract a video into individual frames."""
    reset_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(str(output_dir / frame_name), frame)
        frame_count += 1
    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")


def _infer_task_id(item: dict, gt_video_path: Path) -> str:
    explicit = item.get("task_id")
    if explicit:
        return str(explicit)

    parent_name = gt_video_path.parent.name
    if parent_name:
        return parent_name

    parts = gt_video_path.parts
    if len(parts) >= 5:
        return parts[-5]

    raise ValueError(
        "Unable to infer task_id from gt_path. Please add task_id to summary.json entries."
    )


def _resolve_generated_video(item: dict, gen_video_dir: Path, task_id: str, episode_id: str, gt_video_path: Path) -> Path:
    explicit_name = item.get("generated_video_name") or item.get("generated_video")
    candidate_names = []
    if explicit_name:
        candidate_names.append(str(explicit_name))

    candidate_names.extend([
        f"{task_id}_{episode_id}.mp4",
        gt_video_path.name,
    ])

    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        candidate = gen_video_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Generated video not found. Tried: " + ", ".join(str(gen_video_dir / name) for name in seen)
    )


def process_item(item: dict, gen_video_dir: Path, output_base: Path):
    gt_video_path = Path(item["gt_path"])
    if not gt_video_path.exists():
        raise FileNotFoundError(f"GT video not found: {gt_video_path}")

    episode_id = gt_video_path.stem
    task_id = _infer_task_id(item, gt_video_path)

    gt_root = output_base / "gt_dataset" / task_id / episode_id
    prompt_dir = gt_root / "prompt"
    video_dir_gt = gt_root / "video"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    prompt_content = item.get("prompt")
    if isinstance(prompt_content, list):
        prompt_content = prompt_content[0] if prompt_content else ""
    if not isinstance(prompt_content, str) or not prompt_content.strip():
        raise ValueError(f"Missing prompt for {gt_video_path}")
    (prompt_dir / "prompt.txt").write_text(prompt_content, encoding="utf-8")

    src_image = Path(item["image"])
    if src_image.exists():
        shutil.copy2(src_image, prompt_dir / "init_frame.png")

    extract_frames(gt_video_path, video_dir_gt)

    gen_video_path = _resolve_generated_video(item, gen_video_dir, task_id, episode_id, gt_video_path)
    gen_root = output_base / "generated_dataset" / task_id / episode_id / "1"
    video_dir_gen = gen_root / "video"
    extract_frames(gen_video_path, video_dir_gen)

    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "gt_video": str(gt_video_path),
        "generated_video": str(gen_video_path),
    }


def main():
    args = parse_args()

    summary_path = Path(args.summary_json)
    gen_video_dir = Path(args.gen_video_dir)
    output_base = Path(args.output_base)

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    if not gen_video_dir.exists() or not gen_video_dir.is_dir():
        raise NotADirectoryError(f"Generated video directory not found: {gen_video_dir}")

    with summary_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("summary.json must be a non-empty list")

    print(f">>> Starting preprocessing for {len(data)} items...")
    processed = []
    for item in tqdm(data):
        processed.append(process_item(item, gen_video_dir, output_base))

    print(f"\n>>> Preprocessing Complete. Structure saved in: {output_base}")
    print(f">>> Successfully processed {len(processed)} items")


if __name__ == "__main__":
    main()
