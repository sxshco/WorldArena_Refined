import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests
import yaml
from PIL import Image
from tqdm import tqdm

from config_utils import load_yaml_config

DEFAULT_MODEL_PATH = "../checkpoints/qwenvl3"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_MODEL = "gpt-4.1-mini"
DEFAULT_BACKEND = "local"


def _episode_key_from_name(name: str):
    stem = Path(name).stem
    m = re.search(r"(episode[_-]?\d+)$", stem)
    if m is None:
        return None
    return m.group(1).replace("_", "-")


def load_instruction_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instruction_map = {}
    instruction_map_by_episode = {}

    for item in data:
        gt_path = item.get("gt_path", "")
        prompt = item.get("prompt", "")
        if isinstance(prompt, list) and prompt:
            instruction = prompt[0]
        elif isinstance(prompt, str):
            instruction = prompt
        else:
            instruction = ""

        path_obj = Path(gt_path)
        if len(path_obj.parts) >= 5:
            generated_filename = f"{path_obj.parts[-1].split('.')[0]}.mp4"
        else:
            base_name = os.path.basename(gt_path)
            generated_filename = f"unknown_{base_name}"

        instruction_map[generated_filename] = instruction

        ep_key = _episode_key_from_name(generated_filename)
        if ep_key is not None and ep_key not in instruction_map_by_episode:
            instruction_map_by_episode[ep_key] = instruction

    return instruction_map, instruction_map_by_episode


def sample_frames(video_path, num_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    if total_frames <= num_frames:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    else:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames[:num_frames]


def build_multi_metric_prompt(instruction_text=None):
    instruction_block = ""
    if instruction_text:
        instruction_block = f"\n**VIDEO INSTRUCTION:** {instruction_text}\n"
    return f"""
You are an expert evaluator for robot interaction videos. You are evaluating videos generated for embodied AI manipulation scenarios, specifically focusing on robotic arms interacting with objects in tabletop environments.{instruction_block}
EVALUATION CONTEXT:
- Target scenario: Robotic manipulation (e.g., pick-place, push, grasp)
- Expected agent: Robotic arm/end-effector, NOT human hands
- Expected environment: Tabletop with objects, typical for robot manipulation tasks
- Expected physics: Realistic robot-object interactions following physical laws
CRITICAL EVALUATION PRINCIPLES:
1. Base ALL judgments ONLY on what is visually observable in the sampled frames
2. DO NOT infer information not shown
3. Evaluate temporal coherence across the sampled frames
4. For instruction following: Compare STRICTLY against the provided text instruction
EVALUATION DIMENSIONS & SCORING RUBRICS:
1. Interaction_Quality
- Score 1: Objects pass through robot or other objects; no proper contact
- Score 2: Contact exists but interaction is unrealistic
- Score 3: Mostly plausible interactions with minor issues
- Score 4: Realistic contact physics
- Score 5: Perfect interaction physics
2. Perspectivity
- Score 1: Scene has no coherent 3D structure
- Score 2: 3D structure is unstable
- Score 3: Reasonable 3D consistency with minor issues
- Score 4: Stable camera perspective with consistent depth relationships
- Score 5: Perfect camera geometry and 3D consistency
3. Instruction_Following
- HALLUCINATION CHECK: If the video shows human hands instead of robotic arms, score <= 2 immediately
- Score 1: Completely different from instruction
- Score 2: Partially related but major errors
- Score 3: Follows general intent but with execution errors
- Score 4: Mostly correct with minor deviations
- Score 5: Perfect execution of all specified elements
OUTPUT FORMAT REQUIREMENTS:
You MUST output a SINGLE valid JSON object with EXACTLY these keys:
- Interaction_Quality
- Perspectivity
- Instruction_Following
Each value must be an object with exactly these keys:
- score: integer 1-5
- reason: concise explanation citing specific visual evidence
CRITICAL INSTRUCTIONS:
1. Output ONLY the JSON object, no other text
2. Use observed visual evidence only
3. Be strict about instruction following
4. Consider temporal coherence across all sampled frames
Now evaluate the provided video frames.
""".strip()


def pil_image_to_data_url(image: Image.Image, image_format: str = "JPEG"):
    buf = io.BytesIO()
    image.save(buf, format=image_format)
    mime = "image/jpeg" if image_format.upper() == "JPEG" else f"image/{image_format.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _extract_text_from_chat_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def run_openai_compatible_vlm(prompt, images, api_cfg):
    api_key = api_cfg["api_key"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY (or configured api_key_env) is required for backend=api")

    base_url = api_cfg["base_url"].rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    content = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": pil_image_to_data_url(image)},
        })

    payload = {
        "model": api_cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": api_cfg["max_tokens"],
    }
    if api_cfg.get("json_mode", False):
        payload["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(1, api_cfg["max_retries"] + 1):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=api_cfg["timeout"])
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices returned: {json.dumps(data)[:1000]}")
            message = choices[0].get("message") or {}
            return _extract_text_from_chat_content(message.get("content", ""))
        except Exception as e:
            last_error = e
            if attempt < api_cfg["max_retries"]:
                time.sleep(api_cfg["retry_sleep"] * attempt)
    raise RuntimeError(f"API request failed after {api_cfg['max_retries']} attempts: {last_error}")


def run_local_qwen_vl(model, processor, prompt, video_path, num_frames):
    from submodel.qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"video": video_path, "nframes": num_frames},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs.get("fps")
    if isinstance(fps_inputs, (list, tuple)):
        fps_inputs = fps_inputs[0] if len(fps_inputs) > 0 else None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    gen_trim = generated_ids[:, input_len:]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def extract_json_block(raw_text):
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def normalize_metrics(parsed):
    metrics = {}
    for key in ["Interaction_Quality", "Perspectivity", "Instruction_Following"]:
        val = parsed.get(key, {}) if isinstance(parsed, dict) else {}
        score = val.get("score", 0)
        reason = val.get("reason", "")
        metrics[key] = {
            "score": score,
            "reason": reason,
            "score_normalized": round(score / 5.0, 4) if isinstance(score, (int, float)) else 0.0,
        }
    return metrics


def load_backend_config(config_path):
    cfg = {}
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_yaml_config(config_path)
        except Exception:
            cfg = {}

    api_section = cfg.get("vlm_api", {}) if isinstance(cfg, dict) else {}
    ckpt_section = cfg.get("ckpt", {}) if isinstance(cfg, dict) else {}
    api_key_env = api_section.get("api_key_env", "OPENAI_API_KEY")

    return {
        "model_path": ckpt_section.get("vlm_model"),
        "backend": api_section.get("backend") or os.environ.get("VLM_BACKEND") or DEFAULT_BACKEND,
        "api_base_url": api_section.get("base_url") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_API_BASE_URL,
        "api_model": api_section.get("model") or os.environ.get("OPENAI_MODEL") or DEFAULT_API_MODEL,
        "api_key": os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY"),
        "api_key_env": api_key_env,
        "api_timeout": int(api_section.get("timeout", os.environ.get("VLM_API_TIMEOUT", 120))),
        "api_max_retries": int(api_section.get("max_retries", os.environ.get("VLM_API_MAX_RETRIES", 3))),
        "api_retry_sleep": float(api_section.get("retry_sleep", os.environ.get("VLM_API_RETRY_SLEEP", 2))),
        "api_max_tokens": int(api_section.get("max_tokens", os.environ.get("VLM_API_MAX_TOKENS", 800))),
        "api_json_mode": str(api_section.get("json_mode", os.environ.get("VLM_API_JSON_MODE", "0"))).lower() in {"1", "true", "yes"},
    }


def build_local_backend(model_path):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    return model, processor


def vlm_judge(model_name, video_dir, summary_json, output_root, tmp_root, metrics_filter, num_frames=16, backend="local", model_path=None, api_cfg=None):
    os.makedirs(output_root, exist_ok=True)
    tmp_dir = os.path.join(tmp_root, model_name)
    os.makedirs(tmp_dir, exist_ok=True)

    local_model = None
    local_processor = None
    if backend == "local":
        model_path = model_path or DEFAULT_MODEL_PATH
        local_model, local_processor = build_local_backend(model_path)
    elif backend != "api":
        raise ValueError(f"Unsupported backend: {backend}")

    instruction_map, instruction_map_by_episode = load_instruction_json(summary_json)

    def resolve_instruction(video_filename):
        if video_filename in instruction_map:
            return instruction_map[video_filename]
        ep_key = _episode_key_from_name(video_filename)
        if ep_key is not None and ep_key in instruction_map_by_episode:
            return instruction_map_by_episode[ep_key]
        return None

    videos = []
    video_instruction_map = {}
    for fname in os.listdir(video_dir):
        if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            continue
        instruction = resolve_instruction(fname)
        if instruction is None:
            continue
        video_path = os.path.join(video_dir, fname)
        videos.append(video_path)
        video_instruction_map[video_path] = instruction
    videos.sort()

    print(f"[INFO] backend: {backend}")
    print(f"[INFO] matched videos for VLM eval: {len(videos)}")

    results = []
    for video_path in tqdm(videos, desc=f"{model_name} evaluating", ncols=100):
        item = {"video": os.path.basename(video_path), "metrics": {}, "raw_response_file": None, "error": None, "backend": backend}
        frames = sample_frames(video_path, num_frames=num_frames)
        if not frames:
            item["error"] = "no frames"
            results.append(item)
            continue

        instruction = video_instruction_map.get(video_path, "")
        prompt = build_multi_metric_prompt(instruction)

        try:
            if backend == "api":
                raw_text = run_openai_compatible_vlm(prompt, frames, api_cfg)
            else:
                raw_text = run_local_qwen_vl(local_model, local_processor, prompt, video_path, num_frames)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = os.path.join(tmp_dir, f"{Path(video_path).stem}_{timestamp}.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump({"raw_response": raw_text}, f, ensure_ascii=False, indent=2)
            item["raw_response_file"] = raw_file

            parsed = extract_json_block(raw_text)
            if parsed is None:
                item["error"] = "parse_failed"
            else:
                item["metrics"] = normalize_metrics(parsed)
        except Exception as e:
            item["error"] = str(e)

        results.append(item)

    out_dir = os.path.join(output_root, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{model_name}_summary_val_all_intern.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--output_root", default="output_VLM")
    parser.add_argument("--tmp_root", default="tmp_VLM")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--config_path", default="video_quality/config/config.yaml")
    parser.add_argument("--backend", choices=["local", "api"], default=None)
    args = parser.parse_args()

    backend_cfg = load_backend_config(args.config_path)
    backend = args.backend or backend_cfg["backend"]
    api_cfg = {
        "base_url": backend_cfg["api_base_url"],
        "model": backend_cfg["api_model"],
        "api_key": backend_cfg["api_key"],
        "timeout": backend_cfg["api_timeout"],
        "max_retries": backend_cfg["api_max_retries"],
        "retry_sleep": backend_cfg["api_retry_sleep"],
        "max_tokens": backend_cfg["api_max_tokens"],
        "json_mode": backend_cfg["api_json_mode"],
    }

    out_file = vlm_judge(
        model_name=args.model_name,
        video_dir=args.video_dir,
        summary_json=args.summary_json,
        output_root=args.output_root,
        tmp_root=args.tmp_root,
        metrics_filter=args.metrics,
        num_frames=args.num_frames,
        backend=backend,
        model_path=backend_cfg["model_path"],
        api_cfg=api_cfg,
    )
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
