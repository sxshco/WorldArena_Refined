## WorldArena Video Quality Evaluation Environment and Usage Guide

Scope:
- Standard visual quality metrics in `run_evaluation.sh`, including `image_quality`, `photometric_smoothness`, `motion_smoothness`, `aesthetic_quality`, `background_consistency`, `dynamic_degree`, `flow_score`, `subject_consistency`, `depth_accuracy`, `semantic_alignment`
- `action_following`
- VLM interaction quality / perspectivity / instruction following
- JEPA similarity

Environment split:
- `WorldArena`: standard visual quality metrics + `action_following`
- `World_VLM`: VLM evaluation
- `WorldArena_JEPA`: JEPA similarity

### 1. Prerequisites
- OS: Linux, CUDA 12.8 (aligned with the pinned torch cu128 stack in `requirements.txt`)
- Python: 3.10
- GPU: enough memory for RAFT, VFIMamba, CLIP, SAM, and VLM inference

After cloning the repo, create local placeholder directories before downloading weights:

```bash
cd WorldArena
mkdir -p video_quality/checkpoints/{clip,pyiqa,photometric,motion,raft,dino,aesthetic}
mkdir -p video_quality/checkpoints/{Qwen2.5-VL-7B-Instruct,depth-anything,qwenvl3,sam3}
```

### 2. Base Environment `WorldArena`
Used for all standard visual quality metrics and `action_following`.

```bash
cd WorldArena
conda create -y -n WorldArena python=3.10
conda activate WorldArena

pip install -U pip
pip install "setuptools<81" wheel
pip install --no-build-isolation mmcv==2.2.0
pip install -r video_quality/requirements.txt
pip install ipython
pip install ninja
pip install mamba-ssm
# Optional: pip install jupyter notebook jupyterlab
```

If you hit install issues from a stale build cache, clear cache and retry:

```bash
pip cache purge
pip install --no-build-isolation mmcv==2.2.0
pip install --no-cache-dir -r video_quality/requirements.txt
```

The current standard-metric code path has been checked against a `WorldArena` environment with:
- `torch==2.10.0+cu128`
- `torchvision==0.25.0+cu128`
- `torchaudio==2.10.0+cu128`
- importable `pyiqa`, `SEA-RAFT`, and `VFIMamba`

### 3. VLM Environment `World_VLM`
Used for Interaction Quality / Perspectivity / Instruction Following.

```bash
cd WorldArena
conda create -y -n World_VLM python=3.10
conda activate World_VLM
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 Pillow==10.0.0 PyYAML==6.0.3 tqdm==4.67.1 requests==2.32.5 sentencepiece==0.2.1 transformers==4.57.0 huggingface_hub==0.36.2 safetensors==0.7.0 accelerate==1.12.0 qwen-vl-utils==0.0.14 decord==0.6.0 opencv-python==4.11.0.86 av==17.0.0
```

### 4. JEPA Environment `WorldArena_JEPA`
Used for JEPA similarity.

```bash
cd WorldArena
conda create -y -n WorldArena_JEPA python=3.10
conda activate WorldArena_JEPA
pip install -r video_quality/requirements_jedi.txt
mkdir -p video_quality/JEDi/pretrained_models
cd video_quality/JEDi/pretrained_models
wget -O vith16.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
wget -O ssv2-probe.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
```

### 5. Input Data Rules

#### 5.1 `summary.json`
Each item in `summary.json` must at least contain:
- `gt_path`: absolute path to the GT video
- `image`: absolute path to the initial frame image
- `prompt`: a string or a one-element list

Recommended optional fields for a more stable preprocessing flow:
- `task_id`: explicit task/group id used to build `data/{gt_dataset,generated_dataset}/<task_id>/...`
- `generated_video_name`: explicit generated video filename inside `GEN_VIDEO_DIR`

Example:

```json
[
  {
    "gt_path": "/path/to/task_name/episode0.mp4",
    "image": "/path/to/task_name/episode0.png",
    "prompt": [
      "Lift the narrow-necked bottle using the right arm and hold upright."
    ]
  },
  {
    "gt_path": "/path/to/task_name/episode1.mp4",
    "image": "/path/to/task_name/episode1.png",
    "prompt": [
      "Move the cup to the target area."
    ]
  }
]
```

In the public copy, the sample file `../source/summary.json` uses repo-relative paths so it can be reused after cloning.

#### 5.2 Generated Video Directory for Standard Visual Quality Metrics
For `run_evaluation.sh`, the current preprocessing code expects a flat generated-video directory:
- `GEN_VIDEO_DIR` contains only generated `.mp4` files
- no subfolders inside `GEN_VIDEO_DIR`

The preprocessing resolves each generated video in this order:
1. `summary.json[i]["generated_video_name"]` or `summary.json[i]["generated_video"]` if provided
2. `{task_id}_{episode_id}.mp4`
3. the same basename as the GT video, for example `episode0.mp4`

The task id is resolved in this order:
1. `summary.json[i]["task_id"]`
2. `Path(gt_path).parent.name`
3. a legacy fallback derived from the deeper GT path structure

Recommended standard format:
- include `task_id` in each `summary.json` item
- name generated videos `{task_id}_{episode_id}.mp4`

Example:
- if `task_id = task0` and `gt_path = /.../episode0.mp4`
- then the recommended generated filename is `task0_episode0.mp4`

Important:
- The standard preprocessing step matches generated files by the rules above, not by `<MODEL_NAME>`
- If none of the candidate filenames exist, preprocessing now fails early with a clear error instead of silently skipping the sample

### 6. External Weights and Config
Configure local weights and I/O paths in [config](config/config.yaml). Keep `model_name: test` unless you also update the code that depends on it.

For the three commonly used visual quality metrics, the required config entries are:
- `ckpt.image_quality.musiq`
- `ckpt.photometric_smoothness.cfg`
- `ckpt.photometric_smoothness.model`
- `ckpt.motion_smoothness.model`

Default paths in the public copy are repo-relative placeholders:
- `image_quality`: `video_quality/checkpoints/pyiqa/musiq_spaq_ckpt-358bb6af.pth`
- `photometric_smoothness cfg`: `video_quality/WorldArena/third_party/SEA-RAFT/config/eval/spring-M.json`
- `photometric_smoothness model`: `video_quality/checkpoints/photometric/Tartan-C-T-TSKH-spring540x960-M.pth`
- `motion_smoothness model`: `video_quality/checkpoints/motion/VFIMamba.pkl`

Before running, also confirm these configured paths exist on disk.

Important:
- paths in `config/config.yaml` may now be repo-relative; the evaluation scripts resolve them automatically
- `run_evaluation.sh` and `run_action_following.sh` default to the current `python3` if `WORLD_ARENA_PYTHON` is not set
- local VLM inference uses `ckpt.vlm_model`, which defaults to `video_quality/checkpoints/qwenvl3`

On the first run, `photometric_smoothness` may also download an auxiliary `resnet34` backbone into `~/.cache/torch/hub/checkpoints/` if it is not already cached.

### 7. Run Evaluation
Use the following four flows separately. Do not mix their data layouts.

#### 7.1 Standard Visual Quality Metrics
This is the recommended flow for:
- `image_quality`
- `photometric_smoothness`
- `motion_smoothness`
- and the other non-VLM, non-JEPA metrics supported by `run_evaluation.sh`

Run:

```bash
cd video_quality
bash run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> "<METRIC_LIST>"
```

If your `WorldArena` environment python is not located at the default path, set it explicitly:

```bash
export WORLD_ARENA_PYTHON=/path/to/your/env/bin/python
```

Example:

```bash
cd video_quality
bash run_evaluation.sh my_model ../source/ours_test ../source/summary.json "image_quality,photometric_smoothness,motion_smoothness"
```

What `run_evaluation.sh` actually does:
- validates the requested metrics and required checkpoint paths from `config.yaml`
- preprocesses `summary.json` and `GEN_VIDEO_DIR` into `video_quality/data`
- extracts GT frames into `video_quality/data/gt_dataset/...`
- extracts generated frames into `video_quality/data/generated_dataset/...`
- resizes generated frames with `processing/video_resize.py`
- runs detection/tracking preprocessing
- runs `evaluate.py`

The generated structure used by standard visual quality metrics is:

```text
data
  ├── gt_dataset/
  │   ├── {task_id}/
  │   │   ├── {episode_id}/
  │   │   │   ├── prompt/
  │   │   │   │   ├── init_frame.png
  │   │   │   │   └── prompt.txt
  │   │   │   └── video/
  │   │   │       ├── frame_00000.jpg
  │   │   │       ├── ...
  │   │   │       └── frame_0000n.jpg
  ├── generated_dataset/
  │   ├── {task_id}/
  │   │   ├── {episode_id}/
  │   │   │   ├── 1/
  │   │   │   │   └── video/
  │   │   │   │       ├── frame_00000.jpg
  │   │   │   │       ├── ...
  │   │   │   │       └── frame_0000n.jpg
```

Notes:
- `image_quality`, `photometric_smoothness`, and `motion_smoothness` all read the preprocessed frame folders under `data/generated_dataset/.../1/video`
- They do not directly consume the original `GEN_VIDEO_DIR` during evaluation
- The shared metadata file is written to `output/generated_full_info.json` or `output/<data_name>_full_info.json`

#### 7.2 Action Following
`action_following` uses a different preprocessing pipeline and writes to `data_action_following`.

Run:

```bash
cd video_quality
bash run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON>
```

Its preprocessing builds this structure:

```text
data_action_following
  ├── gt_dataset/
  ├── generated_dataset/
```

For diversity/action variants, prepare three flat generated-video directories:
- `modelname_test`
- `modelname_test_1`
- `modelname_test_2`

If using a specific split, run `preprocess_datasets_diversity.py` first or ensure `config.data_action_following` exists.

#### 7.3 VLM Metrics
VLM metrics (interaction quality, perspectivity, instruction following) support two backends:
- `local`: local Qwen VLM
- `api`: OpenAI-compatible API

You can choose the backend via `VLM_BACKEND` or by passing the 6th argument to `run_VLM_judge.sh`.

Local backend example:
```bash
cd video_quality
bash run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> all "" local
```

OpenAI-compatible API backend example:
```bash
cd video_quality
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=https://your-compatible-endpoint/v1
export OPENAI_MODEL=your_vision_model
bash run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> all "" api
```

Notes:
- The API backend reuses the current frame-sampling logic and sends sampled frames as images to `/chat/completions`
- The output JSON format is kept consistent with the local backend
- If the 6th argument is omitted, `VLM_BACKEND` is used when available

#### 7.4 JEPA Similarity
JEPA similarity requires `WorldArena_JEPA`.

```bash
cd video_quality
export WORLD_ARENA_JEPA_ENV=WorldArena_JEPA
bash run_evaluation_JEPA.sh <VIDEO_DIR>
```

If your GT directory is not `source/gt_video`, pass it explicitly:

```bash
cd video_quality
bash run_evaluation_JEPA.sh <VIDEO_DIR> /path/to/gt_video
```

### 8. Result Aggregation
Metric aggregation requires `WorldArena`:

```bash
python video_quality/csv_results/aggregate_results.py --model_name <MODEL_NAME> --base_dir . --csv_name aggregated_results.csv
```

You can view aggregated and per-metric outputs under `video_quality/csv_results` and the raw JSON outputs under `video_quality/output`.

### 9. Public Release Checklist
- keep `video_quality/checkpoints/` local only; it is ignored by git
- keep generated outputs under `video_quality/output*`, `tmp_VLM/`, and `data*` local only
- if you add new examples, avoid committing absolute paths in JSON, YAML, or markdown
