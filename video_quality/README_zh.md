## WorldArena 视频质量评测环境与使用说明

适用范围：
- `run_evaluation.sh` 中的标准视觉质量指标，包括 `image_quality`、`photometric_smoothness`、`motion_smoothness`、`aesthetic_quality`、`background_consistency`、`dynamic_degree`、`flow_score`、`subject_consistency`、`depth_accuracy`、`semantic_alignment`
- `action_following`
- VLM 交互质量 / 视角质量 / 指令遵循
- JEPA similarity

环境划分：
- `WorldArena`：标准视觉质量指标 + `action_following`
- `World_VLM`：VLM 评测
- `WorldArena_JEPA`：JEPA similarity

### 1. 前置要求
- 操作系统：Linux，CUDA 12.8（与 `requirements.txt` 中固定的 torch cu128 栈一致）
- Python：3.10
- GPU：需要足够显存以支持 RAFT、VFIMamba、CLIP、SAM 和 VLM 推理

### 2. 基础环境 `WorldArena`
用于所有标准视觉质量指标以及 `action_following`。

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
# 可选：pip install jupyter notebook jupyterlab
```

如果因为旧缓存导致安装失败，可以清理缓存后重试：

```bash
pip cache purge
pip install --no-build-isolation mmcv==2.2.0
pip install --no-cache-dir -r video_quality/requirements.txt
```

当前标准指标流程已经在如下 `WorldArena` 环境中验证过：
- `torch==2.10.0+cu128`
- `torchvision==0.25.0+cu128`
- `torchaudio==2.10.0+cu128`
- `pyiqa`、`SEA-RAFT`、`VFIMamba` 可正常导入

### 3. VLM 环境 `World_VLM`
用于 Interaction Quality / Perspectivity / Instruction Following。

```bash
cd WorldArena
conda create -y -n World_VLM python=3.10
conda activate World_VLM
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 Pillow==10.0.0 PyYAML==6.0.3 tqdm==4.67.1 requests==2.32.5 sentencepiece==0.2.1 transformers==4.57.0 huggingface_hub==0.36.2 safetensors==0.7.0 accelerate==1.12.0 qwen-vl-utils==0.0.14 decord==0.6.0 opencv-python==4.11.0.86 av==17.0.0
```

### 4. JEPA 环境 `WorldArena_JEPA`
用于 JEPA similarity。

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

### 5. 输入数据规则

#### 5.1 `summary.json`
`summary.json` 中的每一项至少需要包含：
- `gt_path`：GT 视频的绝对路径
- `image`：初始帧图片的绝对路径
- `prompt`：字符串，或只包含一个元素的列表

为了让预处理更稳定，推荐额外提供以下字段：
- `task_id`：显式指定任务 / 分组 id，用于构建 `data/{gt_dataset,generated_dataset}/<task_id>/...`
- `generated_video_name`：显式指定 `GEN_VIDEO_DIR` 中对应生成视频的文件名

示例：

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

#### 5.2 标准视觉质量评测的生成视频目录
对于 `run_evaluation.sh`，当前预处理代码要求生成视频目录满足：
- `GEN_VIDEO_DIR` 中只放生成的 `.mp4` 文件
- `GEN_VIDEO_DIR` 内不要再有子目录

预处理会按照以下顺序解析每条样本对应的生成视频：
1. 如果提供了 `summary.json[i]["generated_video_name"]` 或 `summary.json[i]["generated_video"]`，优先使用它
2. 尝试匹配 `{task_id}_{episode_id}.mp4`
3. 尝试匹配与 GT 视频同名的文件，例如 `episode0.mp4`

`task_id` 的解析顺序如下：
1. `summary.json[i]["task_id"]`
2. `Path(gt_path).parent.name`
3. 从更深层 GT 路径结构中做兼容性回退推断

推荐的标准格式：
- 在每个 `summary.json` 条目中显式写入 `task_id`
- 生成视频统一命名为 `{task_id}_{episode_id}.mp4`

示例：
- 若 `task_id = task0` 且 `gt_path = /.../episode0.mp4`
- 推荐的生成视频文件名为 `task0_episode0.mp4`

注意：
- 标准预处理是按上述规则匹配生成视频文件，而不是按 `<MODEL_NAME>` 匹配
- 如果所有候选文件名都找不到，预处理现在会直接报清晰错误，而不会再静默跳过该样本

#### 5.3 待评估数据目录结构示例
下面给出一个在运行 `run_evaluation.sh` 之前，用户自行准备好的待评估数据目录示例：

```text
/path/to/eval_case/
├── gt/
│   ├── task0/
│   │   ├── episode0.mp4
│   │   └── episode0.png
│   └── task1/
│       ├── episode1.mp4
│       └── episode1.png
├── generated_videos/
│   ├── task0_episode0.mp4
│   └── task1_episode1.mp4
└── summary.json
```

对应的 `summary.json` 可以写成：

```json
[
  {
    "task_id": "task0",
    "gt_path": "/path/to/eval_case/gt/task0/episode0.mp4",
    "image": "/path/to/eval_case/gt/task0/episode0.png",
    "prompt": [
      "Lift the narrow-necked bottle using the right arm and hold upright."
    ]
  },
  {
    "task_id": "task1",
    "gt_path": "/path/to/eval_case/gt/task1/episode1.mp4",
    "image": "/path/to/eval_case/gt/task1/episode1.png",
    "prompt": [
      "Move the cup to the target area."
    ]
  }
]
```

然后执行：

```bash
cd video_quality
bash run_evaluation.sh my_model /path/to/eval_case/generated_videos /path/to/eval_case/summary.json "image_quality,photometric_smoothness,motion_smoothness"
```

### 6. 外部权重与配置
请在 [config](config/config.yaml) 中配置本地权重和 I/O 路径。除非你同时修改了依赖它的代码，否则请保持 `model_name: test` 不变。

针对最常用的三个视觉质量指标，必须配置：
- `ckpt.image_quality.musiq`
- `ckpt.photometric_smoothness.cfg`
- `ckpt.photometric_smoothness.model`
- `ckpt.motion_smoothness.model`

公开副本中的默认路径已改为仓库相对占位路径：
- `image_quality`：`video_quality/checkpoints/pyiqa/musiq_spaq_ckpt-358bb6af.pth`
- `photometric_smoothness cfg`：`video_quality/WorldArena/third_party/SEA-RAFT/config/eval/spring-M.json`
- `photometric_smoothness model`：`video_quality/checkpoints/photometric/Tartan-C-T-TSKH-spring540x960-M.pth`
- `motion_smoothness model`：`video_quality/checkpoints/motion/VFIMamba.pkl`

运行前还请确认这些配置路径在本机都真实存在。

首次运行时，`photometric_smoothness` 还可能会自动下载一个辅助的 `resnet34` backbone 到 `~/.cache/torch/hub/checkpoints/`，如果本地还没有缓存的话。

### 7. 运行评测
下面四类流程请分别使用，不要混用它们的数据组织方式。

#### 7.1 标准视觉质量指标
推荐用于：
- `image_quality`
- `photometric_smoothness`
- `motion_smoothness`
- 以及 `run_evaluation.sh` 支持的其他非 VLM、非 JEPA 指标

运行方式：

```bash
cd video_quality
bash run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> "<METRIC_LIST>"
```

如果你的 `WorldArena` 环境 python 不在默认路径，请先显式指定：

```bash
export WORLD_ARENA_PYTHON=/path/to/your/env/bin/python
```

示例：

```bash
cd video_quality
bash run_evaluation.sh \
  my_model \
  /path/to/generated_videos \
  /path/to/summary.json \
  "image_quality,photometric_smoothness,motion_smoothness"
```

`run_evaluation.sh` 实际会执行以下步骤：
- 根据 `config.yaml` 检查所请求指标依赖的 checkpoint 路径是否存在
- 将 `summary.json` 与 `GEN_VIDEO_DIR` 预处理到 `video_quality/data`
- 将 GT 视频拆帧到 `video_quality/data/gt_dataset/...`
- 将生成视频拆帧到 `video_quality/data/generated_dataset/...`
- 通过 `processing/video_resize.py` 调整生成帧尺寸
- 执行 detection / tracking 预处理
- 运行 `evaluate.py`

标准视觉质量指标实际使用的数据结构如下：

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

补充说明：
- `image_quality`、`photometric_smoothness`、`motion_smoothness` 实际读取的是 `data/generated_dataset/.../1/video` 下的预处理帧目录
- 它们在正式评测阶段不会直接读取原始 `GEN_VIDEO_DIR`
- 共享的评测元数据会写到 `output/generated_full_info.json` 或 `output/<data_name>_full_info.json`

#### 7.2 Action Following
`action_following` 使用另一套预处理流程，输出写到 `data_action_following`。

运行方式：

```bash
cd video_quality
bash run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON>
```

其预处理生成的目录结构为：

```text
data_action_following
  ├── gt_dataset/
  ├── generated_dataset/
```

如果你要做 diversity / action variant，请准备三个扁平生成视频目录：
- `modelname_test`
- `modelname_test_1`
- `modelname_test_2`

如果使用特定 split，请先运行 `preprocess_datasets_diversity.py`，或者确保 `config.data_action_following` 已正确存在。

#### 7.3 VLM 指标
VLM 指标（interaction quality、perspectivity、instruction following）支持双后端：
- `local`：本地 Qwen VLM
- `api`：OpenAI 兼容接口

默认后端可通过环境变量 `VLM_BACKEND` 控制，也可以作为 `run_VLM_judge.sh` 的第 6 个参数传入。

本地后端示例：
```bash
cd video_quality
bash run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> all "" local
```

OpenAI 兼容 API 后端示例：
```bash
cd video_quality
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=https://your-compatible-endpoint/v1
export OPENAI_MODEL=your_vision_model
bash run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> all "" api
```

说明：
- API 后端会沿用当前抽帧逻辑，把采样帧作为多张图片发送到 `/chat/completions`
- 输出 JSON 结构与本地后端保持一致
- 如果不传第 6 个参数，则优先读取 `VLM_BACKEND`

#### 7.4 JEPA Similarity
JEPA similarity 需要使用 `WorldArena_JEPA` 环境。

```bash
cd video_quality
export WORLD_ARENA_JEPA_ENV=WorldArena_JEPA
bash run_evaluation_JEPA.sh <VIDEO_DIR> /path/to/gt_video
```

第二个参数必须显式提供 GT 视频目录，因为公开仓库不再内置 `source/gt_video`。

### 8. 结果汇总
结果汇总使用 `WorldArena` 环境：

```bash
python video_quality/csv_results/aggregate_results.py --model_name <MODEL_NAME> --base_dir . --csv_name aggregated_results.csv
```

汇总后的 CSV 和各指标结果可以在 `video_quality/csv_results` 下查看，原始 JSON 输出可以在 `video_quality/output` 下查看。
