# WEEK01_BASICSETUP 项目说明

本 README 面向当前 WEEK01 阶段的可复现实验链路，以下是这个Repo的内容：

1. DataPrep 的完整调用流程
2. AUX_Head 中 ONE_HOT 与 CLIP 两条分支的脚本职责、使用模型、输入输出
3. 最终要用到的数据落点在 MERGED_RESULT
4. 我们使用的BC模型和训练链路

> 重要提醒
>
> 我在最终整合 AUX 进入流程前，调整过目录结构；
> 因此Repo内数据或者功能的期望路径可能已经与当前文件所处实际路径不一致。
> 运行前请优先检查每个脚本的默认参数，并按你的现有目录进行覆盖。

为了能够帮助复现，我们在仓库中提供了用到的 HF dataset；但你仍然需要做好路径调整的准备。

---

## 1) DataPrep 调用流程（主链路）

### Step 1. 读取 HF 驾驶数据集

- 脚本：`DataPrep/LLMAnnotation/LLMAnnotation.py`
- 关键参数：
  - `--dataset-dir`：默认指向 `WARM_UP_TASK/vlm/dataset/front_camera_hf` #是的，Week01 Warmup的时候，我仿照VLM-AD的方式，只把font camera的数据做成了一个HF Dataset
  - `--split`：`train` 或 `validate`
  - `--num-samples`：采样条数

该脚本会从 HF dataset 中取出图像与基础元数据（`scene_name`, `timestamp_str`, `index`）。

### Step 2. 用 VLM 模型执行双提示词标注

- 脚本：`DataPrep/LLMAnnotation/LLMAnnotation.py`
- 提示词定义：`DataPrep/LLMAnnotation/Prompt.py` #这个是提示词，请你参考里面的注释，记录了我调提示词的一些发现
- 使用模型：Gemma3-4b-it 视觉语言模型（通过 `mapModelPath("gemma-3-4b-it")` 映射本地模型目录）
- 推理接口：`transformers.pipeline("image-text-to-text")`

脚本会对每张图像跑两种 prompt：

1. FREEDOM 模式：输出自然语言驾驶解释（current/next/reasoning）
2. STRUCTURED 模式：输出结构化动作标志（control/turn/lane）

### Step 3. 汇总并写出标注 JSON

- 输出文件（默认）：`DataPrep/LLMAnnotation/annotation_outputs/llm_annotation_results.json`
- 目前已有可进入流程的文件：`DataPrep/LLMAnnotation/annotation_outputs/llm_annotation_results_500.json`
- 内容结构（核心）：
  - `results[i].metadata`（场景与时间）
  - `results[i].freedom_response`
  - `results[i].structured_response`

这个 JSON 是后续 ONE_HOT 与 CLIP 两条分支的共同输入来源。

---

## 2) AUX_Head 两条分支

`AUX_Head` 的作用是把 LLM 标注结果转成训练可用监督信号：

1. ONE_HOT 分支：结构化动作 -> 离散 one-hot 监督
2. CLIP 分支：自由文本 -> 语义 embedding 监督

### 2.1 ONE_HOT 分支

#### 脚本

- `AUX_Head/ONE_HOT/script/result_parser.py`

#### 输入

- 来自 DataPrep 的标注 JSON：
  - 默认 `--input-json llm_annotation_results_500.json`
  - 默认从 annotation 输出目录解析（可用 `--input-dir` 覆盖）
- 实际使用字段：每条记录的 `structured_response`

#### 处理逻辑

1. 解析 `structured_response`（优先 JSON 解析，失败则 regex 兜底）
2. 映射到固定动作集合：
	- `control_flag`: `go straight / move slowly / stop / reverse`
	- `turn_flag`: `turn left / turn right / turn around / none`
	- `lane_flag`: `change lane to the left / change lane to the right / merge into the left lane / merge into the right lane / none`
3. 生成：
	- 分组 one-hot
	- 扁平化向量 `flat_vector`

#### 输出

- 默认输出文件：`AUX_Head/ONE_HOT/output/structured_one_hot_annotations.json`
- 每条记录包含：
  - `metadata`
  - `flags`（字符串标签）
  - `one_hot`（按组 one-hot）
  - `flat_vector`

---

### 2.2 CLIP 分支

#### 脚本 1：解析自由文本并生成 embedding

- `AUX_Head/CLIP/scripts/result_parser.py`

#### 输入

- 来自 DataPrep 的标注 JSON：
  - 默认 `--input-json llm_annotation_results_500.json`
  - 可用 `--input-dir` 覆盖
- 使用字段：每条记录的 `freedom_response`

#### 使用模型

- 文本编码模型：`sentence-transformers/clip-ViT-B-32`
- 具体加载逻辑在：`AUX_Head/CLIP/scripts/eval.py`
  - 默认从本地缓存目录 `AUX_Head/CLIP/model_weights/.../snapshots/...` 读取
  - `load_model()` + `encode_texts()` 负责编码

#### 处理逻辑

1. 从 freedom 文本中抽取三段：
	- `current_action`
	- `next_action`
	- `reasoning`
2. 分别编码成向量（CLIP text embeddings）
3. 写回到 `content.text_embedding` 字段

#### 输出

- 默认输出文件：`AUX_Head/CLIP/output/freedom_annotations_for_clip.json`
- 每条记录包含：
  - `metadata`
  - 文本切片（current/next/reasoning）
  - 对应 embedding（列表形式）

#### 备注：`eval.py` 的定位

- `AUX_Head/CLIP/scripts/eval.py` 主要提供模型加载与编码函数（`load_model`, `encode_texts`, `encode_images`）

---

## 3) 最终合并：MERGED_RESULT（最终落点）

#### 脚本

- `AUX_Head/MERGED_RESULT/script/make_neo_dataset.py`

#### 输入

1. 原始 HF 驾驶数据集
	- 默认 `--source-dataset-dir WARM_UP_TASK/vlm/dataset/front_camera_hf`
	- 默认 `--source-split validate`
2. CLIP 分支输出
	- 默认 `AUX_Head/CLIP/output/freedom_annotations_for_clip.json`
3. ONE_HOT 分支输出
	- 默认 `AUX_Head/ONE_HOT/output/structured_one_hot_annotations.json`

#### 处理逻辑

1. 基于 metadata + index 对齐样本
2. 对缺失 `image_path` 的情况使用 fallback key（`scene_name + timestamp_str + index`）
3. 将以下字段并入源数据集：
	- `clip_current_action`, `clip_next_action`, `clip_reasoning`（float32）
	- `one_hot_control_flag`, `one_hot_turn_flag`, `one_hot_lane_flag`（int8）

#### 输出（最终）

- 默认输出目录：`AUX_Head/MERGED_RESULT/output_dataset/neo_hf_dataset`
- 这是当前链路最终产物（后续训练直接读取这个 merged dataset）
- 会附带 `build_summary.json` 记录构建参数与匹配数量

---

## 4) 我们训练的是什么模型

当前训练目标是 [`NetworkNvidiaParallel_VLM_AD`](AutoDriveModels/Dummy/model.py) 这个模型。

它是一个“英伟达的BC + 出自VLM-AD 那篇论文的 AUX 监督头”的联合训练结构：

1. 主干输入是 font camera 图像，输入形状在脚本里被整理成 `[B, 4200]` 或 `[B, 3, 70, 320]`
2. 主干输出三维控制回归量：`steering / throttle / brake`
3. 主干中间会产生 `f_ego`，再通过 projector 升维得到 `f_ego_proj`
4. `f_ego_proj` 会进入 AUX 头，去对齐两类监督：

	- CLIP 分支的文本 embedding：`clip_current_action`, `clip_next_action`, `clip_reasoning`
	- ONE_HOT 分支的动作标签：`one_hot_control_flag`, `one_hot_turn_flag`, `one_hot_lane_flag`

5. 训练损失目前是：

	- `MSELoss(output, regression_targets)`
	- `AUX_loss(f_ego_proj, clip_embeddings, action_labels)`



---

## 5) 目前 pipeline 连起来还差些什么

现在的链路已经分成了三段：

1. `DataPrep/LLMAnnotation/LLMAnnotation.py` 负责产出原始 LLM 标注 JSON
2. `AUX_Head/MERGED_RESULT/script/make_neo_dataset.py` 负责把 freedom / structured 两个分支合成 HuggingFace dataset
3. `TRAINING_SCRIPT/dummy_related/train_vlm_ad.py` 负责读取 merged dataset 并训练 `NetworkNvidiaParallel_VLM_AD`

但这三段目前还不是一个完全自动闭环，主要缺口有：

1. `train_vlm_ad.py` 还没有做基本的冒烟测试，因此可能有bug
2. 由于项目文件结构调整，Repo内数据或者功能的期望路径可能已经与当前文件所处实际路径不一致。因此数据或者脚本的加载可能有问题
3. 还没有一个统一的总控入口脚本，把“生成 neo dataset -> 校验 schema -> 启动训练”串成一个命令。
4. `train_vlm_ad.py` 现在还没有真正接上独立验证集，代码里 `val_loader=train_loader` 只是临时占位，不是完整训练闭环。


结论很直接：

- 模型本身已经选定为 `NetworkNvidiaParallel_VLM_AD`
- 数据从 LLM 标注到 merged dataset 已经打通
- 但训练闭环还缺“最基本的冒烟测试”


---

## 6) 目录结构（WEEK01 核心）

```text
WEEK01_BASICSETUP/
├─ README.md
├─ VLM-AD.pdf (复现的论文)
├─ DataPrep/
│  └─ LLMAnnotation/
│     ├─ LLMAnnotation.py
│     ├─ Prompt.py
│     └─ annotation_outputs/
│        └─ llm_annotation_results_500.json
├─ AUX_Head/
│  ├─ general_json_reader.py
│  ├─ ONE_HOT/
│  │  ├─ script/
│  │  │  └─ result_parser.py
│  │  └─ output/
│  │     └─ structured_one_hot_annotations.json
│  ├─ CLIP/
│  │  ├─ scripts/
│  │  │  ├─ result_parser.py
│  │  │  └─ eval.py
│  │  ├─ model_weights/
│  │  └─ output/
│  │     └─ freedom_annotations_for_clip.json
│  └─ MERGED_RESULT/
│  │  ├─ script/
│  │  │  └─ make_neo_dataset.py
│  │  └─ output_dataset/
│  │     └─ neo_hf_dataset/
│  └─ AutoDriveModels/
│     ├─ Dummy/
│     │  └─ model.py (改进的BC模型)
└─ TRAINING_SCRIPT/
	└─ dummy_related/
		├─ train_vlm_ad.py
		├─ dataset_vlm_ad.py
		└─ train_demo.py
  
```

---

## 7) 推荐执行顺序（最短可复现）

1. 先跑 DataPrep 标注
	- 产出：`llm_annotation_results_*.json`
2. 跑 ONE_HOT parser
	- 产出：`structured_one_hot_annotations.json`
3. 跑 CLIP parser
	- 产出：`freedom_annotations_for_clip.json`
4. 跑 MERGED_RESULT 合并脚本
	- 最终产出：`output_dataset/neo_hf_dataset`
5. 训练阶段（AUX 接入）
	- 使用：`TRAINING_SCRIPT/dummy_related/train_vlm_ad.py`

---
