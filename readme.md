# RefDrone 大模型评测

基于 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 框架，对 Qwen3-VL 模型在 RefDrone 数据集上进行视觉定位（Visual Grounding）能力评测。

---

## 项目结构

```
refdrone/
├── VLMEvalKit/
│   ├── eval_qwen3vl.sh                        # 评测启动脚本（默认 4 卡）
│   ├── vlmeval/
│   │   ├── config.py                          # 模型注册配置
│   │   ├── dataset/
│   │   │   └── image_grounding.py             # RefDrone 数据集定义与指标计算
│   │   └── vlm/qwen3_vl/
│   │       ├── model.py                       # Qwen3-VL 模型推理
│   │       └── prompt.py                      # Prompt 构建逻辑
│   └── README.md                              # VLMEvalKit 原始文档
└── readme.md
```

---

## 主要修改说明

### 1. 模型注册 — `vlmeval/config.py`

新增 `Qwen3-VL-8B-Instruct_Grounding` 模型配置，开启视觉定位模式：

```python
"Qwen3-VL-8B-Instruct_Grounding": partial(
    vlm.Qwen3VLChat,
    model_path="Qwen/Qwen3-VL-8B-Instruct",
    use_custom_prompt=True,
    use_vllm=False,
    temperature=0.7,
    max_new_tokens=16384,
    repetition_penalty=1.0,
    presence_penalty=1.5,
    top_p=0.8,
    top_k=20,
    visual_grounding=True,
),
```

### 2. Prompt 构建 — `vlmeval/vlm/qwen3_vl/prompt.py`

针对 VG（Visual Grounding）任务类型定制 prompt 格式，确保模型输出符合坐标解析要求。

### 3. 数据集定义 — `vlmeval/dataset/image_grounding.py`

实现 `refdrone_test` 数据集的加载、图像路径配置及定位指标（IoU）计算。

---

## 环境配置

### 安装 VLMEvalKit

参考 [VLMEvalKit/README.md](VLMEvalKit/README.md) 完成基础环境安装：

```bash
cd VLMEvalKit
pip install -e .
```

### 安装支持 Qwen3-VL 的 transformers

```bash
pip install git+https://github.com/huggingface/transformers
```

---

## 数据集配置

1. 将 RefDrone 数据集放置到 `LMUData` 目录下
2. 在 [image_grounding.py](VLMEvalKit/vlmeval/dataset/image_grounding.py) 中修改数据集路径为实际路径

---

## 运行评测

```bash
bash VLMEvalKit/eval_qwen3vl.sh
```

脚本默认使用 4 张 GPU，评测 `Qwen3-VL-8B-Instruct_Grounding` 模型在 `refdrone_test` 数据集上的表现。也可手动执行：


评测结果保存在 `VLMEvalKit/outputs/` 目录下。
