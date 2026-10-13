# 任务向量的格式、分析和使用

任务向量是在冻结模型之外训练的轻量表示。它不能代替完整模型：在另一个运行环境使用时，仍需提供内容相同的 Qwen3-1.7B 权重、配置和 tokenizer。

## 两种表示

| 表示 | 保存内容 | 模型中的作用 |
|---|---|---|
| LTV | 每个指定 Transformer 层的一维 FP32 张量 | 加到该 block 输入、RMSNorm 之前的原 prompt 最后有效 token，仅在 prefill 生效 |
| LoDA-MT | 每个 Q/K/V 投影的一组低秩因子 `B` 与 `A` | 通过 FP32 旁路实现实际参数更新 `sum(B @ A)`，保持原模型的 GQA、QKnorm 和 RoPE |

这两种文件都不是标准 PEFT adapter。LoDA 的 General/Isolated 是训练时的两个分支，不等同于随后通过几何分解研究的共享/残差成分。

训练目录提供三种导出表示：

- `initial.pt`：未训练的初始化对照。
- `raw.pt`：训练后的更新，尚未执行 LoDA 重标定。
- `post.pt`：主评表示；LoDA 已编码重标定和分支系数，不再额外乘 lambda 或 LoRA alpha/r。

LTV 的 raw/post 仍是同一次训练的表示；三个文件不能算作三次独立训练。

## 文件结构

文件使用普通 Python 容器和 CPU 张量，通过 `torch.load(..., map_location="cpu", weights_only=True)` 读取，不依赖项目自定义 pickle 类。

```python
{
    "schema_version": 1,
    "space": "activation",       # LoDA 为 "parameter"
    "task": "math",
    "seed": 17,
    "metadata": {...},           # 基座指纹、来源、层号、版本等
    "values": {                 # LTV
        "model.layers.14": tensor
    },
    "factors": {                # LoDA；LTV 没有实际因子
        "model.layers.0.self_attn.q_proj": [
            {"B": tensor, "A": tensor},
            {"B": tensor, "A": tensor}
        ]
    }
}
```

以上是结构示意，不是可直接执行的 Python 数据。实际 key、层号、分支数与张量形状以文件内容为准。坐标、任务、seed、基座和训练范围不同的表示不能直接混为同一向量空间。

## 不加载模型的检查与分析

单文件检查只需 PyTorch：

```bash
python portable/use_task_vector.py --inspect /artifacts/post.pt
```

完整来源分析接受包含 `VECTOR_INDEX.json`、`EXPORT_MANIFEST.json`、`BASE_MODEL_CONTRACT.json`、表示文件和训练来源记录的目录或 ZIP。工具核对文件哈希、表示内容摘要与来源一致性：

```bash
python tools/analyze_vectors.py --archive /artifacts/vectors.zip --output /results/analysis
python tools/analyze_vectors.py --archive /artifacts/vectors.zip --output /results/decomposition --save-decomposition
```

每次使用新的输出目录。`--version raw` 或 `--version initial` 可以检查其他版本；`--skip-geometry` 只检查张量和来源。默认分析 `post`，输出 `analysis.json`，包括范数、Gram/余弦、均值和 SVD q=0/1/2。来源中绑定的训练代码与当前实现不同时，报告会明确显示差异。

LoDA 范数和内积包含实际 `sum(BA)` 的分支交叉项，避免把因子的任意缩放误当作参数几何。分解导出属于分析产物；其功能价值仍需在模型上评测。

## 在相同基座上生成

以下演示单个 prompt，假定已安装目标设备可用的 PyTorch 和 Transformers。`BASE_MODEL_CONTRACT.json` 用于逐文件核对基座，不能仅凭模型名称相同就假定权重相同。

```python
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/path/to/multi-task-vector-experiments/portable")
from use_task_vector import load_plain, task_vector_context, verify_base_directory

device = "cuda:0"  # Ascend 使用 "npu:0"
if device.startswith("npu"):
    import torch_npu
    torch.npu.set_device(device)

model_path = "/models/Qwen3-1.7B"
verify_base_directory(model_path, "/artifacts/BASE_MODEL_CONTRACT.json")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True, torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).to(device).eval()

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 12 + 7?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
positions = inputs["attention_mask"].sum(-1) - 1
representation = load_plain("/artifacts/post.pt")
with torch.inference_mode(), task_vector_context(model, representation, positions=positions):
    tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                            use_cache=True, pad_token_id=tokenizer.pad_token_id)
answer = tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
```

LTV 的位置必须对应原 prompt，不能随着新 token 生成不断后移。多样本、padding 和正式评测由实验生成模块处理；上面的简短提问仅用于演示接口，不代表正式任务模板或性能评测。

上下文退出时恢复模型原状态。更换硬件或推理 batch 后，应重新完成冒烟检查与评测；完成表示的迁移和后续实验见 [恢复说明](RESUME.md)。
