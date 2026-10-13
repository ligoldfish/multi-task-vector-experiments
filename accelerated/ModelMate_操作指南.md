**Multi-Task Vector：从创建 Notebook 到提交 ModelMate 训练作业**

编写日期：2026-10-10（北京时间）。适用代码为本次交付的 Qwen3-1.7B / LTV / LoDA-MT 实验。用户指定镜像包含 CANN 8.5.0-a3、PyTorch / torch_npu 2.8.0；八张 Ascend 910B 的实际可见数量和每卡容量须在训练容器检测。本指南和配套入口尚未在你的 ModelMate 实机运行，不把本地静态检查写成线上验证。

推荐顺序是：创建 Notebook → 上传本次修订源码包 → 探测真实环境和资产目录 → 创建持久化依赖环境 → 填写资产映射 → 一次准备七任务 → 提交工程验收 `check` → 提交完整实验 `full`。

当前 `full` 会先保存严格首轮256条面板和主研究的全部 `test` 面板，再对六个 scope、层消融及顺序对照执行正式扩评。`full_formal` 是 `full` 的兼容别名，功能完全相同；正常使用只需 `check`、`full` 两个入口。`formal` 用于主阶段已完整完成后的纯评测恢复，不重新训练、选点或运行 smoke。以下描述针对本次修订代码；旧包中缺少来源签名的产物必须保留，并使用新的输出根运行，不能直接按本指南复用。

**1．创建 Notebook，并打开正确的终端**

登录你已有的 ModelMate 课题，在“模型开发 → Notebook”中新建实例。选择你提供的 Ascend / PyTorch 2.8 镜像，以及能够访问现有模型与数据的同一课题和 DPC 存储。Notebook 可先使用 NPU 调试规格；它不需要提前占用训练作业的八张卡。引擎须在创建 Notebook 时选对，平台训练页通常从 Notebook 继承引擎。[东南大学官方操作指南](https://nic.seu.edu.cn/fwzn/gcslfw/czzn_ksrm.htm)

启动后点击 Notebook 名称进入 JupyterLab。以下 Bash 命令都粘贴到 **Terminal**，不是 Python 单元格，也不是本地 Windows PowerShell。只有代码包校验的 PowerShell 命令在本地运行。

在左侧文件面板选择准备上传代码的目录，在该目录打开终端，然后运行：

```bash
pwd -P
command -v python
python --version
npu-smi info
df -hT /opt/dpcvol/datasets
findmnt -T /opt/dpcvol/datasets
```

以 `pwd -P` 的真实结果作为本次代码上传目录，不猜 `/home/work`、`$HOME` 或其他人的 APP 路径。模型、原始数据、语义模型、依赖环境、划分、checkpoint 和运行结果均使用 `/opt/dpcvol/datasets/` 下的明确目录。Notebook 工作目录保留小型源码和压缩包；平台作业读取该 Notebook 的代码快照。官方说明也区分了代码工作区与 DPC 大文件区域。[官方目录说明](https://nic.seu.edu.cn/fwzn/gcslfw/czzn_ksrm.htm)

Notebook 看见一张调试卡，不代表正式作业只有一张；Notebook 的显存也不代表八卡训练容器的显存。

**2．上传配套代码包并校验解压**

本指南对应 `task-vector-modelmate-repaired-20261010.zip`。它包含修订后的实验代码、测试、原始设计与补充协议，以及探测、配置和四种作业入口；实际文件数量和哈希以包内 `PACKAGE_MANIFEST.json` 为准。包内直接是 `src/`、`configs/`、`deploy/`、`ops/`，没有额外的 `task-vector/` 外层目录；不包含正式模型、数据或 Windows 虚拟环境。

把以下文件上传到刚才的 JupyterLab 文件目录：

- `task-vector-modelmate-repaired-20261010.zip`
- `task-vector-modelmate-repaired-20261010.zip.sha256`

可先在本地 PowerShell 检查你上传的实际文件：

```powershell
Get-FileHash -LiteralPath "你下载的完整路径\task-vector-modelmate-repaired-20261010.zip" -Algorithm SHA256
```

回到 Notebook Terminal，确认当前目录就是上传目录，然后执行：

```bash
sha256sum -c task-vector-modelmate-repaired-20261010.zip.sha256
export TV_UPLOAD_DIR="$(pwd -P)"
export TV_CODE_DIR="$TV_UPLOAD_DIR/task-vector-code-20261010"
```

只有校验显示 `OK` 才解压。下面创建新目录，拒绝覆盖已有源码；如果名称已存在，换一个新的目录名称再执行。

```bash
python - <<'PY'
import hashlib, json, os, stat, zipfile
from pathlib import Path, PurePosixPath

source = Path(os.environ['TV_UPLOAD_DIR']) / 'task-vector-modelmate-repaired-20261010.zip'
target = Path(os.environ['TV_CODE_DIR'])
if target.exists():
    raise SystemExit(f'目标已存在，不覆盖：{target}')
with zipfile.ZipFile(source) as archive:
    for item in archive.infolist():
        relative = PurePosixPath(item.filename)
        if relative.is_absolute() or '..' in relative.parts or '\\' in item.filename or stat.S_ISLNK(item.external_attr >> 16):
            raise SystemExit(f'归档路径不合规：{item.filename}')
    archive.extractall(target)
manifest = json.loads((target / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
for relative, expected in manifest['files'].items():
    actual = hashlib.sha256((target / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'文件校验失败：{relative}')
print('CODE_DIR=', target)
print('Verified files:', len(manifest['files']))
PY
cd "$TV_CODE_DIR"
```

此后新终端需要重新 `cd` 到这次实际的项目目录。后文中的 `TV_CODE_DIR` 是便利变量，平台入口会从自身文件位置寻找当前作业源码，不依赖 Notebook 原来的绝对代码路径。

**3．先用只读探测获取缺失信息**

在项目根目录运行：

```bash
python -B ops/notebook_probe.py --assets-root /opt/dpcvol/datasets --max-depth 5 --npu-memory
```

这一步只输出诊断 JSON：当前 Python、已安装框架版本、NPU 可见情况与容量、DPC 挂载、候选 CANN 激活脚本，以及限定深度内的模型和数据目录线索。不读取数据样本、不下载资产、不创建正式结果，也不会替你猜出最终数据映射。`--npu-memory` 会在独立子进程导入 torch/torch_npu 并查询容量；去掉它则默认不导入框架，仅记录包版本和 `npu-smi`。

重点记录：

| 信息 | 如何使用 |
|---|---|
| Python executable / version | 选用确实带有 torch 与 torch_npu 2.8 的解释器创建环境；Python 至少 3.10 |
| torch / torch-npu 与 NPU import error | 两者应为 2.8 系列；导入失败先处理镜像或 CANN 激活 |
| Notebook device_count | 只说明 Notebook 资源，正式八卡另在作业检测 |
| CANN activation candidates | 只有确认真实存在且适用于本镜像时才填写 `cann_env` |
| Qwen 候选 config | 主模型应是 Qwen3-1.7B，28 层、hidden 2048、16 Q heads、8 KV heads、head_dim 128 |
| dataset_dict.json / state.json / train/test 文件 | 确认数据格式与官方 split，不能只凭目录名判断来源 |

如果模型或数据层级更深，可对已确认的子目录重复探测；不要无限扫描整个 DPC。可以把此步骤的 JSON 输出发给我，再据此填写准确配置。输出不包含完整环境变量或凭据。

**4．在 DPC 创建本实验的依赖环境**

采用独立虚拟环境，并继承镜像里的 Ascend 框架包。这样项目固定 Transformers 4.51.3 时，不修改镜像原环境供 vLLM/verl 使用的依赖。本实验的训练和生成均走 Transformers，不运行 vLLM 或 verl。

在当前终端先设置两个新路径：

```bash
export TV_RUNTIME_DIR=/opt/dpcvol/datasets/task-vector/runtime/py28-transformers4513
export TV_RUN_ROOT=/opt/dpcvol/datasets/task-vector/experiments/exp-20261010-01
export TV_CANN_ENV=""
```

`TV_RUN_ROOT` 是本次独立实验根，不要指向以前已经准备、训练或锁定的实验。`TV_CANN_ENV` 留空表示镜像已激活且 torch_npu 能正常使用；如果探测确认需要激活，把它设置成实际 `set_env.sh` 绝对路径。

下面的 `python` 必须是步骤 3 中确认带 torch/torch_npu 2.8 的解释器；若不同，用其真实绝对路径替换。`--copies` 保证持久化环境中的解释器不是指向系统目录的符号链接。

```bash
bash <<'BASH'
set -Eeuo pipefail
: "${TV_CODE_DIR:?请先设置真实代码目录}"
: "${TV_RUNTIME_DIR:?}"
: "${TV_RUN_ROOT:?}"
if [[ -e "$TV_RUNTIME_DIR" ]]; then
  printf '依赖环境已存在，先核对是否可复用，勿覆盖活动环境：%s\n' "$TV_RUNTIME_DIR" >&2
  exit 2
fi
if [[ -n "${TV_CANN_ENV:-}" ]]; then
  test -f "$TV_CANN_ENV"
  set +u
  source "$TV_CANN_ENV"
  set -u
fi
mkdir -p -- "$TV_RUN_ROOT"
python -m venv --system-site-packages --copies "$TV_RUNTIME_DIR"
export TV_PYTHON="$TV_RUNTIME_DIR/bin/python"
export TV_PERSISTENT_ROOT="$TV_RUN_ROOT"
cd "$TV_CODE_DIR"
bash deploy/setup_modelmate.sh
BASH
```

安装脚本校验并约束原有 torch/torch_npu 版本，安装项目附加依赖，保存安装前后框架版本、constraints 和 pip freeze。证据位于本次 `TV_RUN_ROOT/environment/`。安装需要能访问可用的 Python 包源；若平台不联网，应准备对应 Linux/Python 版本的完整依赖 wheelhouse，并在当前环境设置 `PIP_NO_INDEX=1` 与准确 `PIP_FIND_LINKS` 后重做安装。不要上传 Windows wheel，也不要用 CPU torch 替换 NPU 框架。

若 venv 已创建而依赖安装中断，保留该目录。确认没有作业使用它、修复包源问题并重新设置上述变量后，只重试安装，不重复创建 venv：

```bash
cd "$TV_CODE_DIR"
TV_PYTHON="$TV_RUNTIME_DIR/bin/python" TV_PERSISTENT_ROOT="$TV_RUN_ROOT" TV_CANN_ENV="${TV_CANN_ENV:-}" \
  bash deploy/setup_modelmate.sh
```

如果 Notebook 是纯 CPU 规格、NPU 不可见，当前安装脚本会主动停止。应先核对 Notebook 规格和镜像，而不是跳过检查。CANN 在子 Bash 中激活不会改变父终端；作业配套入口会在自己的子 Bash 中再次激活。

**5．填写模型、数据和运行路径**

复制模板，然后用 JupyterLab 编辑器填写真实路径：

```bash
cd "$TV_CODE_DIR"
cp -n ops/paths.example.json ops/paths.local.json
```

模板中的占位路径必须全部替换。字段含义：

| 字段 | 需要填写的内容 |
|---|---|
| `model_path` | 完整 Qwen3-1.7B 本地目录，包括 config、tokenizer 和所有权重分片 |
| `semantic_model_path` | 完整 all-MiniLM-L6-v2 本地目录，用于 QQP 的语义保持评分 |
| `persistent_root` | 与刚才 `TV_RUN_ROOT` 完全一致的独立实验根；先 mkdir 创建 |
| `runtime_python` | 刚才持久化环境的 `bin/python` 绝对路径 |
| `cann_env` | 已确认的 CANN 激活脚本，已激活时用 JSON `null` |
| `expected_devices` | 正式作业使用 8；不要因 Notebook 只有一张卡而改成 1 |
| `data_sources` | 七个任务键，各自的本地 `path` 或准确官方 `files` 映射 |

六个训练任务共用五种原始数据集（MBPP由两个任务共用），另需未见任务 SAMSum 和一个语义模型：

| 配置任务键 | 原始数据 | 必需官方划分 | 注意 |
|---|---|---|---|
| `math` | GSM8K main | train / test | 训练回答保留推理过程和最终答案 |
| `logic` | ProofWriter OWA，深度 0–3 | train / test | 使用 theory 分组；避免把 CWA 或更深任务混入 |
| `translation` | IWSLT2017 en→de | train / test | 原始 en/de+XML 文件或正确保存的 DatasetDict |
| `paraphrase` | QQP | train / validation | validation 是有标签的官方评测来源，不能使用无标签 test 冒充 |
| `code_generation` | MBPP full | train / test | 与代码修复指向同一份完整官方数据即可 |
| `code_repair` | 同一 MBPP full | train / test | 程序自动构造且验证可执行错误，不需你自造一份混合测试集 |
| `summary` | SAMSum | train / test | 科学流程只用于未见任务评测；当前准备器仍需要两种官方划分 |

这里七个任务键对应 **六种原始数据集**：GSM8K、ProofWriter、IWSLT、QQP、MBPP、SAMSum。NLL 是用各任务 tune 集计算的回答负对数似然，不是另一个需要下载的数据集。

数据可采用：Hugging Face `DatasetDict.save_to_disk` 目录，带官方 split 的 JSON/JSONL/CSV/TSV/Parquet，或程序支持的 ProofWriter / IWSLT 原始目录。只有一个 `state.json` 的单独 HF Dataset 必须声明其官方 `split`，并不足以同时提供 train 与 test。多个单 split 目录可先在独立资产准备步骤保存成完整 DatasetDict；不得重新切分来冒充官方测试。

也可在模板中把某个任务换成明确文件映射，例如：

```json
"logic": {
  "files": {
    "train": [
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-0/meta-train.jsonl",
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-1/meta-train.jsonl",
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-2/meta-train.jsonl",
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-3/meta-train.jsonl"
    ],
    "test": [
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-0/meta-test.jsonl",
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-1/meta-test.jsonl",
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-2/meta-test.jsonl",
      "/opt/dpcvol/datasets/实际ProofWriter目录/OWA/depth-3/meta-test.jsonl"
    ]
  }
}
```

上述文件名只是格式示例，以探测和实际官方文件为准。其余键仍保持完整七任务。QQP 文件映射使用 `train` 和 `validation`。裸 Hugging Face 下载缓存目录不一定是 `save_to_disk` 目录，不能见到 `.arrow` 就直接当完整数据源。

填写后生成运行 YAML 和作业环境设置：

```bash
"$TV_RUNTIME_DIR/bin/python" ops/configure_local.py \
  --paths ops/paths.local.json \
  --config configs/modelmate.local.yaml \
  --settings ops/settings.local.json
```

工具保留原配置的科学超参数，只绑定资产和运行路径；所有资产以及 runtime Python 解析后都必须在 `/opt/dpcvol/datasets/` 内并真实存在，CANN 脚本允许系统路径。它拒绝覆盖已有配置/设置以及已经有数据 manifest 的实验。若填错且尚未准备，先保留旧配置并用一个新的项目目录/配置方案重新生成；不要覆盖已经锁定的证据。

在 JupyterLab 中确认已生成 `configs/modelmate.local.yaml` 和 `ops/settings.local.json`。后续四种作业入口固定使用这两份文件，网页不用重复填写模型路径。

**6．在 Notebook 一次准备完整七任务**

以下操作使用真实 tokenizer 和原始数据，不把正式模型加载到 NPU。准备可能包含 MBPP 程序验证，需要时间。使用同一个独立实验根；不自动下载缺失数据。

```bash
bash <<'BASH'
set -Eeuo pipefail
: "${TV_CODE_DIR:?}"
: "${TV_RUNTIME_DIR:?}"
: "${TV_RUN_ROOT:?}"
cd "$TV_CODE_DIR"
unset TV_MODEL_PATH TV_SEMANTIC_MODEL_PATH TV_PERSISTENT_ROOT
export PYTHONPATH="$TV_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
CFG="$TV_CODE_DIR/configs/modelmate.local.yaml"
PY="$TV_RUNTIME_DIR/bin/python"
test -f "$CFG"
CANN_ACTIVATION="$("$PY" -c 'import json; print(json.load(open("ops/settings.local.json"))["cann_env"] or "")')"
if [[ -n "$CANN_ACTIVATION" ]]; then
  set +u
  source "$CANN_ACTIVATION"
  set -u
fi
if [[ -f "$TV_RUN_ROOT/data/manifest.json" ]]; then
  printf '已存在数据 manifest，本块不重复 prepare。请核对旧清单后进入作业验收。\n' >&2
  exit 2
fi
"$PY" -m task_vector.cli assets --config "$CFG" --hash-weights
"$PY" -m task_vector.cli prepare --config "$CFG" --scope all
"$PY" -m task_vector.cli inventory --config "$CFG"
BASH
```

查看真实准备数量：

```bash
"$TV_RUNTIME_DIR/bin/python" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ['TV_RUN_ROOT']) / 'data' / 'manifest.json'
manifest = json.loads(path.read_text(encoding='utf-8'))
for task, record in manifest['tasks'].items():
    print(task, {name: item['count'] for name, item in record['files'].items()})
PY
```

应包含 `math, logic, translation, paraphrase, code_generation, code_repair, summary`。四个主任务为 train512/tune128/discovery128/test256；MBPP 两任务为 300/37/37/256，并按同一原程序 ID 联合划分。SAMSum 当前准备器也会从 train 物化统一划分，但主科学流程不训练或用它选点。`test_formal` 是最多1000条或可用官方评测面板，MBPP 不保证有1000条；实际数量以 manifest 为准。

`prepare --scope core` 仅四任务；`mvp` 还包含 SAMSum；本次完整研究用 `all`。不要在四任务运行锁定后再扩充同一 manifest。已存在 manifest 时 CLI 会拒绝重建。

注意，原 `assets` 命令只生成资产清单，不会补下载，并且缺资产时也可能退出0。合法 `source.files` 映射在旧资产清单里可能显示 missing，因此不能单看其 `all_scope_ready` 判断成败；要结合明确文件映射和成功生成的准备清单。配套作业入口对此增加了检查。

如果 MBPP 的长度过滤、去重或可验证变异导致无法凑满 300/37/37，程序会严格停止。保留 `capacity_preflight.json` 和 `rejected.jsonl` 后诊断；不能混入 test、减少任务、截断回答或悄悄降低样本数。

**7．确认作业快照，再提交工程验收作业**

先核对本项目代码目录中存在以下启动文件：

```bash
ls -l run_modelmate_check.py run_modelmate_full_formal.py run_modelmate_full.py run_modelmate_formal.py
ls -l configs/modelmate.local.yaml ops/settings.local.json
```

在 Notebook 列表找到本实例，点击“训练”进入作业配置。官方文档列出了启动文件、Worker、资源池和触发方式等配置；实际标签和可选规格以你当前页面为准，不按历史坐标操作。[官方训练作业说明](https://nic.seu.edu.cn/fwzn/gcslfw/czzn_ksrm.htm)

首次工程验收填写：

| 字段 | 本次填写 |
|---|---|
| 作业名 | 例如 `task-vector-exp20261010-01-check`，与实验根形成明确映射 |
| 启动文件 | 本次项目目录内的 `run_modelmate_check.py` |
| 引擎 | 核对继承的是已准备的 CANN8.5 / torch2.8 / torch_npu2.8 镜像 |
| Worker | 1 个节点 |
| 每节点规格 | 实际可用的 8×Ascend910B，核对配额；不要将8张卡填成8个节点 |
| PS | 本项目不用 Parameter Server；不配置/0（按页面允许值） |
| 触发方式 | 人工 |
| 运行参数 | 留空，入口已固定 check；无需再拼 Bash 命令 |
| 环境变量 | 路径已在配置/设置中，无需额外 TV_MODEL_PATH 等覆盖变量 |
| 训练数据集/检查点路径 | 源码直接读取 DPC，不需要通过此字段另复制数据；页面允许时留空 |
| 重试 | 首次使用单次执行，避免平台自动重试与人工重提同时写同一实验 |

如果页面强制要求绑定数据集或检查点字段，不填一个无关数据集凑数；需要确认该平台字段实际挂载含义。不能通过 Notebook 命令获取网页配额、当前排队状态或按钮是否提交成功，这些必须看平台页面。

提交前确认同一 `persistent_root` 没有 RUNNING/QUEUED 的验收、完整或扩评作业。只提交一次，记录作业ID；若页面响应延迟，先查作业列表，不重复点击。

提交前保存 JupyterLab 中所有配置和源码文件。训练使用提交时的代码快照；之后修改 Notebook 源码不会更新已提交的作业。DPC 中的模型、数据和依赖环境仍是共享文件，作业运行及协议锁定后均不要原地改动。

验收入口在实际训练容器执行：激活设置中的 CANN → 用 DPC 依赖环境启动当前代码快照 → 核验 torch/torch_npu2.8 与至少8张可见卡 → 检查七任务清单和本地资产 → 记录 preflight/assets/inventory → 执行真实 NPU smoke。LTV/训推检查使用2条训练样本；LoDA真实GAO检查从当前任务train取16条（8+8工程校准样本），执行1个逻辑步、2次AdamW更新，并从terminal checkpoint恢复后确认额外更新数为0。这些仅是工程校准资产，不替代正式科学统计集合S_t，也不读取tune/test。验收不启动全矩阵训练或最终测试。

在日志中检查实际8卡及每卡容量，不能把用户报告的“八卡共64G”自动解释为每卡8G或每卡64G。smoke只用一张卡；显存是否足够由实际峰值和报错确定。

成功后重点查看：

```text
<persistent_root>/runs/smoke.json
<persistent_root>/runs/preflight.json
<persistent_root>/runs/platform_jobs/check-*/
<persistent_root>/runs/assets.json
```

配套入口会打印 `TASK_VECTOR_PLATFORM_JOB_DIR=...`，对应目录中的 `result.json` 必须有 `completed: true`、`status: completed` 和 `exit_code: 0`；失败时另有 `failure.json`。该目录也保存 metadata、context/preflight、assets、prepared_assets、smoke_validation 等证据。smoke需 `passed=true`，且基座、数据、源码、配置与本次运行匹配；已有 smoke 时只验证复用，不覆盖失败或不一致的旧记录。

**8．验收通过后提交完整实验，保存主测试并完成全部正式扩评**

在平台确认 check 已终态并通过，再从同一 Notebook 提交：

| 字段 | 填写 |
|---|---|
| 作业名 | `task-vector-exp20261010-01-full` |
| 启动文件 | 同一项目内的 `run_modelmate_full.py` |
| Worker / NPU / 镜像 | 与 check 一致：单机8卡、同镜像、同 DPC |
| 参数 | 留空 |
| 配置/输出根 | 使用同一份冻结前确定的 YAML、同一 persistent_root |

完整入口先校验通过的 smoke，再核验四主任务×LTV/LoDA两路线、seed17的8个训练导出。全部完成且请求签名、模型来源和每个导出哈希均一致时，保存 `pool_reuse.json` 并跳过训练池；否则调用 `deploy/launch_pool.py`，让8个独立实例分别使用8张卡，已完成任务仍须验证复用。池完成后，调用 `all --scope mvp --split test_formal`；修订后的工作流先完成 `test` 主阶段，再执行纯评测的正式扩评。训练、checkpoint和预测只有在配置、源码、模型、数据及相应输入签名一致时才会复用。

```text
seed17严格首轮：四任务×256条×（基座+四个LTV+四个LoDA）=9216条记录
→ 首轮有效性门控；未通过时保存负结果并停止依赖阶段
→ 种子29/43的核心训练与有效性/跨seed一致性门控
→ 发现集选择秩与任务组
→ 锁定配置、数据和源码
→ 256条主测试下的 E1–E5 与摘要补充
→ 四个主任务轮流留出的 E6
→ LTV层/多层消融、四种顺序LoDA对照
→ 六任务扩展及对应发现、锁定、最终评测
→ 六个scope、全部层消融及顺序对照的test_formal扩评（不重新训练或选择）
```

主阶段各 scope 完成标志为 `test-completed.json`，正式扩评为 `test_formal-completed.json`；控制实验另有 `analysis/controls/` 下对应 split 的完成标志。首轮独立证据是 `analysis/mvp/first-round-completed.json`，必须核对它的9216条状态/任务/ID记录及逐样本文件。初始化、随机对照和其他实验另行记录，不计入这9216条。完整入口包含全部研究范围，不能把整套作业总生成量冒充首轮计数。

八卡只并行第一批独立训练实例；原有后续流程仍以单卡顺序执行。**不要把它描述成整个研究始终八卡并行**，也不要用 `torchrun --nproc_per_node=8` 或 DDP 改写每个实例。所有实例仍保持 BF16、micro2、effective16、回答长度与原实验一致。

如果不使用配套入口，对应核心命令是：

```bash
python deploy/launch_pool.py --config configs/modelmate.local.yaml \
  --scope mvp --seed 17 --devices 0,1,2,3,4,5,6,7
python -m task_vector.cli run --config configs/modelmate.local.yaml \
  --phase all --scope mvp --split test_formal --device npu:0
```

这两条只应在已分配的训练容器和已激活依赖环境里运行，不是让 Notebook 替平台提交作业。配套入口还处理退出状态与证据；不建议直接把两条命令粘贴到网页“运行参数”字段。

科学门控未通过会保存负结果并停止依赖阶段。配套入口在 `completed` 非 true 时返回非零（门控停止为42），使平台不会把这个状态简单当作完整成功。**这不是应该通过降低门控阈值来消除的工程错误。**

**9．正式扩评中断后的纯评测恢复**

正常完成 `full` 后不必再提交 `formal`。如果主研究及全部控制训练已完整完成，而正式扩评中断，可在旧作业确认终态、没有同实验活动 writer 后，提交同一项目中的 `run_modelmate_formal.py`。作业名可用 `task-vector-exp20261010-01-formal`，其他资源和参数同上。必须保持同一模型、配置、源码、数据、发现方案和锁定系数。

该入口要求现有 smoke 有效，在任何正式扩评前核对 MVP、六任务扩展、四个留出折的锁定、主测试完成记录，以及主训练和控制实验的完成/来源证据；不新建或重跑 smoke。随后调用工作流的 `formal` 阶段，对以下六个 scope 和全部层消融、顺序对照执行 `test_formal` 评测：

```text
mvp
expanded
loo-math
loo-logic
loo-translation
loo-paraphrase
```

它不训练，不读取训练数据重新计算统计，不重新选择秩、任务组、checkpoint或组合权重。顺序对照从已验证的阶段 checkpoint 恢复实际合并模型，并把新 split 的预测、指标和报告保存到独立的 `evaluation/test_formal/` 目录；主面板证据仍保留。来源相同且完整的已生成预测可以复用，不能伪造新的生成次数。

正式面板是最多1000条或可用官方评测样本；MBPP 的上限由官方数据决定。它可包含原256条，是同方案的扩评，不能称为完全独立的新确认集；查看主测试后如果修改方法，应另建运行并使用未解封的独立确认面板。

`run_modelmate_full_formal.py` 仅保留兼容名称，和 `run_modelmate_full.py` 调用完全相同的完整流程，都保存256条主阶段再做全部正式扩评。不要把两个入口当成两项必须连续提交的作业，也不要在同一输出根并发提交它们。缺少可验证签名或阶段完成标志的旧产物不能用 `formal` 补成合格证据；应使用新输出根。

**10．查看运行证据与判断完成**

所有权威结果都在配置的 `persistent_root`，不是 Notebook 临时工作目录。常用位置：

| 位置 | 用途 |
|---|---|
| `environment/` | 依赖安装约束和版本记录 |
| `data/manifest.json`、`data/<task>/` | 官方来源、分组、固定ID、数量、长度和拒绝原因 |
| `runs/platform_jobs/` | 每次平台入口的实际环境、metadata、退出结果与失败证据 |
| `scheduler/` | 初始八卡池的调度状态、子进程日志、PID与退出码 |
| `runs/training/` | 最佳与最后checkpoint、NLL、raw/post、恢复状态 |
| `runs/statistics/` | 冻结基座的任务统计 |
| `runs/analysis/<scope>/gate.json` | E1及功能关系复现门控 |
| `runs/protocol/<scope>.lock.json` | 冻结协议与源码/资产指纹 |
| `runs/analysis/mvp/first-round-completed.json` | seed17的9个核心状态、9216条首轮记录及逐文件哈希 |
| `runs/analysis/<scope>/test-completed.json` | 各scope的256条主测试完成记录 |
| `runs/analysis/<scope>/test_formal-completed.json` | 各scope的正式扩评完成记录 |
| `runs/analysis/controls/<split>-completed.json` | 对应split的层消融与顺序对照完成记录 |
| `runs/evaluation/` | 逐样本生成、指标、截断和代码测试证据 |
| `runs/sequential/` | 顺序LoDA对照，独立于LoDA-MT |
| `runs/report.md`、`runs/report.json` | 从真实产物生成的汇总 |

需要从 Notebook 重建报告时，使用原配置和持久化依赖环境，并清除可能覆盖本配置的旧 TV 环境变量：

```bash
bash <<'BASH'
set -Eeuo pipefail
cd "$TV_CODE_DIR"
unset TV_MODEL_PATH TV_SEMANTIC_MODEL_PATH TV_PERSISTENT_ROOT
export PYTHONPATH="$TV_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
CANN_ACTIVATION="$("$TV_RUNTIME_DIR/bin/python" -c 'import json; print(json.load(open("ops/settings.local.json"))["cann_env"] or "")')"
if [[ -n "$CANN_ACTIVATION" ]]; then
  set +u
  source "$CANN_ACTIVATION"
  set -u
fi
"$TV_RUNTIME_DIR/bin/python" -m task_vector.cli report \
  --config "$TV_CODE_DIR/configs/modelmate.local.yaml"
BASH
```

平台 RUNNING/成功、子进程结束、产物完整和科学效果成立是不同状态。修订后的 CLI 与配套入口会对明确的门控停止返回非零，但仍必须查看 scope、控制实验完成记录、门控和报告，不能用目录数量或显存占用推断完成。联合任务若只形成一个依赖连通分量，其退化 bootstrap 区间不能作为独立统计证据。

根目录的 `runs/report.json` 汇总独立与顺序训练、门控、完成面板、首轮计数及发现关系确认等记录；每次平台作业目录下的 `report.json` 只是报告路径和计数摘要，二者不能混用。顺序对照须核对 `runs/sequential/seed*/<order>/order-completed.json`、`stage*/stage-completed.json` 以及 `evaluation/test/report.json`、`evaluation/test_formal/report.json`：三个seed×四种顺序，共12条顺序、48个训练阶段，评测按split独立保存。逐项核查实际样本、指标、阶段父合并哈希和完成签名，不能只按文件数量判断完成。

**11．失败和恢复时怎么做**

- **依赖安装失败**：保留 environment 日志，核对 Python 包源、镜像、torch/torch_npu2.8和 CANN；不自动升级框架。活动作业使用的持久化依赖环境不能重新安装。
- **资产缺失/不完整**：回传探测输出、配置中的准确路径和 assets 明细；补齐真正的模型分片、tokenizer、官方数据或语义模型，不伪造占位资产。
- **数据容量不足**：保留 capacity/rejected 证据；先解释过滤、分组和来源，不用测试补训练、不改样本数量凑成功。
- **单卡 OOM**：回传实际训练容器 preflight、smoke或failure、峰值显存和错误栈；保持既定模型、batch与回答长度，不擅自降级实验。
- **科学门控停止**：查看 gate 与预测/对照；这是研究结果或方法问题，不能直接当临时故障无限重试。
- **主阶段或控制训练中断**：平台确认旧作业已终态、没有同实验活动 writer 后，可用本次修订版的相同代码、配置、模型、数据与结果根重提 `full`。独立任务会验证并复用完成导出，或从各自 `last.pt` 恢复；顺序任务会核验父阶段、输入、模型与源码签名，恢复已提交的实际合并权重，再继续未完成阶段。不要把某一任务的 `resume_from` 全局填给所有任务。
- **只剩正式扩评未完成**：优先使用步骤9的 `formal`，要求六个主阶段和全部控制训练已经完整提交；它不会为缺失 checkpoint 补训练。`full_formal` 与 `full` 同功能，不是另一个恢复阶段。完整作业已完成时无需重复提交；相同来源的已完成结果可验证复用，不因“目录已经存在”而直接拒绝。
- **旧产物没有签名、来源不一致或导出损坏**：程序会拒绝复用；保留旧结果和失败证据，使用新的 `persistent_root`。不能覆盖、补写签名或删除锁来让旧产物通过。
- **锁冲突**：同一实验根通过 `locks/platform-writer.lock` 防止平台并发写入，初始池另用 `pool-mvp-seed17.lock`；两者均为OS锁，并有独立 `*.owner.json` 诊断归属。Linux保留锁文件；进程退出后OS释放持锁状态，文件存在本身不等于仍被占用，不以删除文件解锁。池异常时会先终止并等待已启动子进程、关闭日志，再释放池锁；若父进程被SIGKILL而子进程仍活，各任务目录的 `training.lock` 继续保护其写入。恢复前仍需查平台活动作业、owner和子进程日志，不能并发重提。
- **已存在 manifest/smoke**：不重复 prepare、不手工覆盖 smoke；配套入口会验证现有smoke。配置/数据/源码不一致需要新实验方案，而非清空旧结果。
- **网页提交超时**：先查看训练列表是否已产生作业ID，不重复点击。

本轮生成修订代码、指南、配套工具和本地检查记录。尚未替你创建 Notebook、上传资产、安装远端环境、提交作业或验证实际NPU。需要进一步精确配置时，提供步骤3的探测输出、七任务资产的官方划分格式，以及当前训练页面的实际资源规格即可。
