# 模型、LTV与共同训练实现逐项审计

审计日期：2026-10-10。审计对象为 `D:\DataSpecificWork\task-vector` 当前源码；平台辅助入口在另一交付快照中，本报告仅在说明其保护边界时引用。已全文阅读原始211行方案及34行补充协议，再沿真实调用路径阅读模型、干预、loss、训练、GAO、评估、smoke、runner、workflow和相关测试。原文行号以仓库 `docs/原始实验方案.md` 为准。

本轮没有运行任何模型、训练、测试、安装或删除操作，也没有改动科学源码。测试列中的“有测试”仅表示静态审阅到相应断言；以前的84项测试通过属于历史证据，不代表本轮重新执行。下文“覆盖”表示默认实现与规范在代码层面对应，不表示真实Qwen3/Ascend已验收；BF16、KV cache、优化器恢复、显存及完整生成行为仍需实机证据。

主要结论：默认模型、LTV注入位置、回答损失、FP32主参数、AdamW/GAO、batch、早停和层消融具备实质实现，未发现默认训练目标被改成最后一个token或仅答案末位置等错误。当前存在两项明确恢复缺口、动态表示未被最终协议锁完整绑定、首轮256评测阶段缺失，以及配置保护和NPU验收不足，不能据此宣称全部原文要求已无遗漏完成。

## 发现与影响

### F1 — P1：训练完成记录与表示来源补全不是一次可恢复提交

`training.train_task` 先保存raw/post并在 [training.py:284](D:/DataSpecificWork/task-vector/src/task_vector/training.py:284) 写出 `training.json`；返回后runner才在 [runner.py:151](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:151) 加入 `request_signature` 并再次写JSON，然后在 [runner.py:153](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:153) 至160补齐raw/post/initial的 `base_fingerprint`、训练ID等元数据。缓存分支 [runner.py:130](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:130) 至136一看到JSON就当成已完成。

存在两个确定中断窗口：

- 内层JSON已写、外层签名未写时，重提在132–133因 `request_signature` 缺失而拒绝，虽然 `last.pt` 可恢复，也不会进入146–150恢复路径。
- 外层JSON已写、表示元数据未全部补齐时，重提直接返回已完成结果；随后评估在 [runner.py:201](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:201) 至202拒绝缺少基座指纹的表示。缓存仅查post文件存在，不检验raw/initial及三份元数据完整性。

这不是模型数学错误，但会阻断符合要求的中断恢复、E1初始化对照和保存重载验收。建议最后才原子写“完整提交”标记，在此前将状态区分为训练完成/导出待完成；重提可只恢复导出及来源补全，并逐项核对三份表示，不能让残缺完成记录阻挡 `last.pt`。

### F2 — P1：部分checkpoint没有绑定实际基座、样本内容和代码版本

[training.py:158](D:/DataSpecificWork/task-vector/src/task_vector/training.py:158) 至161的checkpoint签名包含模型配置及路径、训练设置、任务/seed及train/tune的ID；不包含权重字节指纹、prepared manifest内容哈希、样本实际prompt/target、来源代码哈希或LoDA输入子空间内容摘要。`last.pt` 只保存这份弱签名，见195–201。

runner的较强请求签名确实包含manifest内容和source manifest，见 [runner.py:126](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:126) 至129；但只在已有 `training.json` 时核验，未写入正在训练的checkpoint。因此对于尚未生成完成JSON的LTV训练，若模型目录路径不变但权重被替换，或数据保持ID但target变动，或代码变动，恢复可仍通过169的检查，从旧optimizer/RNG继续在新基座或新目标上训练。LoDA旧统计缓存有额外基座检查，不能据此证明所有路线的部分恢复均安全。

新平台helper对既有smoke的匹配可阻止部分来源漂移，但原 [workflow.py:158](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:158) 至170会保留旧smoke并重新执行，之后仍进入旧last.pt恢复；直接CLI train也不经过smoke门。因此不能把smoke当作checkpoint来源校验的替代品。

建议每份checkpoint保存runner级请求签名、实际基座权重指纹、数据内容哈希、源代码哈希和子空间摘要，加载时先核验；原路径/同ID不能作为数据或模型未变的充分证据。

### F3 — P1：最终锁未完整冻结实际输入表示，候选摘要已记录但未核验

最终锁 [config.py:95](D:/DataSpecificWork/task-vector/src/task_vector/config.py:95) 至123冻结配置、数据manifest、discovery decisions和源码。`source_manifest` 的84–91仅遍历src/configs/deploy/third_party及两份指定文件，并不包含训练post/raw或动态composition文件。

E5调参时已在 [runner.py:445](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:445) 至447记录候选 `representation_digest`，但 `composition_final` 在 [runner.py:478](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:478) 只按路径加载，没有与已锁摘要比较。其他主表示的实际digest也没有统一进入lock并于final入口核验。评估在214–217使用“当前”表示digest创建cache签名：已有同label结果时会拒绝过期cache；若这是首次final访问，替换后仍可生成新结果，协议锁本身不会阻止。

建议锁中记录全部被选择和派生表示的实际digest及依赖摘要，在任何test样本读取前核验。特别是E5现有摘要应成为强制前置检查。该问题与父审计的分析/选择锁条目交叉，不重复声称训练公式错误。

### F4 — P1：首轮256核心评测的阶段要求未实现

原文209及R66/R67要求首轮seed17的4个LTV、4个LoDA-MT及基座，对4任务各256条形成9×4×256=9216的核心证据；211又要求验证有效性、重载/训推一致和跨seed迁移后再进入共享、组合、未见任务。

当前 [workflow.py:176](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:176) 至182依次训练seed17/29/43，并调用 `core_panel(scope,route,seed)`；[runner.py:273](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:273) 默认split为discovery。因此早期E1/E3和有效性门使用128条发现集，而非首轮256。`run_phase('all')` 在 [workflow.py:376](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:376) 至386先core、共享/组合等discovery、lock，才进入final256或formal。无论full256还是full-formal，均不能声称已按原先后次序独立生成首轮9216证据。完整formal面板含256子集，也不等于有独立的首轮256阶段记录。

这是编排缺口。建议增加不改变科学超参的首轮专用阶段、样本/预算记录和阶段门，并区分初轮暴露与后续独立确认；不能只把core默认参数改成test，因为当前 [runner.py:203](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:203) 的test评估依赖最终lock，而最终lock又依赖后续发现决策。合并full/formal还需要拆开控制训练与评测、顺序对照的split证据和恢复接口。

### F5 — P2：默认正确，但配置保护不完整；部分显式字段静默无效

[config.py:30](D:/DataSpecificWork/task-vector/src/task_vector/config.py:30) 至69保护训练数值、主要LoDA参数、生成上限等，但没有为数据规模、seed表、主层/消融范围、a_scale、必要variant和任务清单等建立等价的协议偏差检查。命令行override可能在不声明 `supplement_label` 的情况下缩小实验范围。此处不是指默认配置已偏离；默认 [experiment.yaml:10](D:/DataSpecificWork/task-vector/configs/experiment.yaml:10) 至117基本对应方案。

明确静默字段：配置55提供 `ltv.init_low/init_high`；[training.py:135](D:/DataSpecificWork/task-vector/src/task_vector/training.py:135) 未传递它们，[ltv.py:105](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:105) 固定使用U(-.1,.1)。默认正确，但显式修改不会生效，即使补充标签已声明。建议所有暴露字段要么接入并记录有效值，要么明确禁止/拒绝；主协议范围缺省和用户显式更改都应进入可审计偏差报告。

### F6 — P2：跨输出目录的显式恢复缺少初始化表示

`training.resume_from` 可以指向已有checkpoint，但恢复分支 [training.py:167](D:/DataSpecificWork/task-vector/src/task_vector/training.py:167) 至181仅在非恢复分支保存 `initial.pt`。若在新输出目录恢复，273–275仍返回新目录的 `initial_path`，runner153–160随后加载该文件会失败。默认同目录自动恢复不受此问题影响。

建议将初始化表示或其经过校验的来源一并保存在checkpoint，跨目录恢复时导出；或在训练前明确拒绝缺少初始化文件的恢复组合，避免训练结束后才失败并留下F1所述残缺完成状态。

## T01起逐项矩阵

“覆盖/待实机”指静态调用语义已实现，但设备数值和容量尚无本轮执行证据。“部分”用于明确实现或验收缺口。“偏差”指与原要求的默认编排不同。R编号来自现有需求矩阵，不将建议目录结构当成必须存在的文件名。

| 条目与需求映射 | 原文/补充协议 | 真实实现与调用证据 | 判断及测试证据边界 |
|---|---|---|---|
| T01 · R03 | 原文15：Qwen3-1.7B、28层/2048、BF16；补充7：NPU2.8环境 | [modeling.py:10](D:/DataSpecificWork/task-vector/src/task_vector/modeling.py:10)拒绝thinking开启/非BF16；23加载BF16 native模型并放device；25–31检查qwen3、28/2048和Q16/KV8/head_dim128；[runner.py:50](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:50)接入真实加载 | 覆盖/待实机。`expected_config`可覆盖默认检查值，主配置正确；`test_ltv.py:133`模拟错误GQA并断言拒绝，仅CPU/mock。 |
| T02 · R03/R21/R32 | 原文15/51/81；补充15：非thinking模板、空thinking scaffold属于prompt，EOS监督 | [prompts.py:73](D:/DataSpecificWork/task-vector/src/task_vector/prompts.py:73)–79统一chat template并显式enable_thinking=False；81–95编码prompt、target、EOS和labels；modeling17–22要求EOS并设padding | 覆盖/待实机。两路线共同调用encode_sample，未把模板末尾当回答。TinyTokenizer只断言参数，不替代真实Qwen tokenizer内容检查。 |
| T03 · R18 | 原文46–49：仅v；层14整block输入、norm前；2048维 | [training.py:127](D:/DataSpecificWork/task-vector/src/task_vector/training.py:127)–135冻结基座并构造LTV；[ltv.py:98](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:98)–110按hidden_size建参数；74–76注册decoder block forward_pre_hook | 覆盖/待实机。主配置55为[14]；没有挂到RMSNorm输出或block后。`qwen_hooks.py`不存在，但职责实现在ltv.py；原文建议结构不要求字面文件名。 |
| T04 · R19 | 原文50：独立U(-.1,.1)、FP32优化 | [ltv.py:100](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:100)–110 CPU generator逐层独立抽样FP32；[training.py:130](D:/DataSpecificWork/task-vector/src/task_vector/training.py:130)–135用task/seed/layers派生初始化流；147检查优化参数FP32；初始表示181保存 | 主要求覆盖；配置接线部分，见F5。独立task不会因每任务重置同一个seed而得到相同向量。 |
| T05 · R29/R32 | 原文76/81：teacher forcing始终原prompt末位置 | [losses.py:20](D:/DataSpecificWork/task-vector/src/task_vector/losses.py:20)–30记录p-1；[training.py:69](D:/DataSpecificWork/task-vector/src/task_vector/training.py:69)–71先设置原位置再完整序列forward；[ltv.py:52](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:52)–63仅在该位置scatter addition | 覆盖/待实机。`test_ltv.py:25`用变长样本验证block输入仅指定位置改变；smoke102–114比较完整teacher forcing中p-1和prompt-only首答案logits。 |
| T06 · R30 | 原文77：只prefill注入，通过KV cache传播 | [ltv.py:33](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:33)–44按cache_position/已占用cache检测；48–50门控；[evaluation.py:83](D:/DataSpecificWork/task-vector/src/task_vector/evaluation.py:83)–95明确首步prefill、后续单token+past，要求模型返回cache | 覆盖/待实机。`test_ltv.py:45`检查自动cache门控；smoke117–131比较固定原位置无cache与cache前3token。真实NPU长生成及所有EOS路径未本轮运行。 |
| T07 · I01/R29/R30/R65 | 原文207、推论I01：非重入重算仍同一注入 | [training.py:186](D:/DataSpecificWork/task-vector/src/task_vector/training.py:186)–188启用use_reentrant=False；203–204 adapter上下文覆盖训练/所有backward；70 `_set_positions`强制训练prefill；[ltv.py:135](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:135)–143上下文退出卸载 | 覆盖/待实机。`test_ltv.py:60`对比checkpoint开关loss与梯度；当前测试是CPU FP32 tiny Qwen，未证明NPU BF16峰值和重算内核。 |
| T08 · R18/R20/R33 | 原文47/89：同一冻结基座、独立任务，不相互合并 | [modeling.py:32](D:/DataSpecificWork/task-vector/src/task_vector/modeling.py:32)–34冻结；training127–128再次冻结；LTV对象不注册模型为子module；[loda.py:75](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:75)弱引用基座且83–84冻结；MT训练不调用merge | 训练路径覆盖；来源/恢复一致性部分，见F1/F2/F3。`test_training.py:125`及smoke87–89有基座无grad断言；顺序控制另有独立merge路径，不当成MT。 |
| T09 · R20/R36/R40 | 原文46–50/100/108：固定A，仅B_G/B_I；28层QKV，8+8，异宽GQA | [loda.py:25](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:25)–46固定buffer A并按native输出宽建零B；85–97对subspace每层QKV建adapter；[runner.py:139](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:139)–144从所有层统计构建；主配置57–58 rank8+8 | 默认MT路径覆盖/待实机。共同trainer允许sequential首阶段显式训练A_G，MT默认不启用该标记。`test_loda.py:36`检查形状、冻结、B=0与原模型等价，仅tiny异宽线性模型。 |
| T10 · R40/R63 | 原文108/183：保留native GQA/QKnorm/RoPE，不能套ViT注意力实现 | [loda.py:115](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:115)–119只加native projection输出；保存后的 [ltv.py:146](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:146)–169 wrapper调用原Linear并加精确BA；未替换attention类/重写norm和RoPE | 覆盖/待实机。`test_ltv.py:96`用真实tiny Qwen异宽k_proj验证BA wrapper；不是直接复刻ViT attention。 |
| T11 · R21/R32 | 原文51/61–69/81：全部回答token meanCE，样本内后样本间平均 | [losses.py:36](D:/DataSpecificWork/task-vector/src/task_vector/losses.py:36)–70正确shift，只统计非-100回答，逐样本除token数；[training.py:72](D:/DataSpecificWork/task-vector/src/task_vector/training.py:72)–75按group样本数加权立即backward；[evaluation.py:27](D:/DataSpecificWork/task-vector/src/task_vector/evaluation.py:27)–48 tune同定义 | 覆盖。`test_training.py:18`人工异长回答含EOS并区分token加权错误；33验证大vocab分块loss与梯度；50验证micro累积等价。不是最后token loss。 |
| T12 · R21/R26 | 原文59：prompt≤512、target≤512、总长≤1024，不截断答案 | [prompts.py:81](D:/DataSpecificWork/task-vector/src/task_vector/prompts.py:81)–95单独编码、EOS计入target，超长raise而非截断；collate20–25要求完整回答mask；training156–157统一512限额 | trainer层覆盖；池筛选及拒绝计数由数据审计另核。两个独立上限自然保证总长≤1024。joint推理1536扩展不用于单任务训练。 |
| T13 · R22 | 原文52–54：AdamW lr.001，LTV wd.01、LoDA wd0+GAO | [training.py:149](D:/DataSpecificWork/task-vector/src/task_vector/training.py:149)–152分路线设置wd、lr、betas/epsilon和foreach=False；213–232区分LTV单步与LoDA GAO | 默认覆盖/待NPU优化器实测。配置39–53记录有效默认。不能仅凭包版本确认torch_npu实际optimizer行为与FP32moment恢复。 |
| T14 · R24 | 原文56：LTV micro2×累积8=16；LoDA逻辑16分8/8，每micro2 | [training.py:109](D:/DataSpecificWork/task-vector/src/task_vector/training.py:109)–114校验乘积；62–77逐micro立即反传，释放graph；225–229按half分group并callback累积 | 覆盖。LoDA每方向源8/目标8，源组梯度先完整累积，再构建每参数扰动；没有同时保留4个micro计算图。`test_gao.py:59`和test_training50有等价断言。 |
| T15 · R24/R36 | 原文100；补充23：样本不相交、余batch不丢不复制 | [training.py:52](D:/DataSpecificWork/task-vector/src/task_vector/training.py:52)–59重排并将末尾singleton从前batch移一条；225–227检查两组ID不相交；[gao.py:50](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:50)–54检查组内/跨组唯一；训练241–244记录实际batch及IDs | 覆盖。512正好32批；300尾批12→6/6。异常尾1会变成前批15、末批2，样本保留；补充协议已明确该选择。 |
| T16 · R38/R39 | 原文106：rho_max.05，逐参数norm，双向，无二阶 | [gao.py:56](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:56)–67每逻辑batch抽一次负Uniform rho，对完整源梯度逐张量detach/norm；59循环有序两方向；无create_graph/二阶反传；70–85扰动应用及finally撤销 | 覆盖。第二方向在第一AdamW更新后的参数上计算；step时扰动仍在再撤回，对应固定官方源码顺序。`test_gao.py:42`参考实现比较；104不相交保护；84目标失败撤扰动。 |
| T17 · R25/R39 | 原文57：linear decay/warmup0/clip1；GAO外层clip | [training.py:115](D:/DataSpecificWork/task-vector/src/task_vector/training.py:115)拒绝warmup；153–155按LTV1/LoDA2次实际更新计算linear scheduler；LTV217–221 clip/step/scheduler；[gao.py:79](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:79)–86 target梯度后外clip、step、scheduler | 覆盖。GAO内部扰动规范化与outer clip两者独立；LoDA每逻辑batch scheduler2次。test_gao59断言实际scheduler计数。 |
| T18 · R23 | 原文55/58；补充11：最多5epoch、每epoch调参NLL最小、patience2 | [training.py:113](D:/DataSpecificWork/task-vector/src/task_vector/training.py:113)限制1–5epoch；247–263每epoch tune、严格变好才重置bad/保存best；205–207检查patience；266恢复best_state后导出 | 覆盖。停止依据是回答NLL，不是生成最终指标或discovery/test。`epochs.json`和training.json记录NLL、bad_epochs、停止原因。历史测试有2epoch小模型路径，本轮未跑5epoch边界。 |
| T19 · R23/R68 | 原文211：保存/重载一致；工程恢复范围 | [training.py:194](D:/DataSpecificWork/task-vector/src/task_vector/training.py:194)–201保存adapter/optimizer/scheduler/RNG、next_epoch/batch、最佳state及计数；245–246仅完整逻辑batch后保存；258–260保存best/last | 部分：正常同源同目录恢复有实现，F1/F2/F6边界未覆盖。GAO中途方向失败的RAM更新不会写成完成logical-step；重提从上一个checkpoint重做该逻辑batch。 |
| T20 · R68 | 保存RNG和精确续训；单卡任务换卡恢复 | [training.py:18](D:/DataSpecificWork/task-vector/src/task_vector/training.py:18)–49保存/还原Python、TorchCPU、NumPy及当前NPU/CUDA RNG；167–179恢复所有状态；52–53 epoch样本顺序独立确定 | 部分/待NPU。`test_training.py:91`只验证tiny LTV同源中断续训；118只检查CPU RNG预期。没有本轮NPU RNG、GAO第二方向中断或BF16精确续训证据。 |
| T21 · R37/R68 | 原文101–104/211：raw/post实际增量并保存重载 | [loda.py:148](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:148)–183导出.5 B_G Λ A_G+B_I A_I（raw省Λ）；training267–270同时保存；[representations.py:49](D:/DataSpecificWork/task-vector/src/task_vector/representations.py:49)–67存CPUFP32并weights_only加载；runner180–181默认post | 公式与保存覆盖；行为验收部分/待NPU。LoDA live与导出wrapper运算缩放位置存在FP32舍入顺序差异，smoke固定BF16容差并要求argmax相同；不能描述为所有LoDAlogits bitwise相等。 |
| T22 · R19/R20/R65 | BF16基座、FP32优化参数与分支计算 | modeling23 BF16；training147 FP32 master检查；[ltv.py:57](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:57)仅注入时转换BF16；[loda.py:49](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:49)–58及ltv166–168禁用外autocast并用FP32 BA、随后cast输出 | 默认覆盖/待实机。没有实际调用runtime.autocast_context也不构成偏差：主模型权重/输入激活已BF16，adapter计算显式FP32。loss分块FP32 CE降低临时峰值，但模型仍产生完整logits，不能据此保证单卡8GB可容纳最坏长度。 |
| T23 · R27 | 原文72/78：目标指令/输入保持，greedy换源表示 | [evaluation.py:128](D:/DataSpecificWork/task-vector/src/task_vector/evaluation.py:128)–149从目标sample渲染prompt、仅context换表示；greedy98取argmax，不采样；runner282–285所有source→target，primary_strength1 | 覆盖/待完整生成实测。row保存rendered prompt/variant；不以源任务instruction替换目标instruction。 |
| T24 · R28 | 原文72：各任务生成上限、完整输出、截断率 | [evaluation.py:133](D:/DataSpecificWork/task-vector/src/task_vector/evaluation.py:133)–169按task预算、记录完整prediction/tokenids、generated_tokens、max_new_tokens、stopped_by/truncated/reference预算；主配置74原单任务预算完整 | 覆盖/待数据与NPU。math512/logic32/translation256/paraphrase128/code512；summary256/joint1536为已记录补充。GT不被生成上限截断。 |
| T25 · R30/R65 | greedy batch padding及EOS/cache一致 | [evaluation.py:62](D:/DataSpecificWork/task-vector/src/task_vector/evaluation.py:62)–112处理leftpad、position_ids、cache_position、多EOS及已停止row；71–74要求左padding末位置有效；gen141–146人工左padding | 静态覆盖/待实机。`test_ltv.py:115`只用不在vocab的EOS强制3token，证明tiny batch与单条生成一致；不能覆盖真实多EOS提前结束混合batch。主配置generation_batch_size=1降低该路径风险。 |
| T26 · R43/I04 | 原文116：base/正确/随机同范数/未训练初始化 | [runner.py:273](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:273)–298 core读取initial、正确表示和same-norm random；[representations.py:131](D:/DataSpecificWork/task-vector/src/task_vector/representations.py:131)–160实际矩阵rank/norm null；LoDA初始B=0 | 对照功能覆盖；阶段和完成文件完整性部分，见F1/F4。LoDA零初始化等价基座有test_loda36和smoke147–150断言；random不是复用训练A制造伪随机。 |
| T27 · R31 | 原文79：6/14/22各自训练及联合三独立向量 | 主层14配置55；消融配置111=[6]/[22]/[6,14,22]；[workflow.py:333](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:333)–340逐seed/每层配置/每task调用train；[runner.py:83](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:83)–87分目录；ltv101–110逐层独立参数 | 覆盖/待实机。不是把14层已训练v挪到6/22；联合实例有三个向量同时优化。`test_ltv.py:80`验证三个独立向量及save/reload，但未实际训练28层的全部消融。 |
| T28 · R68/I01 | 原文211：重载、训推、cache验证 | [smoke.py:59](D:/DataSpecificWork/task-vector/src/task_vector/smoke.py:59)–135真实两条train样本做LTV一步更新、基座无grad、保存重载exact、首答案logits、cache/no-cache3token和卸载还原 | 有实机检查实现但待实机执行。`test_platform_artifacts.py:47`用FakeRunner+CPU tiny_model覆盖逻辑，不能当成910B执行记录。smoke不是256最终评测。 |
| T29 · R20/R37/R40/R68 | LoDA真实模型hook/导出重载验证 | [smoke.py:136](D:/DataSpecificWork/task-vector/src/task_vector/smoke.py:136)–176在全部native QKV用eye合成A，检查零更新、固定随机非零B的raw/post live与保存wrapper logit/argmax | 部分/待实机。明确未做真实统计、GAO训练、optimizer checkpoint恢复。logits_to_keep=1在right-padded全序列末位比较，不是每例prompt末首答案；未覆盖LoDA完整训练后长生成一致性。 |
| T30 · R65 | 原文207；补充7/28：910B8卡实际显存、nonreentrant、eager | [runtime.py:17](D:/DataSpecificWork/task-vector/src/task_vector/runtime.py:17)–30明确npu加载torch_npu；64–79峰值与容量；82–117记录包版本、卡数/npu-smi/持久化；modeling23默认eager；training190–191 reset peak、282记录memory | 待实机。用户报告“八卡共64G”不推导单卡实际可用显存；原24GB NVIDIA规划已被批准平台覆盖。原preflight遇非2.8只记review，helper额外严格检查，不能把原preflight记录当兼容性通过。 |
| T31 · R66/R67/R68 | 原文209/211：首轮8独立训练+base、4×256及先验证再后续 | workflow176–185调用core默认discovery；runner273默认128发现集；workflow376–386后续顺序见F4 | 偏差/缺失：没有首轮独立9216阶段记录。seed17→29→43门本身存在，但使用发现集证据；full-formal不能声称补齐首轮256先后顺序。 |
| T32 · R32/R60/R63/R64 | 原文81/174–175/183/185–202：官方差异适配与工程模块 | 训练统一lr、full-answer、block前位置；native QKV取代ViT attention；本报告沿runner→training→adapter→evaluation/smoke核实实际调用，不只检索函数名 | 适配功能覆盖。官方文件固定来源的完整性由来源审计另核；不存在qwen_hooks.py不算缺功能。有效配置字段问题仍见F5。 |
| T33 · R11/R23/R68/I10 | 原文33/35/211：tune只选checkpoint，final锁定后使用 | training247仅tune NLL；runner203–207 final先verify_lock及冻结asset；record_final_access237–251记录暴露；test_workflow242/256/273有顺序保护断言 | 部分：配置/数据/源代码/模型及semantic资产冻结已实现，但动态表示缺口F3；原文初轮测试与最终锁时间关系仍需F4专用协议处理。 |
| T34 · R18–R32/R36–R40/R65/R68/I01 | 与模型/训练有关的验收证据总体 | 真实测试路径是tiny_model→Qwen3随机config；CPU FP32；LoDA形状测试另用TinyModel线性层。NPU smoke有实际入口与证据字段，但本轮未执行 | 待实机，不能用历史84 tests通过宣称真实Qwen3 BF16/NPU和科学有效性。需保存真实model/tokenizerrevision、smoke、训练NLL/steps、内存、failure/recovery及完整生成证据。 |

## 现有测试具体证明什么

- [test_ltv.py:14](D:/DataSpecificWork/task-vector/tests/test_ltv.py:14) 创建vocab37、hidden16、3层、Q4/KV2/head_dim4的CPU FP32随机Qwen3；25、45、60、80、96、115分别覆盖指定位置、cache、非重入梯度、三向量重载、BA wrapper和左padding。其结论不等于28层2048维BF16在910B上通过。
- [test_training.py:18](D:/DataSpecificWork/task-vector/tests/test_training.py:18)、33、50覆盖回答平均、分块CE及micro梯度；91的中断续训发生在验证前，属于同配置同源LTV路径；没有F1完成提交窗口、F2换内容、F6跨目录或GAO第二方向中断测试。
- [test_gao.py:42](D:/DataSpecificWork/task-vector/tests/test_gao.py:42) 的两方向参考例子连非零AdamW wd也对齐；59覆盖micro回调及2 scheduler steps；84覆盖目标失败撤扰动；104覆盖样本组相交拒绝。真实LoDA主协议wd仍为0。
- [test_loda.py:36](D:/DataSpecificWork/task-vector/tests/test_loda.py:36)、58、84、103验证异宽形状、冻结、零B、raw/post实际缩放、异常卸hook以及sequential首阶段白名单。`test_training.py:125`连接共同trainer+GAO+LoDA，但仅tiny配置、rank2+2、identity统计。
- [test_platform_artifacts.py:47](D:/DataSpecificWork/task-vector/tests/test_platform_artifacts.py:47) monkeypatch FakeRunner后检查smoke逻辑；[test_workflow.py:133](D:/DataSpecificWork/task-vector/tests/test_workflow.py:133) 检查smoke来源/设备匹配，146检查执行次序，256/273检查asset和base指纹前置拒绝。这些断言没有覆盖F1/F2/F3。

## 需保留的真实平台验收边界

在解决确定缺口后，应在用户指定CANN/torch/torch_npu2.8镜像、真实Qwen3-1.7B/tokenizer和冻结数据上取得以下已有要求的实机证据；这不是新增科学实验或更换超参：

1. default layer14的两条train smoke完整通过并记录910B型号、BF16/FP32 dtype、配置/基座/数据/代码摘要、实际峰值与首答案/cache证据；任何失败保留原因，不悄悄放宽容差。
2. 真实LoDA统计和rank8+8训练至少覆盖一个完整逻辑batch的两方向GAO、每micro释放、外clip及2个optimizer/scheduler steps，并检查全部28层A冻结、B更新和基座无grad；随后用实际checkpoint/表示验证raw/post重载行为。
3. 在同源同目录下中断并从last.pt恢复，核对后续样本序列、RNG、实际更新计数、NLL、最佳状态及完整生成；修复后补上F1三个导出文件、F2来源变化拒绝、F6显式跨目录边界验收。
4. 实测micro2与最长完整回答的训练/验证峰值，以及单卡独立8任务并行时的资源；若OOM只能记录失败并提出有标签的补充，不可擅改精度、模型、batch、损失或target完整性。
5. 保存分别训练6/14/22与联合三向量的记录。工程smoke通过只证明局部语义，不证明各任务有效，更不证明跨seed、共享、组合、未见任务结论成立。

本报告只提出问题与修复建议。科学源码、默认实验参数及交付入口均未在本轮改动。
