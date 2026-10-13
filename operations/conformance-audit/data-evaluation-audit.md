# 数据、生成、评测与统计一致性审计

审计日期：2026-10-10。只读核查；本轮未安装依赖、未重跑测试、未加载正式资产、未运行训练或 NPU 作业，唯一写入是本审计文件。

依据为 `D:/DataSpecificWork/task-vector/docs/原始实验方案.md` 全文（211行，SHA256 `D689BC2CB01A9DFE88B0F42CE2B385AC43AC0A2633DEA53E582882E3CCFD2A5C`）及 `docs/补充协议与实现选择.md` 全文（34行）。现有需求矩阵仅用于映射 R 编号，不作为实现证据。

本审计检查的 `data.py / prompts.py / metrics.py / code_execution.py / joint.py / evaluation.py / analysis.py / runner.py / workflow.py` 以及上述两份协议，在原源码与 `C:/Users/hujin/.codex/visualizations/2026/10/10/01a1246d-d740-7b00-9bbb-fb8379c1ee94/modelmate-runbook/payload` 中逐文件 SHA256 一致，因此以下科研实现判断同时适用两处。新包的配置工具只作为部署接口补充核对。

## 结论与优先处理项

默认配置下，六任务及 SAMSum 的数据适配、主指标、完整回答编码、非截断过滤、原程序 ID 联合划分、确定性生成、原单位迁移矩阵、corpus 指标 bootstrap 和表面形式对照均有可追踪实际调用。不能据此宣称“全部要求已无遗漏”：

1. **明确阶段偏差：首轮 9216 次生成没有按原文顺序落实。** `run_core` 实际用 discovery128；256/正式测试在分解和组合选择之后才运行。见 D33，亦应与主审计中的编排问题合并。
2. **评分证据缺口：ROUGE 没有 scorer signature/version。** 补充协议要求每次输出，当前 summary 的 `signatures` 为空。见 D28。
3. **格式判据不一致：单任务 JSON 允许额外字段。** 指令要求只有 `answer`，解析器却接受任意带字符串 `answer` 的对象。见 D27。
4. **部署格式部分支持：CSV/TSV 不能通用承载七任务的嵌套字段。** 特别是 MBPP 的字符串 `test_list` 会被拆为字符列表，导致参考验证失败。JSON/JSONL、保留类型的 Parquet 和 HF DatasetDict 不受此项影响。见 D31。
5. **联合 bootstrap 的组依赖限制：** 拼接两个子组作为一个组，不能保留“同一 theory/重复问题组配不同对侧组”的共享依赖；CI 可能过窄。这是推断限制，不把原文没有指定的多路聚类方法冒充强制要求。见 D30。
6. **主配置防偏差校验不完整：** CLI 可以显式改样本数或删表面形式对照，不需要 `supplement_label`；默认配置和 configure 工具本身保持正确数值。见 D34。
7. **实机与容量全部未验：** 正式 Qwen3 tokenizer、七任务数据、真实语义模型、NPU 生成与正式指标没有运行证据；MBPP374恰好耗尽300/37/37，不能保证过滤后可运行。见 D16、D35。

## 实际调用链

- `cli.py:67–83`：真实本地 tokenizer → `prepare_dataset`；`--scope all` 明确包含4主任务、2代码任务和 summary。未用伪数据兜底。
- `data.py:411–418,420–481,494–529`：官方源加载/适配 → 完整答案及所有预定形式长度交集 → 跨官方来源组处理 → 训练/调参/发现与评测拆分 → ID/hash manifest。
- `runner.py:46–48,195–235`：读取同一 prepared 文件 → 核对锁及资产 → `generate_predictions` → `score_predictions` → 保存逐样本和总分。
- `runner.py:266–271` → `analysis.py:172–199` → `metrics.py:196–242`：配对组抽样，重算 corpus 指标；不重复运行程序或语义编码器。
- `workflow.py:291–295`：固定 canonical 训练表示，在最终测试上加入输出格式、prompt 改写和逻辑标签重编码；不在这些最终分数上重新选择表示。

以下“已覆盖”指默认科学路径在源码中的实现覆盖；均不表示正式数据或 NPU 验收通过。“部分/偏差”单独说明可证实的缺口。“待实机”是验证状态，不等同于缺失实现。

## 逐项对照

实现路径除特别说明外均相对于 `D:/DataSpecificWork/task-vector/src/task_vector/`；测试路径相对于 `D:/DataSpecificWork/task-vector/tests/`。每项均给出实际文件和一基行号。

| 审计编号 / R映射 | 原文与补充协议 | 实际实现及调用证据 | 判定、限制与建议 |
|---|---|---|---|
| D01 / R03,R27,R61 | 原文15、72、78；补充15 | `prompts.py:20–29,32–61,73–78` 统一7任务及联合任务，`apply_chat_template(...enable_thinking=False)`；`runner.py:277–285` 同一目标样本和指令，仅换源表示。 | **已覆盖；待实机。** system/user/assistant 前缀固定，Qwen非thinking scaffold属于prompt。正式 tokenizer/template 尚未加载验证。 |
| D02 / R04 | 原文21 | `data.py:262–267` 保留官方完整 answer，强制包含 `####` 并保存最终答案元数据；`prompts.py:81–96` 编码完整回答。`test_data.py:20–28` 检查完整答案、mask和EOS。 | **已覆盖；待真实数据。** 没有仅取数字替代训练推理过程。 |
| D03 / R04 | 原文21、72 | `metrics.py:28–56,147–155,215–216` 从完整生成提取最终数值，优先####、boxed，再取末数；精确有理数比较，未解析算错；`evaluation.py:152–166` 保留原始输出和停止原因。`test_metrics.py:13–21` 有人工金例。 | **已覆盖。** 提取容许无####但有明确末数，是显式可审计规则；截断率独立报告，未把截断答案悄悄重生成。 |
| D04 / R05 | 原文22、38 | `data.py:236–243,268–307` 强制OWA，True/False查询深度0–3；Unknown允许QDep=-1但要求source_depth/maxD证明来源；theory文本分组。原始目录路径过滤 `data.py:160–167`。`test_data.py:31–41`。 | **已覆盖；待真实数据。** CWA配置拒绝，Unknown不因负深度误删。原始文件身份仍须由实际资产来源验证，不能用配置中的OWA字样证明任意文件就是官方OWA。 |
| D05 / R05 | 原文22 | `metrics.py:59–61,156–162` 只接受完整True/False/Unknown标签（允许大小写及单个末尾标点）；`prompts.py:100–112` 先逆解标签/格式；`data.py:525` 保存各split标签分布。`test_metrics.py:24–26`。 | **已覆盖。** 长解释不会被搜索其中某个标签而错误判对。 |
| D06 / R06,R14 | 原文23、38 | 默认 `configs/experiment.yaml:33` IWSLT2017 en-de train/test；`data.py:188–213` 读成对en/de、XML，校验配对条数，dev与test区分；`data.py:309–314` 始终en→de。 | **已覆盖；真实原始格式待验。** 原始目录必须确实只含所需语言对；代码发现多个train.en会拒绝。XML按原目录内非dev评测文件汇集为test，应在正式来源清单明确包含哪些官方年份/文件。未查到该原始目录加载器的专门测试。 |
| D07 / R06 | 原文23；补充16–17 | `metrics.py:131–136,163–165,219–228` SacreBLEU chrF char6/word2/beta2，即chrF++；BLEU 13a/exp；主分为corpus chrF++而非句级平均，保存SacreBLEU签名。`test_metrics.py:29–35` 明确区分corpus与句级平均。 | **已覆盖。** `sample_scores` 为句级主分供std/阈值，整体主结果用corpus重新计算。 |
| D08 / R07,R15 | 原文24、28、38 | `data.py:251–255,315–320` 仅label/is_duplicate=1，问题一→问题二，保留source_question/qid；全体官方split先构重复问题连通图，再分配样本。 | **已覆盖。** 非正例不混入生成训练；不预设任务组。 |
| D09 / R07 | 原文24、28；补充16 | `metrics.py:166–180,229–237` iBLEU=.9×refBLEU−.1×sourceBLEU；copy_rate是NFKC/大小写/空白归一化后的完全复制，另存copied_token_fraction；`metrics.py:95–128,191–193` 独立离线all-MiniLM源/参考两路cosine，缺模型报错。`test_metrics.py:38–59`。 | **已覆盖指标；真实语义编码器待验。** 现有测试mock了语义分数，仅证明调度和不静默省略。近复制是否解释收益仍须结合实际复制率/词重叠结果判断，不能只见iBLEU上升就宣称能力共享。 |
| D10 / R08 | 原文25、72 | `data.py:321–327` MBPP full程序和官方test_list（另含提供的challenge测试）；`metrics.py:181–187` 调用真执行；`code_execution.py:61–121` 超时/加载错误/任一测试失败均不得pass_all；`evaluation.py:62–113` 单次greedy。`test_metrics.py:76–91`。 | **已覆盖；真实MBPP/NPU生成待验。** 没有pass@k、多候选挑最好或用字符串相似度冒充执行。程序在CPU子进程执行，NPU仅生成；这不改变单次通过率定义。 |
| D11 / R09 | 原文26 | `code_execution.py:124–181` 先执行正确参考，再尝试确定性AST运算/常量/return变异，必要时函数运行错误；必须观察原程序pass_all且变异失败。`data.py:430–456` 保存mutation、原/错执行链和故障程序prompt；`metrics.py:181–187` 对完整修复再次真执行。 | **已覆盖；正式错误分布待验。** runtime_fault兜底是显式实现选择；应报告各mutation类型占比，不能把全是显式raise的容易错误结果推广为所有真实软件缺陷修复能力。 |
| D12 / R11 | 原文30–35、38 | `configs/experiment.yaml:23` 与 `data.py:482–500` 4任务512/128/128/256；前三类仅官方train；`_group_select`在组内取需要的样本但整组余下成员不会进入后续split；`data.py:366–383` 审计组交集。`test_data.py:53–57`。 | **已覆盖；容量待真实数据。** 组内不足边界不会把同一theory/重复问题组的余样放到tune/discovery。精确样本数优先于完整保留每组所有query，属于文档未要求全组完整保留时的合理实现。 |
| D13 / R14,R61 | 原文38、176–178 | `data.py:87–185` 支持HF save_to_disk、显式files、带official_split的raw、本地原始官方文件；记录原文件SHA/fingerprint/source_counts。未声明split时拒绝；remote仅显式allow_remote且40位不可变revision，默认关闭；`cli.py:70–74` 拒绝覆盖清单并只从本地模型加载tokenizer。 | **已覆盖加载和追踪；官方身份待资产验收。** 文件hash保证追踪同一份内容，不自动证明其下载来源官方。正式本地文件尚未核对来源/版本，不可将“schema可加载”等同于“官方数据已验证”。 |
| D14 / R15 | 原文38 | `data.py:275,319` theory文本hash、QQP正例重复问题union-find组；`data.py:476–481` 官方评测组优先，train中的相同组全部剔除。`test_data.py:31–50` 及 `test_analysis.py:36–41`。 | **已覆盖；真实跨来源重合率待验。** 不仅检查最后抽出的样本ID，也处理跨官方split的同组。 |
| D15 / R16,I05 | 原文38；补充9 | `data.py:323,430–467,508–513` 同一原始program_id构造generation/repair；两任务prompt及全部预定表面形式的长度交集在划分前确定；verification缓存指纹包含代码/测试/setup/执行配置；所有split的原ID序列必须完全相同。`test_data.py:90–113`。 | **已覆盖。** 不是先各自划分再随意配对，也不会只因repair更长而悄悄丢失其一任务的样本。 |
| D16 / R17,I05 | 原文40；补充9、32 | `data.py:74–84` MBPP complete官方ID区间（train601–974、test11–510）；默认300/37/37/256在 `data.py:482–490`；不足时严格抛错，先持久化rejected及capacity_preflight（492–499）。`test_data.py:60–67,79–87,90–109`。 | **实现已覆盖，正式可运行性待真实数据。** 374个train ID恰好全部需要，原程序未通过、构错失败、重复或长度不合格任一条都可使准备停止；不允许宣称已保证完整范围能运行。严格停止符合批准协议，不是应偷偷削减的缺陷。 |
| D17 / R12,I10 | 原文35；补充32 | `data.py:495–507` formal=min(1000,可用)，不小于初始test；test为formal前256；另构与初始test整组隔离的test_confirmation；`audit_splits:377–378` 明确允许formal包含initial/reserve，不虚称独立。`runner.py:237–251` 记录包括缓存读取在内的test暴露。 | **已覆盖；正式容量待验。** `test_formal`是同锁方案扩评，可含已用test；若看完初始test后改方法，必须用未暴露confirmation或新数据，不能把formal自动称新独立证据。formal“完整测试集”指长度/合法性过滤后的eligible panel，拒绝记录必须一起报告。 |
| D18 / R26,R59 | 原文59、168；补充15 | `prompts.py:81–96` prompt≤512、target连EOS≤512，合计自然≤1024；`data.py:457–473` 默认canonical/prompt_paraphrase/output_format/label_recode全交集，保存各variant token lengths，遇超长仅过滤不截断。`test_data.py:20–28,79–87`。 | **已覆盖；真实Qwen token长度待验。** 交集避免不同格式下评测样本发生变化；EOS计入512是公开补充选择。 |
| D19 / R11,R61 | 原文32–38、177 | `data.py:395,494,499,519` 固定split_seed和ID/hash；`runner.py:46–48` 两路线/所有训练seed读取同一manifest文件，不在训练时重新抽样。`data.py:533–547` 每次读split校验完整状态、实际SHA及条数。 | **已覆盖。** 正式多seed仅表示初始化/优化seed变化，固定数据不随17/29/43重新划分。 |
| D20 / R21,R23 | 原文51、58、61–69；补充11、15 | `prompts.py:93–96` prompt mask、完整回答+EOS；`losses.py:36–70` 逐样本回答token平均；`evaluation.py:27–51` 对tune逐样本NLL求平均；训练调用在 `training.py`。`test_training.py:18–30,33–47,91–114`。 | **已覆盖目标与验证接口；训练调度由训练审计复核。** 不另外下载NLL“数据集”。保存最低tune NLL不等同于用最终accuracy选checkpoint。 |
| D21 / R27,R28,R30 | 原文72、76–78 | `evaluation.py:62–113` native KV cache、prefill标记、逐步argmax、无采样/隐式processor；`evaluation.py:116–151` 固定目标prompt、左padding、限制单任务prompt512；生成上限见11–13及配置74。`test_ltv.py:45–59,115–130`。 | **已覆盖；NPU缓存/容量待验。** 不因换源向量换目标指令。小Qwen CPU测试检验缓存及padding语义，不能替代正式Qwen3/NPU验证。 |
| D22 / R28,R49 | 原文72、130、202；补充22 | `evaluation.py:152–166` 保存完整prediction/token IDs、生成及prompt/reference长度、reference超预算标记、eos/length与truncated、metadata/variant；`metrics.py:241–242` 所有面板报告truncation_rate；联合prompt/output上限1536在配置73–74。 | **已覆盖正常生成结果；待实机。** reference超生成预算不被裁短或修改。超过联合prompt上限会严格报错，不能视为生成成功；当前异常只给任务级overlong提示，没有专门持久化超长联合样本清单，建议正式准备时增加联合长度预检证据。 |
| D23 / R49,R50 | 原文123–132；补充19 | `runner.py:277–285,295` 每个source→target增益；`analysis.py:202–211` 保存raw及按目标BASE逐样本ddof1 std列缩放的standardized矩阵；zero_std保留raw并排除聚类。`metrics.py:238–242` 保留单位。`test_analysis.py:44–50`。 | **已覆盖。** 句级分数仅供尺度，不把准确率/chrF/iBLEU直接平均；选择目标平均的是标准化主指标增益并保留原单位结果。 |
| D24 / R50 | 原文130；补充16、20 | `analysis.py:172–199` 先按ID一一对齐，再按group_id配对bootstrap、每轮重算score_fn；`runner.py:266–271` 实际score_fn为 `aggregate_predictions`；`metrics.py:196–234` corpus重算、程序与语义用缓存；默认2000次/.95在配置76–77。`test_analysis.py:36–41; test_metrics.py:29–53,76–85`。 | **已覆盖单任务配对组bootstrap。** 输出是效应置信区间，没有伪装成经多重比较校正的显著性检验；原文未指定额外检验，不将没有FDR当成漏做要求。 |
| D25 / R10,R51 | 原文28、132；补充20 | `runner.py:369–397` 发现集按原组bootstrap重建功能矩阵；`analysis.py:214–250` outgoing标准化profile、average linkage、coassignment稳定性，允许单组及无可识别组；`workflow.py:219–228` 六任务跨seed同分组才后续组内处理。`test_analysis.py:44–50`。 | **已覆盖调用；非平凡聚类稳定性验证不足。** 现有针对此域测试主要验证零方差/不强制分组，未见对有稳定非平凡partition及coassignment阈值边界的独立金例；真实数据分组全部待实验。 |
| D26 / R59 | 原文168 | `prompts.py:21–22,46–70` 预定prompt改写、JSONanswer格式、T/F/U→A/B/C；`prompts.py:100–112` 逆解；`workflow.py:291–295` BASE/原向量及成分完整重评表面变体，保持锁定分解；`data.py:460–473` 同样本长度交集。`test_data.py:70–76; test_metrics.py:24–26`。 | **已覆盖主要混淆对照。** 是固定表示下测试指令/输出表面的变化，未擅自重训为另一实验；label_recode对非逻辑任务等同canonical，其重复结果不能算额外独立证据。 |
| D27 / R59 | 原文168；本实现指令 `prompts.py:60` | `prompts.py:102–106` 仅检查dict及answer为str，没有 `set(obj)=={'answer'}`。 | **部分／判据偏差（小范围）。** `{"answer":"True","extra":"..."}`仍被当合法格式，与自身“single key answer”不符。建议严格匹配单键、另存format_valid；现有test_data.py:70–76只覆盖plain-text拒绝，未测额外键。 |
| D28 / R06,R07,R48；补充指标签名项 | 补充17“每次输出scorer signature/version” | `metrics.py:144–146,188–189` 实际使用stemmed ROUGE1/2/L F1×100；但 `aggregate_predictions:214–238` 只给抽取规则、BLEU/chrF/iBLEU及语义签名，summary没有ROUGE signature或rouge-score版本。`test_metrics.py:94–97` 只检查分数和单位。 | **部分／证据缺失。** 补全rouge-score版本、use_stemmer、keys和F1定义；并建议在统一scorer元数据中记录所有相关包版本。ROUGE计算本身已有实现，不误报为整个SAMSum指标缺失。 |
| D29 / R47,I06 | 原文120、159–166；补充8、18、21–22 | `joint.py:14–45` 每无序pair真实两子任务+各自GT、确定性配对；`joint.py:48–125` 严格双字段JSON、分别原单位分数、二元正确/代码全通过、连续指标冻结阈值及联合达标；`joint.py:128–146; workflow.py:203–205,272–275` 仅BASE tune句级中位数校准并锁阈值及ID；`runner.py:400–485` 同预算raw/matched候选选择及原任务/联合最终评分。`test_joint.py:16–55`。 | **已覆盖；真实联合生成待验。** 联合达标率不是把异质主分直接平均，也不是连续任务天然“正确率”。默认joint_count128为每pair设置；原文未给联合规模，不把它宣称1000条的单任务正式扩评。 |
| D30 / R50,R47；统计解释限制 | 原文130；补充8、20 | `joint.py:40–42` 联合group_id为A.group|B.group；`runner.py:485→delta` 复用 `analysis.py:182–190` 单个group_id抽样。 | **部分／推断限制。** 若同一theory或QQP连通组的多条query与不同对侧组配对，拼接ID不同，bootstrap把共享子组拆成“独立”组。点估计和配对ID仍正确，但CI可能低估相关性；建议预先锁定保留两路组依赖的联合重采样方案，或显式限制和说明其CI解释。没有现有联合共享子组bootstrap测试。 |
| D31 / R14,R16；部署格式接口 | 原文38；新包指南194称CSV/TSV等可加载 | `data.py:65–67` CSV/TSV只作DictReader；`data.py:310–311` 仅translation特殊JSON解码；`data.py:324` `list(row['test_list'])` 遇字符串会拆字符，ProofWriter的questions/triples也没有CSV嵌套解码。新包 `ops/configure_local.py:63–65,81–83` 接受这些后缀。 | **部分／部署输入格式缺口。** 原文数据集要求仍可用HF/JSONL/typed Parquet完整满足，但不能泛称全部七任务CSV/TSV均支持。建议文档按任务缩窄支持矩阵，或对嵌套字段明确JSON解码并加金例。没有正式CSV MBPP/ProofWriter测试。 |
| D32 / R11,R12,I10 | 原文35、132；补充18 | `runner.py:203–217` 最终评测前验锁、源代码/配置/manifest及实际基座+语义资产指纹；`config.py:95–125` 锁不覆盖；`runner.py:237–251` 解封ledger；`workflow.py:272–295` 锁后final。 | **已覆盖正常调用。** 用户仍可直接查看准备后的test文件；这是代码不能消除的人为信息暴露，不将其当自动保证盲测。full和formal是否实际按所需顺序运行，见D33及主编排审计。 |
| D33 / R66,R67,R68,I03 | 原文209–211，首轮9×4×256=9216，然后才共享实验 | `workflow.py:176–181` 首轮调用 `core_panel` 不传split，`runner.py:273` 默认为discovery（128）；`workflow.py:380–382` 先run_discovery再锁再run_final；`suite.py:31–38` 仅计算9216预算，没有专用首轮完成记录。 | **偏差／阶段缺口。** 256样本及9核心状态最终可以评分，但没有按原文在共享投入前完成256首轮，也没有真实9216完成证据。建议独立首轮256阶段、冻结/解封与完成记录，并在进入分解前门控；不能把inventory里的数字当实际生成。如何与I10独立确认兼容，需主审计统一协议。 |
| D34 / R11,R12,R26,R59；防偏差校验 | 原文30–38、59、168；补充32 | `config.py:30–69` 校验训练超参/部分生成设置，但不校验data.counts/counts_by_task/formal_test_size或evaluation.variants/data.eligibility_variants；`cli.py:41–48` 显式--set可修改；`data.py:482–490` 应用自定义counts，`workflow.py:291` 按可删的variants循环。新包configure仅保留模板超参，不提供这些覆盖。 | **默认覆盖，防偏差校验部分。** 一个直接CLI配置可以缩样本或删格式控制且无需supplement_label，不符合“科学变更应明确补充”校验意图。建议主协议schema锁定这些值/必需集合，变更需已批准补充标签和新root；不声称默认新包已经擅自变更它们。 |
| D35 / R02,R04–R17,R26–R28,R49–R50,R59,R68 | 原文11、207、211；补充34 | 已读取两份JUnit原文件：`validation-cpu28.xml` 与 `validation-cpu214.xml` 各tests84/failures0/errors0/skipped0；对应 `docs/本地验证报告.md:7–12,27,31` 明确CPU/小模型、无NPU、无正式资产。 | **待实机／不得升级结论。** CPU只验证程序语义、矩阵、人工金例、缓存和小Qwen行为。没有正式容量、NPU版本/性能/峰值、真实all-MiniLM输出或E1–E6功能结果证据。本轮只读取报告，没有复跑84项。 |

## 现有测试能证明和不能证明的内容

原始JUnit报告路径：

- `C:/Users/hujin/.codex/visualizations/2026/10/10/01a1246d-d740-7b00-9bbb-fb8379c1ee94/validation-cpu28.xml`
- `C:/Users/hujin/.codex/visualizations/2026/10/10/01a1246d-d740-7b00-9bbb-fb8379c1ee94/validation-cpu214.xml`

本域现有测试8项data、9项metrics、4项joint、4项analysis，以及training/LTV的相关小模型测试，均出现在两份通过报告中。测试证据应按以下边界解读：

- **较强的程序语义证据：** EOS/mask、Unknown深度、跨split QQP连通、组分配不重用、MBPP区间和共享ID、真实小程序构错/执行/超时、corpus重算与缓存不再执行、联合阈值不从test校准、零方差不强制分组、左padding/cache小模型行为。
- **弱或缺少专门测试：** 真实HF save_to_disk往返、原始IWSLT en/de/XML文件加载、非平凡稳定聚类及阈值边界、所有表面形式的真实Qwen token长度交集、CSV嵌套字段、额外JSON键拒绝、ROUGE签名、联合两路组依赖bootstrap。它们不能因“全套84通过”被省略。
- **完全待线上证据：** 真实7任务的各split容量与拒绝原因、MBPP374全部合格、实际语义模型推理、Qwen3-1.7B NPU greedy与停止行为、真实峰值与吞吐、首轮9216、每个最终评分面板及置信区间、跨seed和表面控制后的共享结论。

## 建议的审计闭环

先由主审计统一解决首轮阶段顺序及正式完整范围；随后修正格式判据/ROUGE元数据、明确本地输入格式支持范围，并对联合CI的依赖假设做显式协议选择。真实准备先产出source/capacity/rejected/ID/hash证据，严格遵守MBPP容量冲突规则；线上验收通过后再判断功能效果。所有这些建议均未在本轮改写源码或科学设置。
