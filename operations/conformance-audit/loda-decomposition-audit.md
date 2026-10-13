# LoDA-MT、实际更新分解与顺序对照的独立符合性审计

审计时间：2026-10-10 20:49 +08:00。源码根目录：`D:\DataSpecificWork\task-vector`。本报告依据原文、补充协议、实际函数体、测试函数体和已保存官方源码逐项核对。没有以既有覆盖审计/需求矩阵中的“已实现”作为实现证据，没有修改科研源码、安装依赖、运行训练或调用模型。本轮仅阅读测试，未重新执行测试。

“覆盖”表示静态检查确认算法及默认接线存在，不表示真实模型/NPU实验通过；“部分”表示主计算存在但证据、接口或测试不足；“缺失”表示未找到所需接口/自动验收；“偏差”表示当前执行顺序与原文明示要求不符；“待实机”表示只有真实平台才能验证。文档没有明确规定的数值不升级为原文要求。

## 总体判断与需优先处理的缺口

LoDA-MT 的主要数学路径与原文及用户批准协议一致：固定基座统计、token二阶矩均值、TopEig General、白化 Isolated、A/√3、8+8、FP32 B、0.5 General + 1 Isolated、固定基座重标定、raw/post、官方一阶双向顺序 GAO均可定位到函数体。实际更新几何与任务子空间分解使用 BA 所表示的矩阵，未发现直接拼接 A/B 因子代替实际更新的情况。

需要明确处理或记录：

1. **P1，首轮次序偏差（L59）**：`core` 实际评测 discovery128；`all` 先选择分解/组合后才最终测试。现有 full256/full-formal 均没有独立执行并留存原文“seed17，九种核心状态×四任务×256=9216，成立后再投入后续研究”的阶段。
2. **P1，顺序纯扩评/恢复接口缺失（L54–L55）**：输出路径不含split，驱动每次调用训练并覆盖同名stage/report；不能把再次执行ablations当成无重训的正式扩评。已有真实合并权重，可补独立评测，无须为扩评重新训练。
3. **P2，统计缓存签名不含源码/模板/实际样本内容（L02）**：同ID、同模型、同loda设置而prompt模板或样本正文改变时，旧统计可以通过现有签名检查。应冻结输入token内容或数据清单内容hash以及统计/模板实现hash。
4. **P2，缺少shared对随机对照的直接配对区间（L40）**：所有逐样本预测存在，但当前只分别计算各条件相对BASE的区间。两条相对BASE的区间不能替代shared−random_shared或group−random_group的直接配对区间。
5. **P2，顺序驱动及随机子空间的自动验收缺口（L38、L57）**：未找到run_sequential、双累加、阶段真实模型继承、纯扩评防重训的端到端测试；随机shared控制没有直接的span秩/范数/q=0/参数重参数化测试。已有stage0开关测试不能证明整个驱动正确。

## 依据与官方快照

原文：[原始实验方案.md](D:/DataSpecificWork/task-vector/docs/原始实验方案.md)，211行，SHA256 `D689BC2CB01A9DFE88B0F42CE2B385AC43AC0A2633DEA53E582882E3CCFD2A5C`。补充协议：[补充协议与实现选择.md](D:/DataSpecificWork/task-vector/docs/补充协议与实现选择.md)，34行，SHA256 `1A2018229CC4FEBA040CC74002EA4072AE5E53113248C77E26A6176B0DD4087D`。下表“原”指原文行号，“补”指补充协议行号。

官方依据是本地已保存、固定commit的源码快照，而非目前网络HEAD；本轮未另行联网更新。LoDA commit为 `a07b79b290c2534ad0322679882a66ea67334c03`，见[official_sources.json:44](D:/DataSpecificWork/task-vector/third_party/official_sources.json:44)。重新计算的两个文件hash均与清单一致：

- [官方decomposed_lora.py](D:/DataSpecificWork/task-vector/third_party/LoDA_ICML2026/models/decomposed_lora.py)：`89ACC79367A2D26B6D7F952D823FA5361A239EA0364FF5A15AF2EF22480E53FA`。
- [官方dlora.py](D:/DataSpecificWork/task-vector/third_party/LoDA_ICML2026/methods/dlora.py)：`C7526617D7E81917268C2EC3DAFDAF0A200AFEE026DCA270B813825FCB687700`。

原文只要求将ViT算法接入Qwen语言模型（原183），不能要求逐字保留ViT分类器、cosface、类别分组或官方优化器/调度器。AdamW/线性调度/语言模型回答损失是原文明确采用的适配。固定theta0统计及post主结果为用户批准内容，不应误报为与官方顺序统计不同而构成违规。

## 逐项矩阵：LoDA-MT与优化

| ID | 原文/协议要求 | 实际源码证据 | 判定 | 测试证据、缺口与建议 |
|---|---|---|---|---|
| L01 | 原47、89：所有任务同一冻结theta0，任务更新独立 | [runner:50](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:50)加载基座；[runner:137](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:137)训练传同基座；[training:127](D:/DataSpecificWork/task-vector/src/task_vector/training.py:127)冻结；LoDA训练仅安装/移除hook，不merge | 覆盖 | test_loda36–55检查冻结、零更新、退出恢复；test_training125–145覆盖非零LoDA训练后基座冻结。缺多任务依次训练后theta0逐张量不变的端到端断言。 |
| L02 | 原90、补11：MT统计必须与当前冻结基座及真实输入匹配 | [runner:89](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:89)只在theta0采集/加载；97–104签名含base、loda、maxprompt、trainIDs | 部分 | 基座hash变更会拒绝旧统计；签名不含源码/模板/样本正文/清单内容hash。代码或输入正文改变但ID不变时可能复用不一致统计。建议缓存签名加入内容和实现hash；新增模板变化、同ID正文变化拒绝缓存的测试。 |
| L03 | 原90：训练集中取128条 | [subspaces:78](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:78)count默认128；83–91检查唯一ID、official train、固定seed17随机无放回选择 | 覆盖 | test_subspaces114–133测试2样本toy收集；尚缺128样本真实容量、重复ID、非train拒绝的专门测试。固定统计seed17是公开实现选择，不随训练seed重抽。 |
| L04 | 原90：每条最多64个均匀非padding prompt tokens | [subspaces:39](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:39)均匀位置含两端；125–138完整prompt前向、单条无padding、只取指定位置 | 覆盖 | test_subspaces14–20覆盖完整/唯一/均匀；114–133覆盖token数量。超过锁定prompt长度直接报错，不截断。 |
| L05 | 原90、108：收集各层QKV输入，QKV共享输入基 | [subspaces:50](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:50)找到所有q/k/v且核input width；101–121只hook q_proj输入一次 | 覆盖 | Qwen原生q_proj输入位于block input RMSNorm之后；hook不在residual RMSNorm之前，也不收集Q投影后的feature。toy collector没有RMSNorm，真实28层位置仍需NPU/真实Qwen检查。 |
| L06 | 补10：未中心化、按所有所选token求二阶矩均值 | [subspaces:110](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:110)selected.T@selected；146除各层实际token总数；147标注centered=False | 覆盖 | test_subspaces114–133明确比较token mean与不等长sample mean，能排除错误的“每样本二阶矩再均值”。官方decomposed_lora193–206同样按token数累计二阶矩。 |
| L07 | 补7、10、28：NPU前向、FP32统计/B；允许CPU谱计算适配 | [subspaces:93](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:93)device FP32矩阵及112 addmm；157–171转CPUfloat64谱分解；[loda:25](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:25)master FP32 | 覆盖；待实机 | 数学计算device路径存在；没有真实Ascend执行证据。CPUfloat64仅谱与统计后处理，不是把模型前向/反传移到CPU。 |
| L08 | 原91–94：UG=TopEig8(sum_t S_t) | [subspaces:243](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:243)current+others；166–171对称eigh降序取最大特征值 | 覆盖 | test_subspaces23–27对角能量排序；63–74见任务集合与A尺度。官方373–390使用PSD矩阵SVD，PSD条件下与TopEig取同一主子空间；不需机械照搬SVD API。 |
| L09 | 原95–99、E6原121：S_other只含当前scope源任务，排除heldout | [runner:70](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:70)留出taskset；89–108只取源统计；[subspaces:228](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:228)seen_tasks并243求除自身外之和 | 覆盖 | test_subspaces63–76排除heldout；test_workflow221–239实际检查不读取heldout train也不生成其统计文件。MT四折需分别重建并重训，已有runner scope隔离。 |
| L10 | 原99：Isolated为Cholesky白化→特征分解→映回→QR | [subspaces:174](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:174) L^-1 S_self L^-T；198取top eig；200 L^-T映回；201 QR | 覆盖 | test_subspaces30–52既检查对角例子又检查稠密广义特征子空间残差；未额外从UG中投影移除Isolated，原文没有要求两分支正交。对应官方408–415。 |
| L11 | 原106：正则1e-4 tr(S_other)/d | [subspaces:181](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:181)精确公式；184加I；187–205单列有限精度jitter及元数据 | 覆盖 | test_subspaces30–38检查正则值/jitter=0；79–81检查other零能量时报错。官方_get_L用逐次eps修复而无该trace项；这是原文明定适配，不能误标偏差。建议补jitter触发记录测试。 |
| L12 | 补10：A=正交输入基转置/√3 | [subspaces:237](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:237)scale默认1/√3；252两个基同尺度；config59相同 | 覆盖 | test_subspaces63–74验证AA^T=I/3；对应官方375、390、425。config.validate_config未直接锁a_scale；默认正确，但若要求所有科学偏离均强制supplement_label，应补该键校验。 |
| L13 | 原48–49：28层attention QKV，General8+Isolated8 | [modeling:25](D:/DataSpecificWork/task-vector/src/task_vector/modeling.py:25)核28层/Qwen；[subspaces:236](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:236)8+8；[config:44](D:/DataSpecificWork/task-vector/src/task_vector/config.py:44)主rank限制 | 覆盖；待实机 | 现有测试使用2rank和小模型；实际28层、2048/1024异宽8rank部署须实机验收。 |
| L14 | 原46–47、100：theta0和两个A冻结，只训两个B | [loda:29](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:29)MT AG/A_I buffer；42–46 BG/BI Parameter；83–84冻结模型 | 覆盖 | test_loda36–49检查参数数量、dtype、冻结。stage0可训AG仅由独立顺序driver打开，不能据此误判MT不冻结。 |
| L15 | 原50：B=0初始化，未训练更新等价基座 | [loda:42](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:42)BG/BI零；[training:181](D:/DataSpecificWork/task-vector/src/task_vector/training.py:181)保存initial | 覆盖 | test_loda36–55验证raw/post零B精确等价；真实Qwen/NPU smoke136–150有同检查，但本轮无远端执行证据。 |
| L16 | 原108：Q/K/V异宽，共享输入子空间，保留GQA/QK norm/RoPE | [loda:25](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:25)每层单份AG/AI；39–46按native out width建各B；115–119只hook线性输出 | 覆盖 | test_loda47–49验证Q6、KV3异宽；58–78比较实际BA与hook。真实注意力由原Qwen代码继续处理，未替换为ViT多头实现；部署需实机。 |
| L17 | 原103：ΔW=.5 BG Λ AG + BI AI | [loda:53](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:53)前向0.5+1；167–175导出将0.5及Λ吸收进BG | 覆盖 | test_loda58–78验证raw/post dense实际更新、前向与merge。没有额外LoRA alpha/r缩放。 |
| L18 | 原101、106、补11：固定theta0统计重标定，lambda3，raw/post均存，post主 | [subspaces:209](D:/DataSpecificWork/task-vector/src/task_vector/subspaces.py:209)Λ=1−e_other/(e_other+3e_self+eps)；[loda:140](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:140)固定S_self/S_other；[training:267](D:/DataSpecificWork/task-vector/src/task_vector/training.py:267)raw/post保存；[runner:180](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:180)默认post | 覆盖 | test_subspaces55–60验证1−eta及独立eps；test_loda58–95覆盖导出/重载。公式和官方66–104一致，MT统计来源的差异已批准。 |
| L19 | 补11：checkpoint按raw训练态每epoch回答NLL选择，post用于主评测 | [training:143](D:/DataSpecificWork/task-vector/src/task_vector/training.py:143)训练set_recalibrated(False)；247–266逐epoch NLL最低、恢复best再导出 | 覆盖 | test_training125–145覆盖完整LoDA微型训练；没有针对“raw最佳与post最佳不同仍严格选raw”的专门测试，建议补。 |
| L20 | 原58：最多5epochs，patience2，完整恢复保存/重载 | [training:109](D:/DataSpecificWork/task-vector/src/task_vector/training.py:109)轮数上限；194–201保存adapter/optim/scheduler/RNG；206、250–260 patience与best/last | 覆盖 | LoDA representation roundtrip测试存在；独立任务恢复测试主要为LTV路径，LoDA GAO第二direction后的精确恢复建议单独测。顺序driver恢复不因此自动成立。 |
| L21 | 原52–57：AdamW .001、wd0、micro2、effective16、clip1、linear decay、warmup0 | [training:109](D:/DataSpecificWork/task-vector/src/task_vector/training.py:109)batch校验；149–155 AdamW/线性更新调度；[gao:79](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:79)clip1 | 覆盖 | 默认config38–53正确；GAO每方向step后scheduler.step，两次方向在同一线性step序列上。官方dlora用epoch调度/Adam或SGD，语言模型本项目按原文采用AdamW/线性调度，属预定适配。 |
| L22 | 原51、61–69：完整回答每样本token均值后样本均值 | [losses:36](D:/DataSpecificWork/task-vector/src/task_vector/losses.py:36)causal shift、mask、每样本除counts；[training:62](D:/DataSpecificWork/task-vector/src/task_vector/training.py:62)micro losses.sum/组样本数 | 覆盖 | GAO组梯度不按长答案token数加权；micro图逐个backward并释放。训练长度与EOS编码由共同prompts路径提供，非LoDA自定义标签损失。 |
| L23 | 原56、100：16逻辑batch分两组8，样本不相交，不按类别 | [training:52](D:/DataSpecificWork/task-vector/src/task_vector/training.py:52)shuffle逻辑batch；225–231近等分/ID排重；[gao:52](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:52)互斥唯一ID | 覆盖 | test_gao104–109检查重叠拒绝；test_training尾批检查存在。补23授权尾批不丢不复制，singleton从前批移1；不足16不冒称8/8。 |
| L24 | 原106、官方271：rho=-rho_max*U(0,1)，每逻辑batch只抽一次并复用双向 | [gao:55](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:55)一次CPU RNG draw；59双方向共用rho | 覆盖 | test_gao42–56精确rho、optimizer step2。CPU抽样与官方NPU/CUDA抽样序列不逐位相同，但分布与一次复用语义一致；RNG可保存恢复。 |
| L25 | 原106：逐参数张量梯度归一化，不计算二阶 | [gao:60](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:60)source反传；67每p.grad.detach/norm+eps | 覆盖 | test_gao24–54独立autograd.grad(create_graph=False)参考；没有用所有B拼接的单个global norm，也未保留源梯度图。 |
| L26 | 原106、官方298–299：双向按顺序两次优化，而非平均梯度一次step | [gao:59](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:59)第一方向完整step再第二方向；81、86–87各optimizer/scheduler一步 | 覆盖 | test_gao42–56逐参数精确对独立参考、optimizer state step2；59–81验证micro2组8与全组梯度等价、scheduler2次。 |
| L27 | GAO撤销扰动、非有限梯度/异常不污染模型 | [gao:62](D:/DataSpecificWork/task-vector/src/task_vector/gao.py:62)检查finite；70–85 finally撤扰动；optimizer在扰动态step后撤，匹配官方290–294 | 覆盖 | test_gao84–101覆盖第一次target前向失败撤扰动。缺第二方向失败/optimizer.step异常/NPU finite错误用例；已完成第一方向不会被回滚，恢复应依赖最近完整logical checkpoint。 |

## 逐项矩阵：实际更新几何、共享分解与对照

| ID | 原文/协议要求 | 实际源码证据 | 判定 | 测试证据、缺口与建议 |
|---|---|---|---|---|
| L28 | 原117、151：LoDA geometry按实际ΔW/BA，而非A/B展平 | [representations:83](D:/DataSpecificWork/task-vector/src/task_vector/representations.py:83)∑xy〈BxAx,ByAy〉=∑(BxᵀBy)⊙(AxAyᵀ)，包括分支交叉项；[analysis:15](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:15)Gram/cosine | 覆盖 | test_analysis23–33检查非正交gauge变换不改变内积、norm、夹角。测试只有单term；建议补General+Iso多term和跨module dense对照。 |
| L29 | 原117：分层子空间夹角 | [analysis:132](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:132)QR小核SVD得到实际BA的左右奇异子空间；143–155输出input/output principal angles；[runner:319](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:319)遍历每pair、QKV各module | 覆盖 | test_analysis23–31检查等价gauge input夹角约0。输出按layer.Q/K/V模块分别报告，未用人为相同AG夹角当自然共享证据。缺不等rank/零rank/output角的专门测试。 |
| L30 | 原117：跨seed稳定、不由初始化决定，两个空间关系一致性 | [runner:311](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:311)trained/initial cosine及关系、跨空间；[workflow:255](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:255)所有seed几何相关另报 | 覆盖 | LoDA initial ΔW=0，所以initial cosine为undefined，代码转None而非人为0相似；不能用不存在的initial几何相关证明UG天然共享。functional_across_seeds独立保存，稳定性不是仅看几何。 |
| L31 | 原136–139：均值s=mean(v)，residual=v−s | [analysis:81](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:81)权重全1/T；92–96构造shared/residual、重建误差 | 覆盖 | 缺独立mean数值/两向量和全组composition退化的测试；函数体公式正确。 |
| L32 | 原142–146：归一化列组成V做SVD，shared投影原向量 | [analysis:49](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:49)归一化Gram eig实现等价SVD；63–68对原始targets求投影权重 | 覆盖 | test_analysis11–20使用非等norm向量，明确对dense归一化SVD子空间投影原X，而非投影归一化X后忘记尺度。参数矩阵BA版本缺同等dense验证。 |
| L33 | 原149–150：q0/1/2、发现集选、test锁定 | [config:68](D:/DataSpecificWork/task-vector/src/task_vector/config.py:68)强制候选；[runner:325](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:325)仅discovery允许选；[workflow:290](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:290)final传locked selected | 覆盖 | test_workflow292–296确认test前禁止现场选；analysis253–263源任务BASE std标准化目标并列保留简单候选（补19）。final原始指标完整留存。 |
| L34 | 原156：四任务至多rank4，不将低秩本身当发现 | [analysis:60](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:60)effective_rank/used_rank及singular values；方法还需functional干预与随机控制 | 覆盖 | metadata足以披露rank上界；代码不根据“rank≤4”自动宣称发现。真实结论仍需人工审阅功能证据。 |
| L35 | 原119：shared/residual/reconstruction/delete_shared四种条件 | [runner:343](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:343)三种实际干预；351 delete_shared标记alias_of residual；[analysis:94](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:94)数学同一条件 | 覆盖 | 删除shared与residual完全同义，明确alias而不重复生成是保真去重，不构成省略独立实验；test_analysis11–20检查重建。需参数多term重建干预测试。 |
| L36 | 原119：跨任务施加成分 | [runner:336](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:336)global shared测全部target；342–350每源shared/residual/reconstruction/random测全部target | 覆盖 | 既有全局shared又有任务投影s_t，未仅在原任务上测。缺直接验证全部source×target条件数/label映射的集成测试。 |
| L37 | 原119：随机同秩控制；E1原116同范数 | [representations:131](D:/DataSpecificWork/task-vector/src/task_vector/representations.py:131)参数null保持实际矩阵奇异值/数值rank并随机左右子空间；[analysis:100](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:100)随机q维任务子空间投影后匹配真实shared norm | 覆盖 | q在此是“任务向量子空间维数”，不是每个QKV矩阵的LoRA rank；不能把两种rank混称。参数null不复用人工共享AG。test_analysis32–33仅检查random_like总norm。 |
| L38 | 随机控制需可核验q维span、norm、零q与真实BA参数不变性 | [analysis:106](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:106)共用随机basis的top-q投影与逐目标范数匹配；[representations:145](D:/DataSpecificWork/task-vector/src/task_vector/representations.py:145)小核SVD保留实际奇异谱 | 部分 | 全tests未见random_subspace_control直接测试。应补activation/parameter两空间、q0/1/2、退化源、每module谱/norm、span秩、gauge变换、随机seed复现测试；不能由random_like单norm断言推导完整随机成分验收。 |
| L39 | 原157：UG人为共享不能证明天然共享，必须测更新与分支功能 | [runner:489](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:489)raw/post两版本的General、Isolated、full均在全部目标上评测；geometry按BA | 覆盖 | 分支保留0.5与Λ缩放；raw/post功能区别可追踪。未将LoDA General等同于跨任务SVD shared。branch panel没有专门接线测试；真实收益待实机。 |
| L40 | 原119、130：shared跨任务作用超过随机同秩，报告paired CI | [runner:349](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:349)每条件只delta(condition,BASE)；[runner:570](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:570)group同样只对BASE | 部分 | 缺直接paired bootstrap(shared,random_shared)、paired bootstrap(group,random_group)。预测均保存，可独立汇总补齐；均值原始差可以算，但两条对BASE的CI不能替代该contrast CI。建议在analysis中增加直接对照表及其单测。原文未规定“显著性阈值”，不能擅自加显著性门控。 |
| L41 | 原132、补20：功能迁移优先、average linkage、允许无稳定组 | [runner:369](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:369)对group_id做bootstrap功能矩阵；[analysis:214](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:214)outgoing profile、average linkage、coassignment阈值、无稳定返回一组 | 覆盖 | test_analysis44–50只测试zero-std不强迫分组。缺存在稳定组/不稳定组/多seed分区不一致的运行接线测试；阈值.8为补20公开选择，非原文数值。 |
| L42 | 原152、补20：仅六任务跨seed稳定组才进行组内SVD | [workflow:219](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:219)六任务、全部seed bootstrap稳定且无序partition完全一致；不满足就跳过并记录理由 | 覆盖 | 源码严格门控，不强迫四任务两组。缺直接group_stability_across_seeds全条件测试。 |
| L43 | 原152：去global残差后组内SVD | [workflow:239](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:239)只用global_decomp.residual；[runner:555](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:555)final同一残差路径 | 覆盖 | test_workflow299–316验证独立helper对global残差投影、global+group+specific重建、正交；实际workflow/runner手写同语义，未调用该helper，测试不能单独证明生产接线，应补actual runner集成。 |
| L44 | 组global/group/specific/reconstruction与同rank随机组功能确认 | [runner:558](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:558)group decomposition及random null；563–571五种条件×全部目标 | 覆盖 | 没有遗漏random_group；缺L40直接对照CI和group_final条件数/锁定q/全部target接线测试。 |
| L45 | 原162–166：组合mean(shared)+sum(residual) | [analysis:40](D:/DataSpecificWork/task-vector/src/task_vector/analysis.py:40)精确权重；[runner:414](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:414)成对组合采用此方法 | 覆盖 | q0退化sum有test_analysis20；mean分解/满rank全组退化mean有metadata，不称新算法。 |
| L46 | 原120、166：original/sum/mean/weighted/decomposed、同预算/范数 | [runner:400](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:400)全部策略、相同候选数、raw/matched、tune仅源任务；435–459零强度、退化/实际不同候选数公开 | 覆盖 | test_workflow281–289检查真实坐标去重和组合秩保留。组合通过完整factor terms保存，不强行截回rank8。用户批准联合面板由共同composition_final实现，非本报告重复审计评分部分。 |
| L47 | 原179–181：TIES/TSV/KnOTS基于实际参数增量 | [baselines:39](D:/DataSpecificWork/task-vector/src/task_vector/baselines.py:39)先求实际BA；107、132、169各baseline；[runner:415](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:415)LoDA组合实际调用 | 覆盖 | test_baselines38–120提取固定作者纯函数比较TIES/TSV/KnOTS并验证因子gauge不变；本轮阅读实现/测试，未执行。精细baseline算法另由总体审计合并；TSV明示作者orthogonalization变体，不能泛称任意TSV。 |
| L48 | 参数干预应保留完整组合rank、前后可逆、原生attention不变 | [representations:112](D:/DataSpecificWork/task-vector/src/task_vector/representations.py:112)累加全部terms；[ltv:147](D:/DataSpecificWork/task-vector/src/task_vector/ltv.py:147)实际参数wrapper，161–169FP32求所有BA；172–194 finally恢复原模块 | 覆盖 | test_workflow281–289检查组合秩；test_loda58–100检查前向/导出/异常hook清理；smoke152–175核真实模型raw/post保存后logit/token一致。NPU完整组合rank内存/算子待实机。 |

## 逐项矩阵：顺序LoDA与完整执行

| ID | 原文/协议要求 | 实际源码证据 | 判定 | 测试证据、缺口与建议 |
|---|---|---|---|---|
| L49 | 原85、110：四循环顺序、每order新theta0、继承上一阶段模型，独立报告 | [sequential:22](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:22)四轮转；80–89每order fresh model、阶段循环；132merge后继续；176–179独立route/report | 覆盖 | test_loda110只检查一个循环结果；完整4order×4stage×3seed实际执行没有驱动测试。analysis.sequential_orders配置虽列四个循环，driver实际按任务列表生成同样循环；默认一致。 |
| L50 | 补24、官方262–282：stage0 AG+BG可训练，isolated关闭；后续冻结A训B | [sequential:100](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:100)开关；[loda:29](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:29)AG Parameter/buffer及BI requires_grad；stage0 AI零 | 覆盖 | test_loda103–110检查stage0参数开关。官方GAO条件dlora256–257不含task==0禁用，因此stage0使用GAO不是发现的偏差。 |
| L51 | 顺序统计应来自当前合并模型，而非MT固定theta0全任务S | [sequential:103](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:103)current_before当前模型；109–119只current+past，自身覆盖MT seen_tasks；127–130训练后raw干预重新采current | 覆盖 | 与官方dlora127–141初始化当前统计、198–203训练后当前特征相同流程；未从未来任务读取统计。缺模型状态stamp测试防止错用theta0/全任务MT缓存。 |
| L52 | 补24、官方194–219/432：训练后及merge后分别累加past两次 | [sequential:127](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:127)current_adapted；132–135merge后current_merged；previous+=两者 | 覆盖 | 官方dlora205–219两次_process_feature_mat，decomposed_lora432明确previous+=current。实现保留该double accumulation，没有替换成单累加；尚无自动集成测试。metadata引用methods/dlora.py:181–199是旧行号，准确快照位置194–219，应更新来源注释但不影响数值。 |
| L53 | 顺序stage0直接加General；后续用当前训练后S与past重标定后merge | [sequential:29](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:29)stage0Λ=1，后续current_adapted/past；[loda:186](D:/DataSpecificWork/task-vector/src/task_vector/loda.py:186)按native dtype merge | 覆盖 | 对应官方after_task304–314；gamma3/eps1e-12使用本项目统一值。保留官方核心机制，不声称逐位复现ViT优化器、classifier及dtype运行。 |
| L54 | 正式扩评需复用训练状态、保留不同split证据 | [sequential:81](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:81)路径无split；120训练无cache；174/179覆盖stage/report；[workflow:350](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:350)closure读取请求split并实际评测merged模型 | 部分；纯扩评接口缺失 | 首次all(test_formal)确实评测正式split；从test改formal再次run_ablations会重训并覆盖。已保存150行merged_projection_weights，可以base+精确QKV恢复后只评测。应增加split目录/模型与ID/锁签名、保留256结果、禁止评测入口调用train_fn。 |
| L55 | 保存重载、可恢复、安全复用 | [sequential:146](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:146)代数累计；150真实merged权重；153真实native舍入累计update；120每次调用train_fn | 部分 | 权重证据保存完整；driver没有stage完成缓存/恢复校验、独立评测入口、order锁或拒覆盖。不能套用runner.train的独立任务resume语义。应校验基座/训练数据/统计/源码/次序hash和已完成stage边界，仅恢复确切状态。 |
| L56 | 原110“小规模”顺序对照，范围需清楚 | [sequential:94](D:/DataSpecificWork/task-vector/src/task_vector/sequential.py:94)可显式train_samples但至少128；[workflow:345](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:345)默认不设，使用全部512；全部seed共48训练stage | 部分：规模未单独锁定 | 原文未给“小规模”的数值，不能判定512是硬偏差，也不能说作者明定48阶段规模。应公开记录实际train/eval数量、seed和预算。原文无E7编号；层消融/顺序是独立要求。 |
| L57 | 顺序核心机制需要自动化验收 | [tests/test_loda:103](D:/DataSpecificWork/task-vector/tests/test_loda.py:103)仅stage0开关/cyclic order；全tests未见run_sequential或_sequential_recalibrate调用 | 缺失：驱动级验收 | 补独立无训练/可注入callback的16stage小模型测试：每order fresh base、阶段继承、统计三时点、stage0 A训练、lambda来源、双累加、native BF16舍入实际update、未来数据禁读、失败恢复/防覆盖/纯扩评不训。不能把helper开关测试称完整顺序集成验收。 |
| L58 | ModelMate/Ascend BF16+FP32实际运行与数学有效性 | [modeling:23](D:/DataSpecificWork/task-vector/src/task_vector/modeling.py:23)BF16 on device；[smoke:136](D:/DataSpecificWork/task-vector/src/task_vector/smoke.py:136)QKV zero/raw/post前向检查；[training:282](D:/DataSpecificWork/task-vector/src/task_vector/training.py:282)记录memory/device | 待实机 | 代码路径及CPUtoy测试不证明NPU统计addmm、GAO backward、autocast禁用、完整Qwen生成/显存均兼容。真实raw/post/GAO、512长回答、各baseline扩rank峰值须平台证据。现有smoke明确不做科学子空间或GAO训练。 |
| L59 | 原209–211：首轮九核心状态四任务256=9216，成立后才进入分解/组合/E6 | [workflow:176](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:176)core调用默认[runner:273](D:/DataSpecificWork/task-vector/src/task_vector/runner.py:273)discovery128；[workflow:380](D:/DataSpecificWork/task-vector/src/task_vector/workflow.py:380)all先discover、lock后final | 偏差 | 必须补首轮专用编排/预先冻结的首轮协议/完成和生成数记录，随后门控推进；后期获得256结果不能追认已先做首轮。原9216只九核心状态，E1随机/initial另列。full/formal合为一个入口可行，前提是内部保留顺序阶段且顺序扩评不重训。 |
| L60 | 工程保护：原47、100及补10–11的MT默认值不应静默改成另一训练协议 | [config:51](D:/DataSpecificWork/task-vector/src/task_vector/config.py:51)numerical检查未列a_scale/epsilon；[training:141](D:/DataSpecificWork/task-vector/src/task_vector/training.py:141)直接使用train_general_a/isolated_enabled | 部分：非默认配置保护 | 默认配置正确，不能据此判主算法已经偏离；但无supplement_label添加train_general_a=True/isolated_enabled=False或更改a_scale/epsilon仍可能通过校验。建议将MT入口限制与顺序内部开关分开校验，补显式偏离拒绝测试。此项为防止静默偏离的工程保护，不新增科研实验要求。 |

## 数学核验说明

实际BA内积使用

`<B1 A1, B2 A2>_F = sum((B1.T @ B2) * (A1 @ A2.T))`。

代码对所有分支term成对求和，因此General/Isolated交叉项与组合新增rank均保留。它对任意可逆因子变换`B→BQ, A→Q^-1 A`保持不变。相反，拼接A/B后算cosine会随Q改变；实际代码没有采用这种错误方法。

设各原始任务表示列为X，列norm为D，归一化任务矩阵为Z=X D^-1。代码用小Gram `Z.T Z=R E R.T` 得到相同主子空间，并计算原始X的投影权重 `D^-1 R_q E_q^-1 R_q.T D^-1 (X.T X)`；所得shared为`U_q U_q.T X`。这解释了为什么源码没有显式构造巨大的参数展平矩阵，却仍符合原文SVD定义。

随机shared控制中的q是任务表示子空间维数。参数空间null先保持每个BA模块的奇异谱/矩阵rank，随机化左右奇异子空间，再构造共同q维span并按真实shared范数匹配。这里存在两种rank，审阅结果时必须分别标注。代码提供了符合该null定义的路径，但缺直接自动验证，且原文并未指定null的采样分布/重复次数，不能将当前一次随机draw当作已经证实共享能力的充分统计证据。

顺序真实累计update是native BF16权重逐stage合并后相对original theta0的差，未用代数factor和冒充精确merged模型。两份产物都保存：`cumulative_update.pt`是代数和；`actual_cumulative_update.pt`与`merged_projection_weights.pt`保存实际舍入结果。正式重评应恢复后者，不能仅安装代数和假装逐stage合并舍入不存在。

## 官方实现与项目适配的准确边界

| 环节 | 固定官方快照 | 当前项目 | 审计判断 |
|---|---|---|---|
| 统计 | decomposed_lora193–206按token二阶矩mean | subspaces110–146相同统计形式，128prompt每条≤64token | 公式一致；语言prompt抽样为原文适配 |
| General/Isolated | 官方373–425，PSD SVD、Cholesky+映回QR、A/√3 | subspaces166–206、245–252 | 主子空间数学等价；trace正则原文明定 |
| 重标定 | 官方66–104：1−eta，past/current，merge_gamma来自args | MT固定theta0/other，lambda3；顺序使用训练后current/past | MT差异用户已批准，顺序保留来源语义 |
| GAO | dlora269–299逐LoRA张量归一化、负rho、顺序双step | gao57–85；生成任务sample-disjoint；AdamW、外clip1 | 保留核心；优化器/clip来自本项目原文 |
| stage0 | decomposed_lora262–267训AG+BG、禁AI/BI；304–307直接加General | sequential100–102、36、loda参数开关 | 一致；官方不存在按task0自动禁GAO规则 |
| past累加 | dlora205–219两次process，decomposed_lora432每次相加 | sequential127–135 adapted及merged两次 | 一致；未覆盖完整driver自动验收 |
| 原生模型 | 官方ViT fused等宽QKV、cosface分类器、SGD/Adam、epoch调度 | Qwen异宽QKV原生hook、回答CE、AdamW线性调度、checkpoint NLL | 原文要求的语言模型适配，不声称逐位ViT复现 |

## 建议补充的最小自动验收集合

1. **S-cache**：基座、数据正文、prompt模板/统计源码任一改变，即使ID相同也拒绝旧统计；不改变时跨任务/seed复用固定S。
2. **BA/SVD**：多模块、多term、异宽QKV的dense BA Gram/夹角/归一化SVD投影/重建/gauge不变；零和近退化矩阵。
3. **Null**：activation与parameter的q0/1/2、span有效rank、每模块rank/谱和总norm、同seed复现、不同seed随机方向、parameter gauge不变。
4. **Contrast**：shared与random_shared、group与random_group按同ID/同group配对bootstrap；原始指标单位、零std、随机控制base一致性。
5. **Group production**：实际run_discovery/group_final而非仅helper，确认六任务+跨seed稳定门控、global residual输入、locked q、全部source×target和随机控制。
6. **Sequential production**：注入fake train/collect/eval callback检查fresh base、阶段继承、三次采样与double accumulation、实际BF16舍入、无未来数据、16stage输出、故障恢复、禁止覆盖、纯扩评不调用训练。
7. **First round**：仅seed17九核心状态×四任务×256先完成并留存9216核心生成清单；随机/initial额外分项；未通过时后续选择/组合/E6均不得触发。
8. **NPU验收**：真实Qwen3加载与全部28层异宽hook；统计、两次GAO step、完整回答反传、保存/重载raw/post前向/生成等价；阶段merged模型恢复一致与实际峰值显存。不把CPU测试结果当作此项通过。

## 审计源码快照hash

下列hash在只读审计末尾取得，后续源码修改会使本报告行号/结论需要复核。

```text
src/task_vector/loda.py             097E7E3F8B3EC77C599BB40885B91D33085152057C72188818ECAA40962ED5DF
src/task_vector/subspaces.py        D0B81DCE5C893799517F31D095EB0F6970B9E47FDD0820E464D7FD5E9C0D8EA2
src/task_vector/gao.py              EF9AF70FAF1E08B4D0DF6BA9396E051721BAD831B2613E2EC46BB7C009FBDB29
src/task_vector/sequential.py       B8CC882D1F167350883F107299BCAC755CE27DF4893804F3EA19663D7BE02DD2
src/task_vector/analysis.py         305EC42F36C8C048B95F8708631137A4AD6415B529283653158EBFF7C5EE77AE
src/task_vector/representations.py  624D84FEC2EB33D07F9F63593E67C5A8DE2009DB18D7B70848ABA79750EE8E9E
src/task_vector/runner.py           EA16E1835F7407095E1DA7D96838F21052C6C577AE416248B104775E92A2CC72
src/task_vector/workflow.py         7BFDB3CACDF1179B35DDC712063EECE316BB057678B7AE422C96AC12650EC1D0
src/task_vector/training.py         4B08EACADD000421146FB550C95E1331CFC4F29B72B2D0B5075C7A448D7066E9
```

本报告不修改实验协议、不推定任何共享能力结论，也不以新增测试建议替代真实代码缺口修复。算法主体“覆盖”和整体实验“完整符合原文”是不同判断；L54–L59与L02/L40应在宣称完整符合之前明确处置。
