from pathlib import Path
import csv
import hashlib
import json
import re
from collections import Counter

SRC = Path(r'D:\DataSpecificWork\task-vector')
OUT = Path(__file__).resolve().parent
DELIVERY = OUT.parent
PAYLOAD = DELIVERY / 'modelmate-runbook' / 'payload'

def ref(name, line=1, label=None, base=SRC):
    path = base / name
    assert path.is_file(), path
    assert 1 <= line <= len(path.read_text(encoding='utf-8').splitlines()), (path, line)
    return f'[{label or path.name + ":" + str(line)}]({path.as_posix()}:{line})'

def code(name, line=1):
    return ref('src/task_vector/' + name, line)

def evidence(*items):
    return '；'.join(code(name, line) for name, line in items)

C='默认路径已覆盖'
P='部分覆盖/有缺口'
O='原文可选，未加入'
E='经授权平台替换'

# Each original requirement retains its exact wording from the source traceability table.
# Status describes implementation coverage, never scientific success or NPU validation.
DETAILS = {
'R01':(P,[('workflow.py',194),('runner.py',325),('runner.py',400),('runner.py',510)],'两空间、共享、组合、留出和混淆控制均有实质调用；A01/A04/A05/A06使完整执行与证据闭环尚未成立。'),
'R02':(C,[('reporting.py',28),('runtime.py',82)],'未获得真实训练/科学结果；历史文档中的过强覆盖表述须以本审计纠正。不能把实现或CPU测试写成实验有效。'),
'R03':(C,[('modeling.py',7),('prompts.py',73)],'默认Qwen3-1.7B、BF16、nonthinking，检查28层/2048及GQA尺寸；真实模型/tokenizer和NPU仍待验。'),
'R04':(C,[('data.py',262),('metrics.py',42)],'保留完整官方解题过程+答案；完整生成后按最终数值评分。正式数据来源、长度与容量待验。'),
'R05':(C,[('data.py',268),('metrics.py',59)],'OWA、深度0–3、T/F/U、theory分组；Unknown保留合法负QDep并核来源深度。真实官方文件待验。'),
'R06':(C,[('data.py',309),('metrics.py',131),('metrics.py',219)],'en→de；corpus chrF++主、BLEU辅，有SacreBLEU签名。原始IWSLT目录/年份需在资产清单核实。'),
'R07':(C,[('data.py',315),('metrics.py',166),('metrics.py',229)],'QQP仅正例；iBLEU、源/参考语义cosine、复制率与词重叠均实现。真实离线语义模型尚未验证。'),
'R08':(C,[('metrics.py',181),('code_execution.py',61),('evaluation.py',62)],'单次greedy代码，CPU子进程执行全部测试；任一失败/超时不得通过。不是pass@k。'),
'R09':(C,[('code_execution.py',164),('data.py',430)],'参考程序先通过、确定性变异必须失败、生成修复再执行；保留验证链。错误类型分布需实测报告。'),
'R10':(C,[('analysis.py',214),('workflow.py',219)],'功能层次聚类允许单组/无稳定组，不预设推理/转换两组；六任务稳定条件不成立时不做组内SVD。'),
'R11':(C,[('data.py',340),('data.py',482),('data.py',533)],'默认512/128/128/256，前三类来自train且组隔离，两路线共用固定文件。A09：范围校验仍可被显式override绕过。'),
'R12':(C,[('data.py',495),('workflow.py',278)],'test_formal=min(1000,可用)或all，支持锁定后扩评；formal含初始256，不能自动视作新的独立确认。A02为入口范围缺口。'),
'R13':(C,[('workflow.py',154),('workflow.py',176),('workflow.py',116)],'默认17→29→43，逐seed有效性/方向一致性门存在；但门使用discovery128，首轮256缺口见A01。'),
'R14':(C,[('data.py',87),('data.py',414),('data.py',476)],'官方split声明、来源hash/revision、独立有标签评测及跨来源去重实现；hash不能单独证明任意本地文件的官方身份。'),
'R15':(C,[('data.py',251),('data.py',275),('data.py',476)],'ProofWriter同theory、QQP重复问题连通组，官方评测组优先，train同组剔除；真实重合率待验。'),
'R16':(C,[('data.py',323),('data.py',430),('data.py',508)],'代码生成/修复共用原program_id和所有split ID；先求长度与程序可验证交集再联合划分。'),
'R17':(C,[('data.py',74),('data.py',482),('data.py',493)],'采用已批准300/37/37/256，容量不足明确停止并保存拒绝/容量记录。374恰好耗尽，正式过滤后是否足够尚未验证。'),
'R18':(C,[('ltv.py',74),('ltv.py',98),('training.py',127)],'层14整decoder block输入pre-hook，在RMSNorm前；只训练2048维向量，基座冻结。'),
'R19':(C,[('ltv.py',100),('training.py',130),('training.py',147)],'默认各task/layer独立U(-.1,.1)，FP32主参数。A09：暴露的init_low/high未接线，显式改值静默无效。'),
'R20':(C,[('loda.py',25),('loda.py',83),('training.py',141)],'MT默认28层QKV、8+8，A buffer冻结、B零且仅B优化；顺序stage0另有明确开关。A09需限制MT误开顺序开关。'),
'R21':(C,[('losses.py',36),('training.py',62),('evaluation.py',27)],'完整回答+EOS、causal shift、样本内token均值再样本均值；micro按样本数加权。有人工loss/梯度历史测试。'),
'R22':(C,[('training.py',149),('gao.py',48)],'AdamW lr.001；LTV wd.01、LoDA wd0；LoDA双向GAO。真实NPU优化器兼容性待验。'),
'R23':(C,[('training.py',247),('training.py',266)],'每epoch tune回答NLL，最多5epochs、patience2，恢复最低NLL状态后导出；不按test/accuracy选点。'),
'R24':(C,[('training.py',52),('training.py',62),('training.py',225)],'LTV micro2有效16；LoDA逻辑16近半分组、每组micro2。尾批按批准协议不丢不复制，实际ID/step记账。'),
'R25':(C,[('training.py',153),('training.py',217),('gao.py',79)],'linear、warmup0、clip1；GAO外clip独立于内部逐参数扰动归一化；每逻辑batch两次实际调度步。'),
'R26':(C,[('prompts.py',81),('data.py',457)],'prompt512、完整target含EOS512，合计≤1024；超长过滤并记录，全预定variant取交集，答案不截断后训练。'),
'R27':(C,[('evaluation.py',116),('runner.py',277)],'目标任务prompt/指令固定，仅换源表示；argmax greedy，主干预强度1。调强度结果另列。'),
'R28':(C,[('evaluation.py',152),('metrics.py',241)],'数学512/逻辑32/翻译256/改写128/代码512；保存输出、token、EOS/长度停止、截断率。顺序汇总遗漏见A03。'),
'R29':(C,[('losses.py',20),('ltv.py',52),('training.py',69)],'固定原prompt末位置p−1，teacher forcing不会移到答案末；变长/缓存CPU测试存在。'),
'R30':(C,[('ltv.py',33),('evaluation.py',83)],'仅prefill注入；cache_position/已占用cache门控；无cache参照固定原位置。真实NPU长生成待验。'),
'R31':(C,[('workflow.py',333),('runner.py',83),('ltv.py',101)],'主层14复用独立主实例；6、22和[6,14,22]分别训练，多层三个独立参数联合优化。formal-only未纳入见A02。'),
'R32':(C,[('training.py',149),('losses.py',36),('ltv.py',74)],'官方lr/单标签loss/前后hook差异已按原文统一为.001、完整回答、block前；有本地官方快照及说明。'),
'R33':(C,[('runner.py',118),('loda.py',115),('sequential.py',49)],'MT只安装/卸载更新，独立从theta0训练；顺序单独merge且单独route，不冒充独立向量。A07/A08影响来源/恢复。'),
'R34':(C,[('subspaces.py',68),('subspaces.py',101),('subspaces.py',146)],'训练中固定128条、每条均匀≤64非pad prompt token；收全部28层原生QKV输入二阶矩token均值。A16缓存指纹不全。'),
'R35':(C,[('subspaces.py',166),('subspaces.py',174),('subspaces.py',225)],'General TopEig(sumS)，Isolated使用S_other和Cholesky白化→eig→映回→QR；未擅自从UG投影去除Isolated。'),
'R36':(C,[('loda.py',29),('training.py',225),('gao.py',50)],'MT A冻结/B训练；两组ID唯一且不相交，以样本分组适配生成任务，非类别分组。'),
'R37':(C,[('loda.py',148),('training.py',267),('runner.py',180)],'raw/post均保存；实际ΔW=.5 BG Λ AG+BI AI，post主评测，分支缩放保留。真实重载训练状态验收仍待。'),
'R38':(C,[('gao.py',55),('subspaces.py',181),('subspaces.py',209)],'rho_max.05、lambda3、trace正则1e−4；数值jitter独立记录，不混成干预强度或Lambda eps。'),
'R39':(C,[('gao.py',59),('gao.py',67),('gao.py',81)],'逐参数张量归一化、负rho一次抽取、两个顺序方向/step、无二阶、finally撤扰动；有作者语义参考历史测试。'),
'R40':(C,[('modeling.py',25),('loda.py',39),('loda.py',115)],'Q2048/K1024/V1024各B，共享输入A；只对原生线性输出加增量，保留GQA/QKnorm/RoPE。真实模型待验。'),
'R41':(C,[('sequential.py',22),('sequential.py',100),('sequential.py',176)],'四循环顺序，每order新基座，stage继承实际合并模型，有独立JSON报告；原文“小规模”未指定数值。A03：完整生命周期和扩评证据不足。'),
'R42':(O,[('suite.py',6)],'未加入普通独立LoRA。原文写“如加入”，因此不是功能遗漏，也不能用它替代LoDA-MT。'),
'R43':(C,[('runner.py',273),('representations.py',131)],'BASE/正确/随机同范数/initial完整；LoDA initial为B=0，等价基座且标签保留。首轮阶段不满足见A01。'),
'R44':(C,[('runner.py',311),('analysis.py',132),('workflow.py',255)],'LTV cosine；LoDA实际BA cosine与每QKV输入/输出子空间夹角，initial/cross-seed/cross-space关系均有保存；零initial几何为undefined。'),
'R45':(C,[('runner.py',273),('analysis.py',202)],'每源到每目标相对BASE的有向4×4/扩展6×6原单位矩阵及列标准化矩阵；首轮256执行先后仍缺。'),
'R46':(P,[('runner.py',325),('analysis.py',100),('runner.py',548)],'shared/residual/reconstruction/delete_shared及随机同秩均有；delete_shared明确alias residual。A04：缺shared−random和group−random直接配对区间。'),
'R47':(C,[('runner.py',400),('runner.py',462),('joint.py',14)],'original/sum/mean/weighted/decomposed同候选预算，raw/matched范数双面板，含原任务保留和真正双任务联合输入。A06冻结候选摘要未强制核验。'),
'R48':(C,[('runner.py',70),('workflow.py',315),('runner.py',510),('runner.py',531)],'四主任务逐一留出，LoDA每折重建S/A并重训B，源任务选择结构/系数；SAMSum仅零样本最终评估。ROUGE元数据缺口见A11。'),
'R49':(C,[('analysis.py',202),('metrics.py',238)],'保留准确率/chrF++/iBLEU等原单位，禁止异质原分直接平均；选择时平均标准化增益且保留原分。'),
'R50':(C,[('analysis.py',172),('analysis.py',202),('runner.py',266)],'目标BASE逐样本std列缩放，单任务组级paired bootstrap每次重算corpus；零std不造相关。联合组依赖限制A15单列。'),
'R51':(P,[('runner.py',369),('workflow.py',194),('workflow.py',278)],'发现集功能聚类/稳定性已实现，final生成相应矩阵；A05：没有自动汇总“发现关系在test是否保持”的确认表/结论证据。'),
'R52':(C,[('analysis.py',75),('runner.py',325)],'均值s和v−s作为独立method基线，非只实现SVD。'),
'R53':(C,[('analysis.py',49),('analysis.py',75)],'归一化列Gram谱解与SVD数学等价，投影原始未归一化表示；s+r重建。无需物化巨大展平矩阵。'),
'R54':(P,[('config.py',68),('runner.py',325),('config.py',95)],'q0/1/2只在discovery选，final传locked选择；A06：实际训练/候选表示字节未被最终锁完整冻结。'),
'R55':(C,[('representations.py',83),('analysis.py',49),('baselines.py',39)],'LoDA按全部BA实际矩阵计算内积、谱分解和merge，包含分支交叉项；不拼接A/B坐标。'),
'R56':(C,[('workflow.py',219),('workflow.py',239),('runner.py',548)],'六任务且各seed稳定分组相同才触发；先去global再组内残差SVD，含random_group，不强迫触发。'),
'R57':(C,[('analysis.py',60),('runner.py',489),('reporting.py',41)],'保留effective rank与General/Isolated/full raw/post功能；未以rank≤4或相同UG自动宣称能力共享。科学解释仍须实际结果。'),
'R58':(C,[('analysis.py',40),('runner.py',435),('representations.py',122)],'平均、权重与范数匹配齐备，标实际不同候选数和退化；q0=sum等同不虚称新算法，不静默截回rank16。'),
'R59':(P,[('prompts.py',32),('prompts.py',100),('workflow.py',291)],'固定表示下输出JSON/prompt改写/逻辑重编码均重评；A11：单键answer指令与允许额外键的解析判据不一致。'),
'R60':(C,[('config.py',81)],'两必需官方实现固定commit的选定原始源码快照已存third_party，可追来源；不是完整git clone。LoDA缺LICENSE源站404已记录，不影响已存算法文件。'),
'R61':(C,[('data.py',87),('data.py',533),('runtime.py',82)],'transformers4.51.3及datasets/指标版本有固定；安装后pip-freeze，框架2.8保留；正式资产加载和完整环境锁证据尚待ModelMate。'),
'R62':(C,[('baselines.py',107),('baselines.py',132),('baselines.py',169),('runner.py',415)],'TIES/TSV作者orthogonalization变体/KnOTS真实参数增量均接入LoDA组合；官方函数数值对照历史测试存在，未对LTV错套参数merge。'),
'R63':(P,[('loda.py',25),('config.py',30),('training.py',135)],'ViT→Qwen语言生成适配正确；A09：schema未覆盖全部科学范围，LTV初始化配置静默忽略、MT可显式误开顺序配置。'),
'R64':(C,[('cli.py',51),('packaging.py',8)],'建议结构职责由更细模块实现，无qwen_hooks.py等字面文件不算漏功能；A13：原package/source_manifest未纳入新ops/根包装入口。'),
'R65':(E,[('runtime.py',17),('runtime.py',64),('training.py',186)],'按用户授权改为Ascend910B/8NPU ModelMate；BF16、micro2、非重入保留。实际单卡容量/峰值/完整长回答可运行性待验。'),
'R66':(P,[('workflow.py',176),('runner.py',273),('workflow.py',376)],'A01：八独立训练+基座都有，但core先做discovery128，分解选择后才final；缺原文严格首轮256阶段。'),
'R67':(P,[('suite.py',31),('runner.py',273),('reporting.py',21)],'A01：9216只是inventory预算，不是实际完成清单；需九核心状态×4×256独立记录。随机/initial、发现/调参应另计。'),
'R68':(P,[('smoke.py',59),('smoke.py',136),('workflow.py',116),('training.py',194)],'有工程smoke、有效性及跨seed门；A01/A06/A07/A08/A14尚未闭环。真实Qwen/NPU/GAO重载、恢复和科学有效性全部待验。'),
}

INFERENCES = {
'I01':(C,[('training.py',186),('ltv.py',135)],'非重入重算处于同一干预上下文；CPU小模型loss/梯度测试存在，NPU待验。'),
'I02':(C,[('runner.py',70),('runner.py',139),('workflow.py',315)],'每折只看源任务统计、LoDA重新构建/训练；LTV独立向量复用符合数学定义。'),
'I03':(P,[('suite.py',31),('reporting.py',21)],'预算口径已说明，但没有首轮9216真实清单和分项完成验收，见A01。'),
'I04':(C,[('loda.py',42),('runner.py',273)],'未训练B=0等价基座，保留initial标签；零更新CPU与smoke检查存在。'),
'I05':(C,[('data.py',482)],'MBPP已采用300/37/37而非两个128；容量不足停止。'),
'I06':(C,[('joint.py',14),('joint.py',48),('runner.py',462)],'真实双子任务输入/目标、分别评分及联合达标，非只在原任务测试就宣称组合能力。'),
'I07':(C,[('runner.py',311),('workflow.py',255)],'跨LTV/LoDA比较任务关系矩阵，未计算不同维原向量直接cosine。'),
'I08':(C,[('analysis.py',92),('runner.py',351)],'delete_shared明确为residual alias；s+r代数重建。实际完整Qwen行为仍待验。'),
'I09':(C,[('representations.py',101),('runner.py',414)],'组合保留全部因子项及实际矩阵，不静默压回rank16；更大rank的峰值待NPU实测。'),
'I10':(P,[('data.py',499),('runner.py',237),('config.py',95)],'有独立组隔离confirmation reserve及暴露ledger，但A01尚未接首轮；formal含已暴露256，后改方法不能追认为新独立确认。'),
}

FINDINGS = [
('A01','P1 · 原文明确阶段偏差','首轮256/9216没有独立执行阶段',
 'workflow.run_core调用core_panel时省略split，实际默认discovery128。all先run_discovery选择分解/组合，再lock和final(test或test_formal)。--phase core --split test也不会传入split。suite仅列9216预算；report计数又明确只是保存面板量，不能证明物理生成或首轮完成。',
 [('workflow.py',176),('runner.py',273),('workflow.py',359),('suite.py',31)],
 '增加预先冻结的seed17首轮协议、256核心面板与完成清单/门；随后29/43重现、再共享/组合/E6。不能只把默认split改test，因为现test锁依赖后续discovery。9216仅九核心状态，额外E1随机/initial另计。'),
('A02','P2 · 完整扩评编排/使用要求','四个入口没有同一条完整阶段链',
 'full最终用test256；full-formal能首次把主线、层消融及顺序评测放到test_formal，但跳过独立首轮256。formal-only复用锁后只扩主MVP/expanded/四LOO等final面板，未扩层消融和顺序对照。所有控制都做formal1000是此前使用需求的完整性要求；原文没有逐个规定每个控制必须1000。',
 [('workflow.py',278),('workflow.py',333),('workflow.py',376)],
 'full与formal可合成一个总入口，内部必须保留“首轮256→跨seed门→发现/锁→最终/正式→全部已约定控制”的独立阶段，并共用已训练产物。不是把两个split简单别名化。'),
('A03','P1/P2 · 顺序对照生命周期缺口','顺序算法存在，纯扩评/安全恢复/统一证据不足',
 '四order、stage0可训AG/BG、之后冻A、训练后及merge后双统计累加、真实BF16合并、独立report.json均存在。驱动每次调用重训，输出路径不含split，同名stage/report会覆盖；无纯评测入口、stage完成恢复和order锁。workflow回调直接评测，绕过runner.evaluate的暴露ledger/cache/runtime；结果仍放discovery_metrics，顶层report不扫描sequential。',
 [('sequential.py',49),('sequential.py',120),('sequential.py',150),('sequential.py',174),('workflow.py',350),('reporting.py',10)],
 '利用已存merged_projection_weights恢复实际合并模型，拆训练与split评测，分别保存256/formal结果、锁和暴露记录，并加入独立汇总。顺序不应因扩评重训；不能用累计代数因子和冒充含BF16逐阶段舍入的模型。'),
('A04','P2 · 原文功能证据分析部分缺失','缺shared超过随机同秩的直接配对比较',
 'shared、random_shared、group、random_group都生成并保留逐样本结果，目前只分别计算各条件减BASE的paired CI。两个相对BASE的区间不能替代shared−random_shared或group−random_group的直接配对区间。不是缺随机条件，也不是缺shared算法。',
 [('runner.py',349),('runner.py',570),('runner.py',266)],
 '按同ID/组直接重采样输出条件差及CI，保留原指标单位。不擅加原文未规定的“显著性通过阈值”。'),
('A05','P2 · 原文确认分析部分缺失','缺发现关系在最终测试的明确确认汇总',
 'discovery有功能聚类与跨seed/跨空间关系；final有原单位迁移矩阵、锁定成分结果。report主要汇总文件数、门控和锁，未形成发现→test关系保持/改变、组稳定性和负结果的确认表。现有raw结果可供人工离线分析，但自动实验交付尚未闭环。',
 [('workflow.py',255),('workflow.py',278),('reporting.py',7)],
 '不在test重新选q/组，比较已冻结发现关系与确认结果，并给出逐任务/跨seed证据及失败情况。确认图表形式本身可选，确认分析不应省略。'),
('A06','P1 · 锁定完整性','实际训练与组合表示未被最终锁完整冻结',
 'lock固定config/data/sources/discovery decisions及模型/语义资产；不统一固定post/raw/派生表示。E5记录representation_digest，final按路径加载却不比较；首次final前替换同基座表示可以在旧lock下开始新评测。已有cache签名能抓后续缓存变化，不能替代第一次测试前冻结。',
 [('config.py',95),('config.py',113),('runner.py',445),('runner.py',478),('runner.py',214)],
 '所有主、候选、成分及依赖表示摘要进入lock，test读样本前逐项核验；已有candidate digest必须强制比较。'),
('A07','P1 · 长作业恢复','完成JSON与导出来源信息存在中断窗口',
 'train_task先写training.json；runner随后补request_signature再补raw/post/initial元数据。中断在前者时缓存因缺签名拒绝恢复last.pt；中断在后者时可被当完成，评测又因表示缺base_fingerprint失败。不是正常每逻辑batch checkpoint未实现。',
 [('training.py',284),('runner.py',130),('runner.py',151),('runner.py',153)],
 '导出及三份元数据核验完毕后再原子发布完成标记，残缺提交可恢复导出。覆盖两个中断窗口的故障注入验收。'),
('A08','P1/P2 · 恢复来源','部分checkpoint来源绑定不足，跨新目录恢复缺initial',
 'last.pt签名主要模型配置/路径、训练设置和样本ID，未绑定实际基座、数据正文/manifest、源码和子空间内容。较强runner签名只用于完成JSON。另显式从旧checkpoint恢复到新目录时不导出initial.pt，最后runner加载初始化文件失败；同目录默认恢复不受后一边界影响。',
 [('training.py',158),('training.py',167),('training.py',194),('runner.py',126)],
 '在checkpoint中固定强请求签名和所有依赖；恢复先核验内容。携带或校验初始化表示，或训练前明确拒绝不支持的跨目录组合。'),
('A09','P2 · 配置防偏差','默认正确，完整范围校验与有效字段接线不足',
 'validate_config锁了一批训练/生成数值，但未完整锁counts、主层、seed顺序、代码扩展、LOO范围、必要variants、A尺度及MT顺序开关等。只读探针8种偏离均无需supplement_label便接受。ltv.init_low/high已暴露却未传入adapter，固定U(-.1,.1)，显式改值静默无效。',
 [('config.py',30),('training.py',135),('ltv.py',105)],
 '主协议必需值/集合完整校验，偏离需明确补充；公开字段要么接线记录有效值，要么拒绝。默认未偏离，不能把这个风险写成默认参数错误。'),
('A10','P2 · 作业状态','普通CLI门控失败仍退出0',
 'cli.main无条件return0，即使run返回completed:false。新modelmate_job helper会映射失败码，原deploy/run_modelmate.sh等普通入口未获得同等保护。本轮在重型runner/workflow边界替身下执行真实CLI，确认打印completed:false同时返回0；这不是一次真实训练失败。',
 [('cli.py',108)],
 '统一入口按失败/未完成返回非零并保存明确失败原因，保证平台不会把未完成的科学阶段记为成功。'),
('A11','P2 · 评分/表面控制小缺口','ROUGE元数据缺失，单键JSON判据不严',
 'ROUGE1/2/L stemmed F1×100计算存在，但summary signatures没有rouge-score版本、stemmer/键/F1定义，未满足公开补充“每次输出评分签名/版本”。output_format指令要求single key answer，解析却允许额外键。',
 [('metrics.py',188),('metrics.py',214),('prompts.py',60),('prompts.py',102)],
 '补齐ROUGE可复核签名；严格单键或明确修订格式判据并记录format_valid。不是整套摘要或格式控制没实现。'),
('A12','P2 · 部署输入接口','CSV/TSV不是所有任务通用支持',
 'CSV DictReader返回字符串；MBPP test_list被list(string)拆字符，ProofWriter嵌套字段未通用JSON解码。新指南/配置工具接受这些后缀的泛化描述过宽。HF DatasetDict、JSON/JSONL、保类型Parquet仍可覆盖原文各数据集。',
 [('data.py',65),('data.py',324)],
 '按任务列格式支持矩阵，或显式解析嵌套字段并验证schema；不能因加载了文件就宣称官方数据成功准备。'),
('A13','P2 · 新交付复现','原打包器和source_manifest遗漏新平台辅助入口',
 '新交付ZIP已包含ops和根包装入口；原CLI package白名单未包含它们，再打包会丢失新操作入口。source_manifest同样不覆盖ops/根包装脚本。新helper记录自身摘要提供追踪，但未统一成为协议锁依赖。当前82科学源码与两份快照完全一致。',
 [('packaging.py',8),('config.py',81)],
 '将新helper/包装脚本纳入交付manifest和相关阶段依赖指纹，补可重建包内容验收。不是当前ZIP缺少这些文件。'),
('A14','P2 · 验收证据边界','顺序驱动没有集成测试，真实LoDA/NPU smoke覆盖有限',
 'test_loda的顺序测试只检查stage0开关及循环顺序，没有调用run_sequential；历史“顺序驱动已集成验收”的概括不成立。smoke的LoDA部分用合成A和固定随机B检查前向/保存，没有真实统计+GAO更新+优化器恢复；random_subspace_control、非平凡稳定聚类也缺直接生产接线验收。',
 [('smoke.py',136),('sequential.py',49)],
 '为高风险生产调用补有限关键验收，真实Qwen/Ascend验证另行取得；不把每个建议单测当成原文新增实验。历史84通过不能覆盖未测路径。'),
('A15','统计解释限制 · 非原文明定算法缺失','联合输入CI未完整保留两路组依赖',
 'joint group_id拼接A组与B组后使用单组bootstrap。如果同一theory或QQP连通组与多个另一侧组配对，拼接组不同，共享子组依赖被拆开；联合点估计仍正确，CI可能过窄。原文没有指定多路组bootstrap，不把建议方法升级成强制算法。',
 [('joint.py',41),('analysis.py',182)],
 '在补充协议中锁定保留两路依赖的联合重采样，或限制配对并明示CI假设；保存两侧group ID以便复核。'),
('A16','P2 · 统计缓存来源','固定基座统计缓存未绑定完整输入/实现',
 '缓存签名包含base/loda/maxprompt/train IDs，却没有实际prompt/target或prepared manifest内容hash、模板/统计源码hash。同ID正文或模板变动时，旧S可以被错误复用。更上层完成训练请求检查不能为首次新训练补上此缓存检查。',
 [('runner.py',97)],
 '统计缓存签名纳入真实token输入或内容manifest与实现指纹；变更时拒绝旧缓存并新run，不能凭ID不变认定特征不变。'),
]

def load_original_rows(prefix):
    rows=[]
    for line in (SRC/'docs/需求追踪与验收矩阵.md').read_text(encoding='utf-8').splitlines():
        if re.match(r'^\| '+prefix+r'\d{2} \|', line):
            rows.append([c.strip() for c in line.strip('|').split('|')])
    return rows

original=load_original_rows('R')
inferred=load_original_rows('I')
assert {r[0] for r in original}==set(DETAILS)=={f'R{i:02}' for i in range(1,69)}
assert {r[0] for r in inferred}==set(INFERENCES)=={f'I{i:02}' for i in range(1,11)}
counts=Counter(d[0] for d in DETAILS.values())

lines=['# 原始实验设计与现有代码逐项一致性审计','',
'审计日期：2026-10-10。结论适用于本次核对的代码快照。**不能确认“没有功能缺失”：默认主要算法与E1–E6均有实质实现，但存在明确阶段偏差、证据分析缺口以及恢复/锁定问题，尚未达到完整验收。** 本轮只核查并编写报告，未修改实验源码、安装依赖、删除文件、加载正式模型或提交训练。','',
'## 审计范围和证据','',
'原始微信文档与项目原文副本逐字节一致，SHA256均为 `D689BC2CB01A9DFE88B0F42CE2B385AC43AC0A2633DEA53E582882E3CCFD2A5C`，共211行。核对时以原文和人类批准补充为准，旧需求矩阵只提供R编号，旧“已覆盖”表不当作实现证据。',
'',ref('docs/原始实验方案.md',1,'原文副本')+'；'+ref('docs/补充协议与实现选择.md',1,'公开补充与实现选择')+'；'+ref('docs/需求追踪与验收矩阵.md',1,'既有需求编号')+'。','',
'审计沿数据→模型/干预→损失/优化→表示→选择/锁→生成/评分→统计/报告的实际调用链进行，并对照三个独立领域报告。原源码、旧交付包与新payload中的82个原始文件全部hash一致，因此算法判断同时适用原项目与现交付快照；新ops/helper另列工程审计。','',
'本轮执行证据见 '+ref('audit-checks.json',1,'audit-checks.json',base=OUT)+'：50个Python文件按3.10语法解析/编译通过；默认配置校验通过；8类未经补充的范围偏离被接受；真实CLI在重型边界替身下对completed:false返回0。该探针没有导入torch，没有训练或伪造科学结果。','',
'已重读两份历史JUnit：同一套84项分别在CPU torch2.8.0与2.14.1通过，均为0失败/0错误（版本来自既有本地验证报告，本轮未重装核验）；**本轮未重新执行任何测试套件**。当前本地没有torch可用。正式Qwen3-1.7B、七任务数据、真实语义模型、Ascend/CANN和所有科学效果均未实机验收。','',
'## 如何理解覆盖判定','',
'- “默认路径已覆盖”：默认实现与文档在代码层面对应；不等于真实数据可运行或结论成立。跨项工程风险仍列在该行。',
'- “部分覆盖/有缺口”：已有主功能，但要求中的阶段、分析证据、锁定或配置语义尚未完整实现。',
'- “原文可选，未加入”：普通独立LoRA可选，未加入不算遗漏。',
'- “经授权平台替换”：原24GB NVIDIA容量规划被用户指定Ascend ModelMate覆盖，不算擅自变更模型/精度。','',
'| R编号覆盖类别 | 项数 |','|---|---:|']
lines += [f'| {status} | {counts[status]} |' for status in (C,P,O,E)]
lines += ['', '计数单位是68条既有要求，不是68个独立功能，也不是通过率。一个跨项问题可能影响多条要求；没有另把全部“待实机”误计为算法缺失。以下原文可选/建议仍保持其原有性质。','',
'## 全部原文要求：R01–R68','',
'| ID / 原文行 | 原文要求 | 性质/阶段 | 覆盖状态 | 实际实现入口 | 对照结论、限制与缺口 |',
'|---|---|---|---|---|---|']
records=[]
for r in original:
    rid,origline,requirement,kind,*_=r
    status,locations,note=DETAILS[rid]
    ev=evidence(*locations)
    extra = {
        'R03':[('configs/experiment.yaml',10)],
        'R11':[('configs/experiment.yaml',18)],
        'R17':[('configs/experiment.yaml',24)],
        'R31':[('configs/experiment.yaml',111)],
        'R60':[('third_party/official_sources.json',1)],
        'R61':[('pyproject.toml',1),('deploy/setup_modelmate.sh',1)],
    }.get(rid, [])
    if extra:
        ev += '；' + '；'.join(ref(name,line) for name,line in extra)
    lines.append(f'| {rid} / {origline} | {requirement} | {kind} | {status} | {ev} | {note} |')
    records.append(dict(id=rid,original_lines=origline,requirement=requirement,kind=kind,status=status,evidence=ev,conclusion=note))

lines += ['', '## 必要实施推论：I01–I10','',
'这些是由原文推导出的实施边界，不能混称原文逐字规定。','',
'| ID | 推论 | 判定 | 实现证据与结论 |','|---|---|---|---|']
for r in inferred:
    rid,requirement,*_=r
    status,locations,note=INFERENCES[rid]
    ev=evidence(*locations)
    lines.append(f'| {rid} | {requirement} | {status} | {ev}。{note} |')
    records.append(dict(id=rid,original_lines='',requirement=requirement,kind='实施推论',status=status,evidence=ev,conclusion=note))

lines += ['', '## 人类批准补充与实现选择','',
'只有以下B01–B05对应会话中明确批准内容；其余公开数值选择是代码为落地补齐的协议细节，不伪称原文已定或全部获得逐项人工批准。','',
'| ID | 已批准内容 | 当前实现 | 判定/边界 |','|---|---|---|---|',
f'| B01 | Ascend910B八NPU、CANN8.5.0-a3、torch/torch_npu2.8，DPC持久化 | {code("runtime.py",17)}；configs/experiment.yaml记录环境；部署脚本保留框架 | 平台已适配；八卡显存口径、实际版本/内存/NPU运行待检测。独立单卡任务池不等同DDP。 |',
f'| B02 | E5原任务保留+双任务联合输入、分项评分及联合成功率 | {code("joint.py",14)}；{code("runner.py",462)} | 主要功能覆盖；连续指标阈值来自BASE tune中位数并锁定，联合CI解释限制见A15。 |',
f'| B03 | MBPP300/37/37/256；E6四主任务逐一留出并测SAMSum | {code("data.py",482)}；{code("workflow.py",315)}；{code("runner.py",531)} | 默认覆盖；MBPP374过滤后容量待验；摘要评分器元数据缺口A11。 |',
f'| B04 | 统计prompt token未中心化二阶矩均值，A=basis.T/√3，B FP32 | {code("subspaces.py",110)}；{code("subspaces.py",252)}；{code("loda.py",42)} | 默认公式覆盖；真实NPU统计/FP32优化待验，缓存来源缺口A16。 |',
f'| B05 | theta0固定统计重标定lambda3；raw/post并存、post主；逐epoch NLL选点；主强度1、同预算调强度另列 | {code("loda.py",140)}；{code("training.py",247)}；{code("runner.py",301)}；{code("workflow.py",296)} | 默认覆盖；实际表示锁和保存恢复缺口A06–A08。 |','',
'公开实现选择还包括EOS监督、iBLEU α=.9、chrF++参数、ROUGE F1×100、联合1536预算/128条、聚类average linkage及稳定阈值、组内SVD跨seed相同partition门、尾批不丢不复制、顺序stage0/双统计累加、TIES/TSV/KnOTS具体变体和工程有效性门阈值。其默认路径可追踪；不能把工程门阈值当成原文的统计显著性标准。ROUGE每次评分签名尚未落实，见A11。','',
'## 确认问题及修复验收条件','',
'P1表示应在正式长作业/确认评测之前优先处理，P2表示证据或接口完整性问题；不是凭严重度猜测科学结论。以下问题均未在本轮修复。','']
for fid,priority,title,body,locations,fix in FINDINGS:
    lines += [f'### {fid} · {title}', '', f'性质：{priority}。', '', body, '', '证据：'+evidence(*locations)+'。','', '完成条件：'+fix, '']
    if fid=='A02':
        lines += ['新helper证据：'+ref('ops/modelmate_job.py',1,'modelmate_job.py',base=PAYLOAD)+'；入口说明：'+ref('ModelMate_操作指南.md',1,'ModelMate操作指南',base=PAYLOAD)+'。','']

lines += ['## 实机待验证项：不能用实现或CPU测试替代','',
'| 验证项 | 需要的证据 | 当前状态 |','|---|---|---|',
'| 环境与资产 | NPU实际型号/卡数/单卡容量、CANN/框架版本、模型/tokenizer权重hash、七任务官方来源及语义模型指纹 | 用户报告环境；未独立取得ModelMate运行证据 |',
'| 数据完整范围 | 各任务train/tune/discovery/test/formal/confirmation ID和组交集、全部variant长度、拒绝原因和实际容量；MBPP374合格性 | 准备程序有实现；真实准备尚未验收 |',
'| 两路线真实训练 | NLL曲线、FP32参数、A/基座冻结、真实统计/GAO两方向、逻辑/实际step、最佳点和故障恢复 | CPU微型历史证据；真实Qwen/NPU未验 |',
'| 保存/训推一致 | 训练后LTV/LoDA raw/post重载、prefill/cache、非重入重算、GAO优化器恢复、顺序实际merged权重恢复 | 有局部smoke入口；不足以代表完整真实训练后行为 |',
'| 资源 | 最长完整回答micro2训练/验证峰值、组合扩rank/稠密baseline峰值、八任务并行稳定性、失败/OOM记录 | 没有910B实测，不能保证任意单卡显存均足够 |',
'| 首轮及科学结果 | 实际9216核心清单、额外E1控制、17→29→43功能与迁移稳定、直接null对比、表面控制、E5/E6收益及负结果 | 全部尚无真实科学结论 |','',
'## 修复顺序与“无功能缺失”的完成标准','',
'1. 先修A01/A06–A08/A10：严格阶段、最终表示锁、可恢复提交与来源、失败作业状态；确定完整formal范围并拆开顺序训练/评测（A02/A03）。',
'2. 补A04/A05的直接功能比较和发现→测试确认分析，补A11/A12/A13/A16的元数据、接口与指纹；保持原模型、损失、样本范围与超参数原意。',
'3. 对确实高风险且此前未覆盖的路径做针对性验收（A14），为联合CI显式记录依赖假设（A15）；不把额外建议扩成未经批准的新科学实验。',
'4. 在指定ModelMate环境先真实prepare/preflight/smoke/一个完整LoDA逻辑batch及恢复，保存来源、拒绝、峰值和失败证据；再按首轮→重现→发现/锁→全套确认执行。',
'5. 每条适用R要求都要有默认实现、明确阶段入口、对应产物与验收证据；可选项明确标可选，科学负结果也应正常保存。到那时才可称代码范围完整，仍不能仅因跑完而称共享能力成立。','',
'## 附件与复核入口','',
'- '+ref('data-evaluation-audit.md',1,'数据/生成/指标/统计详细审计：D01–D35',base=OUT),
'- '+ref('model-training-audit.md',1,'模型/LTV/共同训练详细审计：T01–T34、F1–F6',base=OUT),
'- '+ref('loda-decomposition-audit.md',1,'LoDA/分解/组合/顺序详细审计：L01–L60',base=OUT),
'- [可筛选逐项矩阵CSV]('+ (OUT/'requirements-comparison.csv').as_posix() +')',
'- [机器可读审计JSON]('+ (OUT/'requirements-comparison.json').as_posix() +')',
'- '+ref('audit-checks.json',1,'本轮轻量验证、文件hash和历史JUnit读取记录',base=OUT),
'', '报告中的结论是当前快照的覆盖审计。所有确定缺口仍待修复；真实数据与NPU待验项没有被默认为通过。']

report=OUT/'原文与代码逐项一致性审计.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (OUT/'requirements-comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(records[0]))
    writer.writeheader(); writer.writerows(records)
(OUT/'requirements-comparison.json').write_text(json.dumps({'date':'2026-10-10','source_sha256':hashlib.sha256((SRC/'docs/原始实验方案.md').read_bytes()).hexdigest(),'counts':dict(counts),'requirements':records,'findings':[{'id':i,'priority':p,'title':t,'finding':b,'completion':f} for i,p,t,b,loc,f in FINDINGS],'approved_supplement_ids':['B01','B02','B03','B04','B05'],'scientific_source_modified':False,'real_model_data_npu_verified':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(report),'requirements':len(original),'inferences':len(inferred),'counts':dict(counts),'findings':len(FINDINGS),'report_bytes':report.stat().st_size},ensure_ascii=False))
