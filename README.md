# LabSentinel：实验室安全视觉分析原型

LabSentinel 使用 YOLO11s 检测实验室场景中的进食、饮水和防护用品穿戴状态。后端按检测类别生成可复核的风险事件；网页支持图片、视频抽帧和摄像头分析，以及 DeepSeek 辅助报告、问答和历史记录。

## 运行

GitHub 保存源码、配置及评估报告，模型权重、原始图片／标签和完整数据清单需另行准备。克隆后的模型放置、数据路径调整与本次更新范围见 [GitHub 仓库说明](REPOSITORY_NOTES.md)。

1. 使用 Python 3.9 创建 `.venv`，安装 `app/backend/requirements.txt`。
2. Windows 双击 `start.bat`，打开 `http://127.0.0.1:8001/`。
3. 如需智能报告，参考 `.env.example` 在本地 `.env` 设置 `DEEPSEEK_API_KEY`。不要将密钥提交到代码仓库。未配置或接口不可用时，问答会返回简化的本地回答；智能报告需要 DeepSeek 接口。

后端入口为 `app/backend/main.py`，模型推理在 `app/backend/inference.py`，规则在 `app/backend/risk_engine.py`，前端在 `app/frontend/`。

## 数据与实验

统一数据集位于 `datasets/laboratory_safety`，包含 7,217 张图像和 12 类检测目标。原始文件夹中的图片位置保持不变，训练实际使用根目录的 `train.txt`、`val.txt`、`test.txt`，分别为 4,998、1,484、735 张；不要直接用 `images/train` 等子目录代替这些清单。数据标签由分割多边形转换成检测边界框。

最新标注版本为 `coat_audit_20261027_v1`：对 201 张实际训练图片完成衣服专项复核，修正 88 份标签，保留 2 处严重遮挡的候选疑点，现共 54,316 个框。验证集、测试集标签和三份划分清单均未变；全量 7,217 份标签的结构检查通过。备份、逐图记录及后续数据补充方案见 [实验服专项复核报告](asset/coat_audit_20261027/实验服专项复核报告.md)。此次标签修正已用于 `runs/laboratory_safety_corrected/` 训练，2026-10-28 完成测试评估并部署。

本次衣服专项复核之前，2026-10-27 曾完成两轮共 202 张图片的人工复核修正（含 13 个原空标签），并对全部图片及标签执行读取、格式、边界和重复检查，当时共 54,314 个框。R241 保留饮水，R268 保留进食。早期辅助报告与预览后来已清理；本次 `asset/coat_audit_20261027/` 中的完整标签备份保留了此前部署模型对应的标注版本。验证集和测试集曾在此前训练开始前补齐 13 个空标签，更早的旧指标不能直接当作同一标注版本的对比。本次衣服专项复核没有再改动验证或测试标签，且与此前 202 张复核范围可能重叠，不累加为全部不同图片的复核数量。

护目镜类别统一为 `Goggles`（ID 3）和 `No goggles`（ID 11），接口字段分别为 `goggles` 和 `no_goggles`。历史类别命名修正记录在 `asset/class_name_correction.json`，其中的旧实验文件路径只用于追溯，部分文件已清理。当前部署依据为 `models/laboratory_safety/weights/deployment.json`。服务运行期间修改权重后，需要重启后端。

审计发现，按文件名规则归为同组的 31 个前缀组同时出现在原始训练、验证和测试集。数据集实际由图片组成；部分图片名称包含 `_MP4-编号`，这只是命名线索，尚未核实它们是否来自同一视频或采集场景，不能仅凭前缀重合认定数据泄漏。旧指标与旧实验产物已清理，当前模型结果以本轮独立测试集报告为准。

当前划分按图片文件名分组：对 `_MP4-编号` 命名的图片使用编号前的前缀，对其他图片使用去除 `.rf.` 后缀及图片扩展名标记后的名称。这是一种减少潜在同源样本跨集合分布的启发式方法，需要结合图片内容或采集记录核实。原图和标签保持原位置，划分清单已冻结。只读审计命令：

```powershell
.\.venv\Scripts\python.exe -B scripts/dataset_audit.py
```

`train.txt`、`val.txt`、`test.txt`、`data.yaml`、`audit.json` 和标注哈希清单现已合并到 `datasets/laboratory_safety/`。旧的 `laboratory_safety_grouped` 目录归档在 `asset/label_repair_remaining_20261026/archived_grouped_metadata/`；合并前后划分清单哈希一致。三个划分的文件名分组交集均为零，这不等于已经确认真实采集来源相互独立。`--create-grouped-split` 会拒绝覆盖已冻结的清单。

按新划分训练。训练过程用验证集选最佳权重；完成模型选择后，只对最终选定的运行做一次测试集评估：

2026-10-28 已生成单独的上半身裁剪增强版本：`datasets/laboratory_safety_upperbody_20261028/`。新增 3,996 张裁剪图，实验服和普通衣服主要目标各 1,998 张；训练清单共 8,994 张，验证／测试仍为原来的 1,484／735 张。原图、标签与冻结清单未变。使用该版本时，需给下面的训练命令额外指定 `--data datasets/laboratory_safety_upperbody_20261028/data.yaml`；默认不带 `--data` 仍训练原数据集。该数据版本现已完成训练及部署，详细核验与命令见 [裁剪增强数据说明](datasets/laboratory_safety_upperbody_20261028/README.md)。

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
.\.venv\Scripts\python.exe -B scripts/train.py --epochs 100 --patience 20 --device 0
```

训练中断且 `last.pt` 仍含优化器状态时，可用同一入口继续原运行：

```powershell
.\.venv\Scripts\python.exe -B -X faulthandler scripts/train.py --resume runs/laboratory_safety_corrected/weights/last.pt
```

`--resume` 沿用检查点中的训练参数、轮数和目录，恢复前会核对原运行的数据清单及全部标签哈希，另存恢复事件并持续更新 `status.json`。完成训练时已去除优化器的权重不能断点恢复。当前 OpenCV 固定为 `opencv-python==4.11.0.86`，用于处理本机 OpenCV 5.0 在训练读图解码时出现的间歇性原生访问冲突；环境变更及恢复证据见运行目录中的 `recovery_20261027/`。

单独评估某个权重：

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
.\.venv\Scripts\python.exe -B scripts/evaluate_provenance.py --model runs/<运行目录>/weights/best.pt --data datasets/laboratory_safety/data.yaml --split test --device 0
```

评估结果写入 `asset/evaluations/`，包含模型 SHA-256、数据划分、置信度阈值、逐类和整体指标。使用 `--conf 0.001` 计算完整的精确率召回率曲线；界面和后端推理默认阈值为 0.15（15%）。只有在确认新评估报告后才运行 `scripts/deploy_model.py <metrics_with_provenance.json>`。部署脚本会校验分组测试集和权重哈希，备份旧权重，再更新 `models/laboratory_safety/weights/best.pt`。部署后重启服务以加载新权重。

2026-10-27 已完成修正数据训练，在第 70 轮触发提前停止，最佳模型来自第 50 轮。运行目录为 `runs/laboratory_safety_corrected_20261027/`，由 COCO 预训练 `models/yolo11s.pt` 开始，参数为最多 100 epochs、patience 20、batch 8、imgsz 640、seed 42、workers 0、GPU 0。日志在 `asset/training_corrected_20261027/`；运行目录的 `status.json` 记录状态，`results.csv` 保存每轮训练/验证指标，`weights/best.pt` 和 `last.pt` 已保留。中间 epoch 检查点及中断前重复权重已清理；最终权重已移除优化器状态，不适合作为原训练状态的断点恢复。

上述旧模型测试集指标为 mAP50=0.7465、mAP50-95=0.5243、Precision=0.6676、Recall=0.7953，报告见 [旧模型测试评估](asset/evaluations/laboratory_safety_test_ca28182b38d5_20261027T091446Z/测试集评估报告.md)。

2026-10-28，使用实验服专项修正标注的运行 `runs/laboratory_safety_corrected/` 在第 73 轮提前停止。按验证集选出的最佳权重完成同一测试集评估：735 张图片、5,394 个目标，mAP50=0.7222、mAP50-95=0.5023、Precision=0.6491、Recall=0.7955。两个 mAP 分别比旧模型低 2.43 和 2.21 个百分点；该单次结果不支持性能提升结论。该模型曾部署到应用，SHA-256 前缀为 `64ce2033b6ba`。完整结果见 [上一版模型评估与部署记录](asset/evaluations/laboratory_safety_test_64ce2033b6ba_20261028T035521Z/测试集评估报告.md)。

当前部署的是上半身裁剪增强运行 `runs/laboratory_safety_upperbody_20261028/` 的最佳权重，训练于第 59 轮正常结束。2026-10-28 在同一冻结测试集评估得到 mAP50=0.7582、mAP50-95=0.5264、Precision=0.6772、Recall=0.7939；相比替换前的 `64ce2033b6ba` 模型，两个 mAP 分别增加约 3.60、2.42 个百分点。当前权重 SHA-256 前缀为 `0f0ad72e2954`，部署记录仍为 `models/laboratory_safety/weights/deployment.json`，本次替换前备份为 `best_backup_before_grouped_20261028_234552.pt`。已验证项目实际加载并推理成功；此前半身诊断图仍有衣服类别冲突，不能声称误判完全解决。详见 [当前模型评估与部署记录](asset/evaluations/laboratory_safety_upperbody_20261028_test_0f0ad72e2954_20261028T154440Z/测试集评估报告.md)。

清理遵循“保留完整毕设资料”的范围：运行代码、环境、配置、数据集、标签修正证据、最新训练与测试结果保留；旧实验目录、过期指标、重复权重、缓存和临时训练查看页已移除。逐项清单与校验结果见 `asset/cleanup_20261027/`。

## 风险输出边界

- 对同一帧中高度重叠的“实验服”和“未穿实验服”框，采用实验服优先规则：IoU 达到 0.70 时保留实验服并抑制冲突的未穿框，即使未穿框的置信度更高。标注图、检测列表和风险使用同一组处理后结果；接口及新历史记录的 `postprocessing` 保存被抑制的原始框和重叠度。不同位置、不同帧的结果分别处理，不会将整段视频锁定为已穿。此规则不等于逐人跟踪，拥挤遮挡时仍需复核。上述测试集 mAP 是原始模型评估值，未计入这项应用后处理。
- 上传视频时，“未穿实验服”须在连续两个采样关键帧中检出，才纳入视频风险提示。同一帧的多个框不算多帧，隔帧出现也不满足连续条件。孤立帧及零散帧保留为 `pending_events` 待复核；原始检测框、置信度和检测数量保留，页面检测明细注明待复核。页面汇总、Agent 问答和报告使用同一条件，新历史记录保存判定状态及同段视频的衣服邻帧证据。旧记录不回写，更新后应重新分析视频。该规则不提供人员身份跟踪，也不能将采样帧号解释为秒数。
- 检测到进食或饮水时，按业务规则额外判定“未戴口罩（行为推断）”；即使画面存在口罩目标也保留该提示。该风险标记为 `behavior_inference`，不会增加口罩检测框、检测置信度或检测数量。若已有直接 `No Mask` 检测，则沿用直接检测事件，避免重复提示。视频汇总会保留直接检测与行为推断两种来源。
- 当前模型没有逐人关联能力。画面里同时出现五种 PPE，不代表每个人都穿戴齐全；检测到 `No` 类时也需现场确认具体人员。
- 视频按时间均匀抽取最多 40 帧，短暂事件仍可能漏检。摄像头约每秒分析一帧，同一风险在最近三帧中至少出现两次才显示页面告警。历史记录保存带框图像。
- DeepSeek 报告和 Agent 仅解释检测与规则数据。Agent 使用关键词选择读取当前分析、解释风险、查询历史或给出整改建议；它不是独立的视觉判定器。

## Agent 连续对话

同一次会话会保存当前分析的检测快照，以及用户指定的关键帧和风险类别。例如，先问“第一帧护目镜有什么问题”，再问“这个判断靠谱吗”“那怎么处理”，后两轮会继续读取第一帧的护目镜证据。问“那第三帧呢”会切换到第三帧并保留护目镜这一讨论类别；问“整个视频有哪些风险”则恢复整体分析。

针对帧和类别的问答会先筛选检测结果，再计算对应风险与检测框统计。它不会借用其他帧的检出来支持当前帧的判断。不存在的帧、缺少逐帧证据或未关联分析结果时，会直接说明需要补充的信息。

每次上传新文件、重新分析或切换输入模式都会更换分析标识，下一轮聊天清除旧分析的焦点和后台对话记忆；页面已经显示的消息保留原样。摄像头同一次启动使用稳定标识，聊天时读取最新一帧结果。清空会话会同时清除聊天记录、检测快照和讨论焦点。上下文保存在后端进程内存中，重启服务或刷新页面开启新会话后需重新关联分析结果。

当前工具路由仍由规则和会话焦点决定，尚未实现大模型自主工具调用循环。自然语言指代支持常见的连续追问；跨多条历史记录无法确定所指对象时，会要求指定记录。

## 验证与后续数据收集

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
node --check app/frontend/app.js
node tests/test_video_risk.js
```

检测 mAP 不能直接代表安全事件告警质量。下一轮应采集未参与训练的现场视频，人工标注事件起止时间、误报与漏报，并报告事件级精确率、召回率、每小时误报数、告警延迟及运行帧率。可比较“单帧检测”“单帧加规则”“多帧告警”三个版本，保持相同测试视频与阈值。真实视频的人工事件标签目前不在项目中，因此这些指标尚不能给出。

项目已从新测试集抽取 120 张图像，生成 `asset/event_review.csv`。请人工填写 `true_events`：用 `|` 分隔 `drinking`、`eating`、`no_gloves`、`no_head_mask`、`no_lab_coat`、`no_mask`、`no_goggles`，没有事件填 `none`，尚未复核保持空白。训练完成后可运行：

```powershell
.\.venv\Scripts\python.exe -B scripts/event_review.py score --review asset/event_review.csv --model runs/<运行目录>/weights/best.pt --device 0
```

该脚本输出的是**人工复核图片的帧级事件指标**；它不能代替真实视频的告警延迟和每小时误报评估。
