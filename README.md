# LabSentinel：实验室安全视觉分析原型

LabSentinel 使用 YOLO11s 检测实验室场景中的进食、饮水和防护用品穿戴状态。后端按检测类别生成可复核的风险事件；网页支持图片、视频抽帧和摄像头分析，以及 DeepSeek 辅助报告、问答和历史记录。

## 运行

1. 使用 Python 3.9 创建 `.venv`，安装 `app/backend/requirements.txt`。
2. Windows 双击 `start.bat`，打开 `http://127.0.0.1:8001/`。
3. 如需智能报告，参考 `.env.example` 在本地 `.env` 设置 `DEEPSEEK_API_KEY`。不要将密钥提交到代码仓库。未配置或接口不可用时，问答会返回简化的本地回答；智能报告需要 DeepSeek 接口。

后端入口为 `app/backend/main.py`，模型推理在 `app/backend/inference.py`，规则在 `app/backend/risk_engine.py`，前端在 `app/frontend/`。

## 数据与实验

原始数据集位于 `datasets/laboratory_safety`，包含 7,217 张图像和 12 类检测目标。原始划分为训练 5,096、验证 1,414、测试 707 张。数据标签由分割多边形转换成检测边界框；原始数据及训练权重保留不动。

审计发现，按文件名规则归为同组的 31 个前缀组同时出现在原始训练、验证和测试集。数据集实际由图片组成；部分图片名称包含 `_MP4-编号`，这只是命名线索，尚未核实它们是否来自同一视频或采集场景，不能仅凭前缀重合认定数据泄漏。原有 `asset/metrics.json` 只能作为旧实验记录：当前部署权重的 SHA-256 与 `runs/laboratory_safety2/weights/best.pt` 相同，而旧指标文件生成时间早于该部署权重的替换时间，旧指标与当前权重的对应关系尚未核实。

新划分按图片文件名分组：对 `_MP4-编号` 命名的图片使用编号前的前缀，对其他图片使用去除 `.rf.` 后缀及图片扩展名标记后的名称。这是一种减少潜在同源样本跨集合分布的启发式方法，需要结合图片内容或采集记录核实。原图和标签保持原位置。命令：

```powershell
.\.venv\Scripts\python.exe -B scripts/dataset_audit.py --create-grouped-split
```

输出在 `datasets/laboratory_safety_grouped/`：`train.txt`、`val.txt`、`test.txt`、`data.yaml` 和 `audit.json`。三个新划分的文件名分组交集均为零；图像数量分别为 4,998、1,484、735。这不等于已经确认真实采集来源相互独立；仍需结合图片内容和采集记录核实组内及组间关系。

按新划分训练。训练过程用验证集选最佳权重；完成模型选择后，只对最终选定的运行做一次测试集评估：

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
.\.venv\Scripts\python.exe -B scripts/train.py --epochs 100 --patience 20 --device 0
```

单独评估某个权重：

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
.\.venv\Scripts\python.exe -B scripts/evaluate_provenance.py --model runs/<运行目录>/weights/best.pt --data datasets/laboratory_safety_grouped/data.yaml --split test --device 0
```

评估结果写入 `asset/evaluations/`，包含模型 SHA-256、数据划分、置信度阈值、逐类和整体指标。使用 `--conf 0.001` 计算完整的精确率召回率曲线；界面和后端推理默认阈值为 0.15（15%）。只有在确认新评估报告后才运行 `scripts/deploy_model.py <metrics_with_provenance.json>`。部署脚本会校验分组测试集和权重哈希，备份旧权重，再更新 `models/laboratory_safety/weights/best.pt`。部署后重启服务以加载新权重。

曾启动一次分组训练进行流程检查，并在完成 5 个 epoch 后按要求停止；`runs/laboratory_safety_grouped2/` 中的权重仅为中途检查点，**未部署、未作为正式实验结果**。后续完整训练由项目所有者自行执行。新运行目录会自动增加编号，以上评估命令中的 `<运行目录>` 需替换为训练输出路径。

## 风险输出边界

- 进食或饮水不会自动生成“未戴口罩”检测；只有模型检测到对应 `No` 类，才产生该 PPE 事件。
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
```

检测 mAP 不能直接代表安全事件告警质量。下一轮应采集未参与训练的现场视频，人工标注事件起止时间、误报与漏报，并报告事件级精确率、召回率、每小时误报数、告警延迟及运行帧率。可比较“单帧检测”“单帧加规则”“多帧告警”三个版本，保持相同测试视频与阈值。真实视频的人工事件标签目前不在项目中，因此这些指标尚不能给出。

项目已从新测试集抽取 120 张图像，生成 `asset/event_review.csv`。请人工填写 `true_events`：用 `|` 分隔 `drinking`、`eating`、`no_gloves`、`no_head_mask`、`no_lab_coat`、`no_mask`、`no_goggles`，没有事件填 `none`，尚未复核保持空白。训练完成后可运行：

```powershell
.\.venv\Scripts\python.exe -B scripts/event_review.py score --review asset/event_review.csv --model runs/<运行目录>/weights/best.pt --device 0
```

该脚本输出的是**人工复核图片的帧级事件指标**；它不能代替真实视频的告警延迟和每小时误报评估。
