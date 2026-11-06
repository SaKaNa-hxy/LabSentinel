# GitHub 仓库说明

本仓库保存 LabSentinel 毕业设计的源码、配置模板、测试、数据版本摘要和测试集评估报告。2026-11-06 同步本地迭代，保留此前 GitHub 提交历史。

## 本次迭代

- 同一帧的实验服与未穿实验服框在 IoU ≥ 0.70 时优先保留实验服，并保存冲突框处理证据。
- 上传视频中的未穿实验服风险需要连续两个采样帧确认；孤立检出保留为待复核，页面、Agent、报告和新历史记录使用一致证据。
- 进食／饮水可按业务规则产生“未戴口罩（行为推断）”，与直接检测区分，且不增加检测框数量。
- 同步护目镜类别名称校正、Agent 连续对话证据处理、上半身裁剪增强、数据核验和训练恢复脚本，以及相关回归测试。
- 更新冻结分组划分和当前部署模型的说明，移除旧版未核实指标及已合并的 `laboratory_safety_grouped` 配置。

## 模型与指标

当前本地部署权重来自 `runs/laboratory_safety_upperbody_20261028/weights/best.pt`，SHA-256 为 `0f0ad72e29548695b6ad8dea6d162b2a36fd8361f85e13acc264722312a6b69a`。部署记录见 `models/laboratory_safety/weights/deployment.json`。

2026-10-28 的冻结测试集为 735 张图片、5,394 个目标：Precision 0.6772、Recall 0.7939、mAP50 0.7582、mAP50–95 0.5264。报告、逐类表及溯源记录位于 `asset/evaluations/laboratory_safety_upperbody_20261028_test_0f0ad72e2954_20261028T154440Z/`；此前两版模型报告保留用于追溯。

这些指标来自单次模型检测评估，未计入应用层冲突框抑制和连续帧规则，不能替代真实现场视频的事件级告警评估。文件名分组无交集不等于已核实真实采集来源独立。

## 克隆后准备

1. 使用 Python 3.9 创建 `.venv`，安装 `app/backend/requirements.txt`。
2. 将与当前 12 类名称及顺序一致的专用检测权重放到 `models/laboratory_safety/weights/best.pt`。仓库只含部署元数据，不包含权重；如使用其他权重，需重新评估并更新部署记录。
3. 按 `.env.example` 创建本地 `.env`。使用 DeepSeek 时自行填写密钥。
4. Windows 双击 `start.bat` 启动应用。

训练与评估另需完整本地数据包：原图、标签、冻结的 `train.txt`／`val.txt`／`test.txt`、哈希清单、裁剪清单及相关核验证据。当前有效原数据划分为 4,998／1,484／735 张；裁剪增强训练集为 8,994 张，验证／测试不变。数据 YAML、图片清单及部分溯源 JSON 使用原机器路径，迁移时需调整并重新核验；只克隆源码不能直接复现实验。

## 未上传的本地资料

`.env`、虚拟环境、模型 `*.pt`、原始／裁剪图片和标签、完整划分与哈希清单、标注修改前备份及逐图记录、训练运行目录、用户历史记录和截图、日志及缓存均保留在本地。评估目录只同步文字报告、JSON 和 CSV；图表和样例图片留在本地。README 中指向运行目录、完整清单或早期归档的路径用于本地实验追溯，不保证全部存在于源码仓库。

## 系统范围与检查

YOLO 提供视觉检测，Agent 根据规则和会话上下文筛选证据，调用 DeepSeek 解释风险和整改建议；当前未实现人员身份关联或大模型自主多步工具调用循环。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
node --check app/frontend/app.js
node tests/test_video_risk.js
```
