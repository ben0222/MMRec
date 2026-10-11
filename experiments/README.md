# MMRec 实验记录与可恢复搜索

`run.py` 保留原有单次实验接口，一次调用可以执行整个搜索，记录日志、Git
快照、配置、指标和 CSV。`sweep.py` 在外层展开配置，逐组启动独立的
`run.py`，每组完成后立即保存记录。模型、损失、训练、Early Stopping 和评估
数学逻辑均未修改。**恢复粒度是组合，不是 Epoch**；未完成的组合从头训练。

以下 Mac 命令在仓库根目录执行，使用已经安装项目依赖的 `.venv`。
Dry-run 只加载配置，不读取训练数据、不训练、不创建或修改实验结果。

```bash
.venv/bin/python experiments/sweep.py --model VBPR --dataset sports \
  --sweep-id vbpr-sports-baseline --dry-run
```

默认 VBPR 配置得到 seed=999 和 7 个 reg_weight 值。输出包含全部组合及状态、
完成/待运行数量、本次计划数量、配置指纹、Commit 和源码指纹。

## 配置与调度

使用 MMRec 的真实 `Config`，依次合并 overall、dataset、model、可选 mg，
最后应用 `--config` 和 `--set`（后者优先）。`hyper_parameters` 决定笛卡尔积，
搜索参数必须是列表，seed 必须包含在其中；拒绝重复参数名和重复组合。
`--config` 支持 YAML 映射和 JSON 对象，`--set` 使用 YAML 值。

```bash
.venv/bin/python experiments/sweep.py --model VBPR --dataset sports \
  --sweep-id vbpr-small --config /path/to/overrides.yaml \
  --set 'seed=[999]' --set 'reg_weight=[2.0, 1.0]' \
  --set epochs=1000 --set stopping_step=20 --max-combinations 1 --dry-run
```

子进程使用当前 `sys.executable`，参数以列表传入，不进行 Shell 拼接。
搜索参数被包装成单元素列表，`--single-combination` 在加载数据和训练前再次
拒绝多组合配置。使用 `Popen` 管理进程组，便于信号转发和回收；终端输出直接
继承，`run.py` 同时归档 `full.log`。

`--max-combinations N` 限制本次启动次数，失败尝试也计数，达到上限后状态为
`paused`。`--max-runtime-minutes N` 只在新组合启动前检查时间，超过预算则停止
启动。**无法预测单组时长，运行中的组合可能超过预算；这不是严格费用上限，
也不因时间预算杀死训练。** `--continue-on-error` 允许失败后继续，但最终仍
返回非零退出码。子进程失败码会传播，信号退出映射为 128+信号编号。
`--no-save-model` 和 `--mg` 与原运行器一致，并纳入恢复校验。

原运行器仍可在本地 CPU 使用（需本地数据；以下不是本次已执行的训练）：

```bash
.venv/bin/python experiments/run.py --model VBPR --dataset sports --tag cpu-check \
  --set use_gpu=false --set 'seed=[999]' --set 'reg_weight=[2.0]' --no-save-model
```

## 持久化与选优

```text
experiments/runs/<run_id>/
  metadata.json, resolved_config.json, results.json, full.log
  git_commit.txt, git_status.txt, code_diff.patch, untracked_source/
  combinations/<index>.json
experiments/sweeps/<sweep_id>/
  sweep_config.json                  # 固定搜索空间、配置、选项和代码身份
  manifest.json                      # 数量、状态、验证集最优组合索引
  combinations/000/
    config.json, context.json         # 单组合输入和校验上下文
    record.json                      # 状态、run_id、尝试历史、指标和结果指纹
```

关键 JSON 使用同目录临时文件、flush、fsync、原子替换；先保存组合记录，
再更新 manifest。每次尝试使用新的 run 目录，不覆盖旧日志或历史结果。
manifest 的 `best_by_validation_index` 根据 `valid_metric` 和
`valid_metric_bigger` 选取；测试指标只用于报告，绝不用于 Sweep 选优。
MMRec 原有控制台 `BEST` 仍使用历史测试集选优逻辑，分析时请使用结构化记录。

`runs.csv` 每个组合一行，`results.csv` 每个独立 run 一行；Sweep 全局最佳在
manifest 中。CSV 原子更新保留历史行，按 run_id/组合编号防止重复索引，冲突
记录报错。文件锁防止并发调度和 CSV 更新竞争，不提供跨主机分布式锁。
Git 跟踪代码/配置，忽略 runs、sweeps、两个 CSV 和 CSV 备份；历史实验数据
没有迁移或删除。

查看进度和日志：

```bash
cat experiments/sweeps/vbpr-sports-baseline/manifest.json
cat experiments/sweeps/vbpr-sports-baseline/combinations/000/record.json
# 将 RUN_ID 替换为 record.json 中的 run_id
tail -f experiments/runs/RUN_ID/full.log
```

## 安全恢复

恢复指定原 sweep_id，并重复原配置覆盖、mg、保存模型选项；预算、最大组合
数和 continue-on-error 可以改变。不会自动创建新 ID 或自动接受新的 Commit。

```bash
.venv/bin/python experiments/sweep.py --model VBPR --dataset sports \
  --sweep-id vbpr-sports-baseline --resume --max-combinations 1 --dry-run
```

只有满足以下校验的组合才能跳过：

- 固定搜索配置、选项、Commit 和实际源码内容一致，包括未提交修改。
- 组合身份和上下文一致，run 元数据标记 completed，Git 快照一致。
- 结果和配置快照可解析，模型名称、数据集、超参数及完整配置一致。
- 实际结果只有一个组合；验证 Epoch、全部配置要求的验证/测试指标有效，
  验证分数匹配选择指标，结果指纹和 record 中的指标一致。

设备字段由机器 CUDA 可用性派生，校验排除 `device`；`use_gpu` 和 `gpu_id`
仍参与校验。包版本和数据文件内容没有自动校验，迁移时须自行保持一致。
请勿将凭证放入实验配置或命令行；快照会对疑似凭证键脱敏。

completed 跳过，pending 执行，failed 重试，running 视为中断后重试。若子进程
已写完全部文件并标记 completed、父进程尚未更新 record，恢复会校验并接纳
该结果，避免重复训练。JSON 损坏、结果缺失或身份不匹配会报错，**不会静默
跳过，也不会自动覆盖损坏文件**；先备份并检查证据。manifest 是可重建汇总，
恢复以独立记录为准。若初始化时 sweep_config 写入失败，残留目录需人工检查。

同一工作副本只允许一个 Sweep 调度器。锁由子进程继承，父进程被强制杀死而
训练仍在运行时，另一调度器无法重复启动。SIGINT/SIGTERM 会停止训练进程组
并保存 interrupted；SIGKILL/断电无法清理，但已提交的结果保留。不要在搜索
期间修改代码；代码变化会使后续调度或恢复失败。先提交代码，再在 AutoDL
使用相同 Commit 和配置建立正式 Sweep。

## AutoDL：独立于 SSH 的自动关机启动

正式 GPU 任务统一通过 `run_autodl.sh` 启动。要求显式 `--autodl`、Linux root、
`/root/autodl-tmp`、`/usr/bin/shutdown` 和 setsid/nohup/flock/sync/tar。
环境检查用于防误用，**必须在实际 AutoDL 容器中执行**。Mac 无法触发此路径。
关机命令来自 [AutoDL 官方说明](https://www.autodl.com/docs/save_money/)。

在 AutoDL 仓库根目录、激活项目 Python 环境后执行：

```bash
AUTODL_PYTHON="$(command -v python)" \
  bash experiments/run_autodl.sh --autodl \
  --model VBPR --dataset sports --sweep-id vbpr-sports-baseline \
  --max-combinations 1 --max-runtime-minutes 120
```

恢复正式任务（相同配置；这里的原任务没有额外覆盖）：

```bash
AUTODL_PYTHON="$(command -v python)" \
  bash experiments/run_autodl.sh --autodl \
  --model VBPR --dataset sports --sweep-id vbpr-sports-baseline \
  --resume --max-combinations 1 --max-runtime-minutes 120
```

启动器通过 nohup+setsid 后台执行、关闭标准输入，输出启动 PID 和独立日志
路径，SSH 断开通常不影响任务。worker 的 EXIT trap 在正常结束、失败及可处理
终止信号后保存训练退出码、完成可选备份与 sync，然后关机。第二个活跃启动器
被拒绝，不会关机干扰第一个。查看输出路径的 launcher.log 和 exit_status.txt；
父启动命令成功只表示已提交后台任务，训练退出码以文件为准。

可选备份（目标须为已挂载、可写的持久存储，并位于 experiments 之外）：

```bash
AUTODL_PYTHON="$(command -v python)" AUTODL_BACKUP_DIR=/root/autodl-fs/mmrec-backups \
  bash experiments/run_autodl.sh --autodl \
  --model VBPR --dataset sports --sweep-id vbpr-sports-baseline --resume --max-combinations 1
```

备份包含 runs、sweeps 和已有 CSV，使用唯一 tar.gz 文件；不自动 Push 或上传。
备份失败会记日志，仍尝试关机；默认只 sync 本地文件。无远程持久存储时，应
在后续无卡维护时段下载备份。不要把唯一备份留在可能释放的实例本地磁盘。

关机失败记录命令退出码，需及时从 AutoDL 控制台关机。trap 无法保证处理
SIGKILL、余额不足被平台强制停止、系统崩溃、掉电或平台清理进程。下次启动
后用 resume 校验，完成组跳过、未完成组重新训练；Epoch 50 中断不意味着能
从 Epoch 50 继续。nohup/setsid 无法对抗平台强制终止。时间预算不是备用强制
关机计时器。本次未在实际 AutoDL 环境验证关机。

Mac 上仅预览启动脚本（不训练、不关机）：

```bash
bash experiments/run_autodl.sh --check-only \
  --model VBPR --dataset sports --sweep-id vbpr-sports-baseline --max-combinations 1
```

## 历史中断结果：仅来源校验与预览

重要历史 run 为 `20261011_110206_VBPR_sports_baseline-full_47f69f81`。
AutoDL 的 `/root/autodl-tmp/experiment_backups/interrupted_vbpr/` 不是 Mac 本地
路径，需先自行复制 recovered JSON、原配置和日志等证据。

```bash
.venv/bin/python experiments/sweep.py --model VBPR --dataset sports \
  --sweep-id vbpr-sports-baseline \
  --import-results /path/to/recovered_vbpr_results.json
```

此选项为**只读预览**，不写入 Sweep、不跳过训练。接受对象或组合列表，输出
`verified: false` 和缺失/不匹配的来源字段。对象可附 git_commit、model、dataset、
resolved_config、seed、search_space（完整组合列表）、valid_metric（小写）及
source（日志路径或来源标识），与当前计划比较；即使字段齐全也不自动认定
日志恢复结果可信。本阶段没有自动导入能力，未读取或修改 AutoDL 历史数据。

## 备份、迁移与验证

一起保存 runs、sweeps、CSV、所用代码/配置和数据。run_id 引用仓库内相对布局，
不依赖旧主机绝对路径。在无训练运行时可以创建独立备份：

```bash
tar -czf /path/to/new-mmrec-backup.tar.gz \
  experiments/runs experiments/sweeps experiments/runs.csv experiments/results.csv
```

目标使用新名称避免覆盖；不存在的 CSV 从命令中移除。在目标工作副本恢复
原目录结构，保持 Commit/源码/配置一致，再执行 resume dry-run。不要边训练
边手工修改、合并 CSV 或搬动组合目录。

测试使用临时目录和 Mock 训练/子进程；关机生命周期测试使用临时脚本及伪关机
命令，生产关机路径不执行：

```bash
.venv/bin/python -m pytest tests/test_experiment_records.py -v
.venv/bin/python -m pytest tests/test_sweep.py -v
.venv/bin/python -m pytest tests/ -v
git diff --check
```
