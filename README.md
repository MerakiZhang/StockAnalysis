# StockAnalysis

A 股指数增强选股程序：对沪深300 / 中证500 / 中证1000 的当日成分股，用 qlib LightGBM（Alpha158 特征、
滚动训练）打分，按账户板块权限过滤后取前 1%（下限 5 只）组成股池，日频按换仓规则更新。

| 股池 | 状态 | 基准 | 研发区 2018~2022（超额年化 / IR） | 保留集 2023+（已看两次，非干净） |
|------|------|------|------------------------------|-------------------------------|
| hs300 | observe | 沪深300 | +16.9% / 0.89 | -1.2% / -0.08 |
| zz500 | observe | 中证500 | +11.1% / 0.61 | -16.4% / -1.05 |
| zz1000 | observe | 中证1000 | +12.7% / 0.69 | +0.5% / 0.03 |

**当前三池均为观察跟踪（observe），不建议据此交易**：2023 年以来扣除成本后没有稳定超额（见 `RULES.md` 第 4 节）。
超额为含成本（佣金 + 印花税 + 滑点）口径，日均换手约 13%，年化成本约 4.8%。
规则与依据见 `RULES.md`，代码架构与扩展方法见 `ARCHITECTURE.md`，重构过程与决策记录见 `REFACTOR_PLAN.md`。

## 环境

```bash
conda activate qlib   # 含 qlib 0.9.7 / lightgbm / akshare / pytest
# qlib 数据: ~/.qlib/qlib_data/cn_data（来源 GitHub chenditc/investment_data，可用 QLIB_DATA 覆盖）
# 内存：默认 QLIB_KERNELS=2；训练中证1000 时建议单独进程运行
```

## 日常使用

```bash
# 1. 生成当期股池（启动时自动比对 GitHub 数据版本，落后则下载并原子替换）
python -X utf8 scripts/run_selection.py --pool hs300        # 或 --pool all
#    输出 data/pool/{pool}/{日期}.csv + {日期}_brief.md（新进 / 调出原因 / 留存 / 排名）

# 2. 每年初（或外推超过一年时）重新滚动训练模型（固定 200 棵树）
python -X utf8 scripts/run_train.py --pool hs300            # 各池单独运行

# 3. 基本面缓存（仅研究补充特征时需要，日常选股不依赖）
python -X utf8 scripts/update_data.py
```

## 研究与验证（默认只看研发区，保留集须加 --show-holdout）

```bash
python -X utf8 scripts/run_backtest.py --pool all                     # 组合回测
python -X utf8 scripts/run_model_compare.py --pool all                # 模型 A/B：RankIC + TopK 未来超额 + 回测
python -X utf8 scripts/run_sensitivity.py --pool hs300 --hold-thresh 5,10,20
python -X utf8 scripts/run_factor_check.py --pool all                 # 补充基本面特征体检
python -X utf8 scripts/check_consistency.py --pool all                # 选股与回测逐日同口径检查
python -X utf8 -m src.data.sync --check                               # 只比对数据版本
python -X utf8 -m pytest tests -q                                     # 单元测试
# 任意脚本可用 --set 点路径覆盖配置，如 --set model.lgbm.learning_rate=0.1 --tag exp
```

## 目录

```
config/global.yaml         全局默认（资金、成本、撮合、回测窗口、模型公共参数、数据同步）
config/pools/*.yaml        每池一个文件（加池 = 加文件）
src/config.py              配置发现、深合并、注册表校验 → PoolSpec
src/registry.py            通用注册表（过滤规则 / 打分器 / 策略）
src/data/                  market（qlib）/ sync（GitHub 数据同步）/ fundamental（akshare 缓存）
src/factors/               base（行情宽表）/ alpha158（特征与处理器）/ supplementary（补充基本面特征）
src/selection/             universe / filters / signal / rebalance / pool
src/models/                scorers（打分器注册表）/ rolling（滚动训练与推理）
src/backtest/              engine（池无关回测）/ strategies（EntryAwareTopkDropout 等）
src/eval.py                IC 检验工具 + 内存监控
scripts/                   run_selection / run_train / run_backtest / run_model_compare / run_sensitivity /
                           run_factor_check / check_consistency / update_data
tests/                     pytest 单元测试
data/pool/{pool}/          股池输出（入库）；data/pool/legacy/ 为重构前的旧池
data/models/ data/mlruns/  模型产物与 qlib Recorder（不入库，可重训生成）
data/reports/              报告（不入库）
```
