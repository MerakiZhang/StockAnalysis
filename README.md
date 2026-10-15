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
规则、依据与评估记录见 `RULES.md`，代码架构、扩展方法与已修复问题见 `ARCHITECTURE.md`。

## 环境

```bash
conda activate qlib   # 含 qlib 0.9.7 / lightgbm / akshare / pytest；在项目根目录运行
# qlib 数据: ~/.qlib/qlib_data/cn_data（来源 GitHub chenditc/investment_data，可用 QLIB_DATA 覆盖）
# 内存：默认 QLIB_KERNELS=2；训练中证1000 时建议单独进程运行
```

## 日常使用

所有功能通过一个入口调用：`python -X utf8 -m stockanalysis <命令>`（`--help` 查看参数）。

```bash
# 1. 生成当期股池（启动时自动比对 GitHub 数据版本，落后则断点续传下载并原子替换）
python -X utf8 -m stockanalysis select --pool all          # 或 --pool hs300
#    输出 data/pool/{pool}/{日期}.csv + {日期}_brief.md（新进 / 调出原因 / 留存 / 排名）

# 2. 每年初（或外推超过一年时）重新滚动训练模型（中证1000 内存大，逐池单独运行）
python -X utf8 -m stockanalysis train --pool hs300

# 3. 数据
python -X utf8 -m stockanalysis sync --check              # 只比对数据版本
python -X utf8 -m stockanalysis update-data               # akshare 基本面缓存（补充特征用）
```

## 研究与验证（默认只看研发区，保留集须加 --show-holdout）

```bash
python -X utf8 -m stockanalysis backtest --pool all                   # 组合回测
python -X utf8 -m stockanalysis compare --pool all                    # 模型 A/B：RankIC + TopK 未来超额 + 回测
python -X utf8 -m stockanalysis grid --pool hs300 --hold-thresh 5,10,20
python -X utf8 -m stockanalysis rounds --pool hs300                   # LightGBM 树数选择
python -X utf8 -m stockanalysis factors --pool all                    # 补充基本面特征体检
python -X utf8 -m stockanalysis check --pool all                      # 选股与回测逐日同口径检查
python -X utf8 -m pytest tests -q                                     # 单元测试
# 任意命令可用 --set 点路径覆盖配置，如 --set rebalance.hold_thresh=5 --tag exp
```

## 配置与目录

```
config/global.yaml          系统级：资金、成本、撮合、回测窗口、模型公共参数、数据同步
config/pool_defaults.yaml   所有股池共用的选股规则（池容、换仓、打分器、过滤规则）——改一处，全部池生效
config/pools/<name>.yaml    单个股池：只写与默认不同的项（_template.yaml 为模板）
stockanalysis/              代码包：config / data / features / models / selection / backtest / research / cli
tests/                      单元测试
data/pool/<name>/           股池输出（入库，观察跟踪记录）
data/models/ data/mlruns/ data/reports/ data/fundamental/   模型、训练记录、报告、基本面缓存（不入库）
```

加股池、改池容、改选股规则、加过滤规则等常见修改的具体做法见 `ARCHITECTURE.md` 开头的"怎么改"表。
