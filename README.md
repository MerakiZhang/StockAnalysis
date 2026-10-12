# StockAnalysis

A 股多因子选股程序：维护两个股票池，均经 qlib 回测验证后投入使用。

- **稳健池（stable）**：基本面 + 技术面多因子综合打分（股息率/低波动/股息连续性/盈利质量/估值/反转），30 只，**月度调仓**，仅主板，对标沪深300。回测 2018~2026：超额年化 +8.9%，样本外（2023+）超额 +14.5%/IR 1.10
- **短线池（short）**：超跌反转 + 缩量低吸（IC 实证方向），8 只，**周度调仓**，全市场全板块（纸上跟踪），带 -5% 止损/10 日强制轮换，对标中证1000。⚠️ 全区间超额 -2.5%（成本侵蚀），样本外 +19.2%/IR 0.93 —— **仅供观察跟踪，不建议实盘重仓**

选股规则的完整说明见 `RULES.md`，代码架构与扩展指南见 `ARCHITECTURE.md`。

## 环境

```bash
conda activate qlib   # conda 环境名 qlib（含 akshare/qlib/baostock）
# qlib 数据: ~/.qlib/qlib_data/cn_data（默认路径，可用 QLIB_DATA 环境变量覆盖）
# 本机 32G 内存默认 QLIB_KERNELS=2；内存紧张时 set QLIB_KERNELS=1
```

## 日常使用

```bash
# 1. 更新数据（稳健池每月、短线池每周运行前先跑）
python scripts/update_data.py                    # 全量（含估值，较慢）
python scripts/update_data.py --skip-valuation   # 快速（短线池够用）

# 2. 生成当期股池 + 调仓简报
python scripts/run_selection.py --pool stable    # 每月一次
python scripts/run_selection.py --pool short     # 每周五收盘后

# 3. 回测验证（改规则后必须重跑，数字须对齐 RULES.md 基线）
python scripts/run_backtest.py --score real                  # 双池全量
python scripts/run_backtest.py --pools short                 # 只跑短线池
# A/B 实验（不覆盖基线报告）：
#   --weights '{"neg_max_ret_20":1.0}' --tag xxx   覆盖因子权重
#   --boards main,cyb                              板块覆盖
# 参数敏感性网格：python scripts/run_sensitivity.py --pools stable

# 4. 单因子有效性检验（IC/分层/相关性，含滚动 IC 衰减监控）
python scripts/run_factor_check.py --pool stable
python scripts/run_factor_check.py --pool short
```

## 目录

```
config/selector.yaml   # 池子参数、因子权重、成本、过滤规则（唯一规则来源）
src/config.py          # 配置加载与校验
src/data/              # market=qlib 行情 / fundamental=akshare 基本面缓存
src/factors/           # registry=因子注册表 / technical / fundamental / base
src/selection/         # universe=过滤 / scoring=打分合成 / pool=股池生成
src/backtest/          # engine=回测引擎 / strategies=短线止损策略
src/eval.py            # 因子 IC 检验 + 内存监控
scripts/               # update_data / run_selection / run_backtest / run_factor_check / run_sensitivity
data/fundamental/      # akshare 基本面缓存（yjbb/fhps/xjll/估值，parquet）
data/pool/             # 股池输出 stable_YYYY-MM.csv / short_YYYY-MM-DD.csv + 简报
data/reports/          # 脚本生成的报告（可再生产物，不入库）
```
