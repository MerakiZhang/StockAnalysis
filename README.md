# StockAnalysis

A 股全市场规则选股程序。流程：全市场 → 剔除烂股 → 按账户板块权限过滤 → 各池按买点入池、卖点出池，每池最多 10 只。
每只股票都有明确的买入原因和当前卖出线（止损价、均线价、剩余持有天数等），每个交易日收盘后更新。

| 股池 | 状态 | 思路 | 基准 | 研发区 2018~2022（年化 / 超额） | 保留集 2023+（年化 / 超额） |
|------|------|------|------|------------------------------|---------------------------|
| long | production | 质量价值 + 分红 + 预期，持有数月到数年 | 中证800 | +15.7% / +15.5% | +12.2% / +7.6% |
| short_reversal | production（偏弱） | 急跌 / RSI 低位反转，持有 5 天 | 中证1000 | +5.1% / +4.0% | +8.2% / +2.0% |
| short_breakout | observe | 放量突破（对照池） | 中证1000 | -3.6% / -4.8% | -6.5% / -11.7% |

均为扣除成本（佣金 + 印花税 + 滑点）后的口径，单一起点回测；6 个错开建仓日合成的保留集结果
（long +13.0% / +8.8%，short_reversal +7.4% / +1.8%，short_breakout -6.7% / -11.4%）结论一致。

**注意**：
- 状态按事先定好的规则判定：研发区和保留集都满足"扣成本后绝对年化 > 0 且超额 > 0"才标 production。
- short_reversal 虽然达标，但研发区参数邻域不稳（持有天数从 5 改为 20 即转负）、单一起点最大回撤约 -48%、
  年化成本约 4.8%，仓位宜轻，并以 2026-10-15 起的实盘跟踪为准。
- long 的收益集中在少数股票（如 2020~2022 年的煤炭股），2018 年亏约 22%。
- 规则、依据与评估记录见 `RULES.md`；代码架构、修改方法与防未来约定见 `ARCHITECTURE.md`；重构过程见 `REFACTOR_PLAN.md`。

## 环境

```bash
conda activate qlib   # 含 qlib 0.9.7 / akshare / baostock / akquant / pytest；在项目根目录运行
# qlib 数据: ~/.qlib/qlib_data/cn_data（来源 GitHub chenditc/investment_data，可用 QLIB_DATA 覆盖）
# 长桥 CLI（可选，选股简报的"体检"栏）：~/AppData/Local/Programs/longbridge/longbridge.exe，需已登录
```

## 日常使用

所有功能通过一个入口调用：`python -X utf8 -m stockanalysis <命令>`（`--help` 查看参数）。

```bash
# 1. 更新补充数据（断点续传；首次全量约数小时，之后每日增量）
python -X utf8 -m stockanalysis update-data                # 基本面 / 事件 / baostock / 市场温度
python -X utf8 -m stockanalysis update-data --only events  # 只更新事件（两融、龙虎榜、公告等）
python -X utf8 -m stockanalysis update-data --status       # 只看各数据源截止日（滞后的标 ⚠），不下载

# 2. 生成当期股池（启动时自动比对 GitHub 行情数据版本，落后则下载并原子替换）
python -X utf8 -m stockanalysis select --pool all          # 或 --pool long
#    输出 data/pool/{pool}/{日期}.csv + {日期}_brief.md：
#    持仓（买入价、收益、卖出线）、次日计划买入 / 卖出（原因）、接近买点名单、长桥体检
#    --no-health 跳过长桥体检

# 3. 只比对行情数据版本
python -X utf8 -m stockanalysis sync --check
```

成交口径：T 日收盘出信号，T+1 收盘成交；止损 / 止盈按收盘价判断，次日执行，不是盘中触价。
价格一律显示不复权价，与行情软件一致。

## 研究与验证（默认只看研发区 ≤ 2022-12-31，保留集须加 --show-holdout）

```bash
python -X utf8 -m stockanalysis quality --pool long                    # 烂股剔除效果体检
python -X utf8 -m stockanalysis event-study --pool long                # 条件事件研究（5 / 20 / 60 日超额、胜率、逐年）
python -X utf8 -m stockanalysis points-grid --pool long --starts 6 \
    --vary "points.stop_loss=0.05,0.08,null" --vary "points.max_hold=5,10"   # 买卖点参数网格 + 滚动验证
python -X utf8 -m stockanalysis backtest --pool all                    # 组合回测
python -X utf8 -m stockanalysis check --pool all                       # 防未来检查（数据截止到 T 与延续之后结果须一致）
python -X utf8 -m pytest tests -q                                      # 单元测试
# 多数命令可用 --set 点路径覆盖配置（如 --set points.stop_loss=0.05），--tag 给报告加后缀避免覆盖
```

## 配置与目录

```
config/global.yaml          系统级：资金、账户板块权限、成本、撮合、回测窗口、数据同步
config/pool_defaults.yaml   所有股池共用的规则（烂股剔除、准入）——改一处，全部池生效
config/pools/<name>.yaml    单个股池：基准、优先分、买卖点（points 段）；加池时复制一个现有池改名
stockanalysis/              代码包：config / data / features / models / selection / backtest / research / cli
tests/                      单元测试
data/pool/<name>/           股池输出（入库，实盘跟踪记录）
data/reports/               研究报告（不入库）
data/raw/{akshare,baostock,longbridge}/   原始数据缓存（不入库，可重新下载）
data/panel/                 由 raw 生成的紧凑表（不入库，可随时重建）
```

加股池、改买卖点、加选股条件等常见修改的具体做法见 `ARCHITECTURE.md` 开头的"怎么改"表。

旧版三个指数池（沪深300 / 中证500 / 中证1000，LightGBM 打分）2023 年以来扣成本后没有稳定超额，
已于 2026-10-17 删除，需要时见 git 历史（`main` 分支 2026-10-16 之前的提交）。
