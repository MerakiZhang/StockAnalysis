# 计划：A 股全市场 → 剔除烂股 → 板块权限 → 短线进攻池 / 中长线持有池（各 10 只，买点入池、卖点出池）

## Context

现在三个指数池用 LightGBM 只按量价排名，选出来的股票基本面差（沪深300 这期 5 只里 4 只净利润下滑），2023+ 扣成本后也没有超额。

用户的决定：
- 退役模型打分，改为规则选股。
- 范围从三个指数改为 **A 股全市场**。
- 建两个股池：**短线进攻**、**中长线持有**，各 10 只，两个都要能赚钱。
- 每只股票有明确的**买入点**（满足才入池）和**卖出点**（触发就出池）。

## 一、本机环境：每个库在项目里怎么用（均已实测）

| 工具 | 版本 | 在项目里的角色 | 怎么用 |
|---|---|---|---|
| **qlib** | 0.9.7 | 数据底座 + 计算 + 回测 | cn_data 日线（OHLCV / 成交额 / vwap / 复权因子、全市场 all 股票表、交易日历）；表达式引擎算因子（Alpha158 的 158 个 + 自定义表达式）；回测引擎（成本、涨跌停、成交量上限）沿用现有 `backtest/engine.py` |
| **baostock** | 0.9.40 | 历史"烂股"判定 + 估值 + 真实公告日 | `query_history_k_data_plus`：2006+ 逐日 isST / tradestatus / peTTM / pbMRQ / psTTM / 换手率；`query_profit_data` 等季频财报带 **pubDate**；`query_forecast_report` 业绩预告。新建 `data/baostock_data.py` 逐股下载到 parquet，断点续传 |
| **akshare** | 1.19.1 | 已有基本面缓存 + 事件数据 | 现有 `data/fundamental.py`（业绩报表 / 现金流 / 分红）；新增两融（按日）、龙虎榜、股东户数（带公告日）、限售解禁、高管增减持、回购、公告类型（排雷）、业绩快报 |
| **akquant** | 0.3.66 | 技术指标函数库 | `akquant.talib`：MACD / STOCH（KDJ）/ BBANDS / RSI / ATR / MFI / OBV / ADX / SAR，用 numpy 数组调用（已实测），逐股算出宽表，不用自己实现指标。它自带的事件驱动回测不做主回测（保持选股与回测同口径），最多作为交叉核对 |
| **lightgbm** | 4.7.0 | 退役 | 新池不用；旧三池停用后与 lgbm 代码一起删除（另起提交） |
| **长桥 CLI** | 本机已登录 | 程序可调用：择时数据 + 实盘体检 | `longbridge market-temp CN --history`：2018+ 逐日市场温度 / 估值 / 情绪，可回测，作为择时候选。每日选股后，`subprocess` 调 `capital`（资金流）、`institution-rating`、`news` / `filing`（`--format json`），把入池股的资金流、评级、近期公告写进简报"体检"栏 |
| **MCP（长桥 / 通达信 / 同花顺）** | 仅对话中 | 规则来源 + 人工复核 | 程序调用不了，没有可回测的历史。用途：① 提供选股规则和阈值（长桥预设策略、通达信 / 同花顺常用条件）；② 我在对话里对股池做深度复核（新闻、公告、研报） |

## 二、qlib 因子和 MCP 选股规则：结合用

- MCP 规则的来源是平台的预设策略（格雷厄姆、高质量盈利、CANSLIM、高股息、MACD 零轴上金叉、KDJ 低位金叉、放量突破、主力流入等）。它们只能查"今天谁符合"，没法回测，直接照用就不知道历史上是否赚钱。实测长桥这些预设策略近 3 个月多数为负。
- 做法：
  1. 把每条 MCP 规则翻译成本地条件，用 qlib / baostock / akshare 的历史数据计算。
  2. 和 qlib 的因子（Alpha158、表达式）一起，在研发区 2018~2022 做**条件事件研究**：触发后 5 / 20 / 60 日的超额、胜率、盈亏比、逐年稳定性、阈值邻域平稳性。
  3. 只保留稳定有效的条件组成买点 / 卖点。
- 一句话：**规则思路取自 MCP，计算和验证用 qlib，最后用 MCP 和长桥 CLI 做实盘复核。**

## 三、选股流程（每个交易日收盘后，数据最新交易日 T）

```
A 股全市场（qlib all，当日在市）
  │ ① 剔除烂股（两池共用，exit 规则：已持有的也强制卖出）
  │ ② 账户板块权限（global.yaml account.boards: [main] 或加 cyb/kcb/bse，准入规则）
  │ ③ 各池流动性门槛（短线：日均成交额 ≥ 1 亿；中长线：≥ 5000 万）
  ├──► 短线进攻池：短线买点规则 → 按优先分排队 → 上限 10 只
  └──► 中长线持有池：中长线买点规则 → 按优先分排队 → 上限 10 只
  每日：先检查持仓的卖出点 → 卖出 → 空位从当日满足买点的候选中按优先分补入
```

**① 烂股剔除清单**（全部可按历史时点判定，研发区核对剔除比例和效果）：
- ST / *ST / 退市整理（baostock 历史 isST，修掉现在"只有当前快照"的近似）。
- 上市不满 1 年。
- 近 20 日停牌。
- 股价 < 2 元（面值退市风险）。
- 总市值 < 30 亿（市值退市风险、操纵风险）。
- 扣非 / 归母净利润连续两年为负，或净资产为负。
- 营收 < 3 亿且亏损（退市新规口径）。
- 近 1 年有立案调查、监管警示或风险提示公告（akshare 公告类型）。
- 未来 20 日有大比例解禁（> 流通股 5%）：只限制买入。

**② 板块权限**：从池级配置移到 `config/global.yaml` 的 `account.boards`，两池共用；改一处，两池同时生效。

**两池关系**：资金独立、各自回测；默认允许同一只股票同时在两个池里（逻辑不同），可配置为互斥。

**不足 10 只**：保持买点纪律，空位留现金；`fill: always` 时用优先分最高的候选补满（研发区对比两种做法）。

## 四、两个池的规则（初始候选，以事件研究结果定稿）

**中长线持有池**（持有数周到数月）
- 买点（同时满足）：
  - 质量：ROE ≥ 8~12%，经营现金流 / 净利润 ≥ 0.5，净利润同比 > -20%。
  - 估值：行业内 PE / PB 排名不在最贵 30%（行业取 baostock 证监会分类）。
  - 至少一项加分：股息连续、业绩预增、股东户数下降。
  - 趋势：站上 MA120 且收盘 / MA20 < 1.10（不追高）。
- 优先分：估值 + 质量 + 成长 + 低波动 + 中期动量，截面排名等权合成。
- 卖点（任一触发）：
  - 新财报质量门槛不通过。
  - 估值进入行业最贵 10%。
  - 跌破 MA250。
  - 从持有期最高点回撤 20%。
  - 减持或立案事件。
- 约束：单一行业 ≤ 3 只。

**短线进攻池**（持有数天到两周）
- 买点：研发区对比几类规则，选优或合用。
  - A 强势缩量回踩：站上 MA60，近 5 日回调 3~12%，缩量，K 线收盘位置转强。旧研究里反转 / 缩量方向有效。
  - B 放量突破：20 日新高，成交额 ≥ 1.5 倍均量，CMF > 0，近 5 日涨幅 < 20%。
  - C 指标共振：MACD 零轴上金叉 + KDJ 低位金叉 / RSI 金叉（akquant.talib）。
  - 加分：龙虎榜机构净买入、融资买入放大。
- 优先分：资金面近似指标（CMF / MFI / OBV 斜率 / 收盘相对均价 / 量比）+ 信号强度。
- 卖点（任一触发）：
  - 止损 -5% 或 2×ATR。
  - 从最高点回撤 8%。
  - 跌破 MA10。
  - 持有满 10 个交易日。
- 择时：全市场宽度低、指数在 MA60 下或长桥市场温度过低时，停止开新仓。

成交：T 日收盘出信号，T+1 收盘成交；短线另用 T+1 vwap 对照，按两者中较差的验收。止损在次日收盘成交，不是盘中触价，简报和 RULES 里写明。

## 五、实现结构（复用现有分层：注册表 + yaml 声明 + 选股与回测共用纯函数）

```
config/global.yaml          + account.boards；+ quality_floor（烂股清单参数）
config/pools/short.yaml     universe: all，max_positions: 10，points.buy / points.sell / path 规则
config/pools/long.yaml      同上
config/pools/hs300|zz500|zz1000.yaml → enabled: false
stockanalysis/
  data/baostock_data.py     新：逐日 ST / 停牌 / 估值 / 换手，季频财报 pubDate，业绩预告
  data/events.py            新：两融、龙虎榜、股东户数、解禁、增减持 / 回购、公告排雷（akshare 缓存）
  data/longbridge_cli.py    新：调用长桥 CLI（市场温度历史、实盘体检），失败只告警不中断
  features/technical.py     新：akquant.talib 逐股指标 + qlib 表达式 → 宽表
  features/events.py        新：事件 → 逐日面板（严格按公告日可见）
  features/regime.py        新：市场宽度、涨跌停家数、指数趋势、市场温度 → 择时状态
  selection/filters.py      + 烂股规则（exit）、解禁回避（entry）；boards 改读 account.boards
  selection/points.py       新：买点 / 卖点规则注册表（面板化，yaml 声明）
  selection/positions.py    新：纯函数状态机 step_positions()（止损 / 移动止盈 / 最长持有 / 排队补位 / 行业上限 / 择时），选股与回测共用
  models/scorers.py         + multifactor（优先分：截面排名等权合成）
  backtest/strategies.py    + signal_points（调用 step_positions）
  selection/store.py        CSV / 简报：买入原因、买入价、当前卖出线（止损价 / 止盈价 / 均线价 / 剩余天数）、卖出原因、接近买点名单、长桥体检栏
  research/event_study.py   新：条件事件研究
  research/points_grid.py   新：规则参数网格 + 研发区滚动验证（2018~20→21，2018~21→22）
  cli/                      update-data 加 baostock / events；新增 event-study、points-grid
```

## 六、分阶段执行（每阶段单独提交，⛔ 处汇报后再继续）

0. **计划入库与进度跟踪**：本计划写入项目根目录 `REFACTOR_PLAN.md`，末尾加"进度跟踪"表（阶段 / 任务 / 状态 ☐◐☑ / 提交 / 备注），每完成一项就更新并随该阶段一起提交。在 main 上新建分支 `refactor/rule-pools` 执行。
1. **数据层**：baostock、akshare 事件数据、长桥 CLI 封装。全市场全历史首次下载约数小时，后台运行。防未来单元测试。
2. **烂股剔除 + 板块权限**：规则落地；研发区统计每年剔除只数，以及剔除组与保留组的后续收益差。⛔ 汇报剔除效果。
3. **指标与事件研究**：technical / events / regime；把 MCP 规则翻译后与 qlib 因子一起做事件研究。⛔ 汇报哪些规则稳定有效，定两池候选规则。
4. **买卖点框架**：points / positions / signal_points / 选股输出 / 选股与回测一致性检查，端到端跑通。
5. **中长线池**、6. **短线池**：研发区网格 + 滚动验证。⛔ 各自汇报、定参数。
7. **冻结 → 保留集评估一次 → 定状态**：
   - 扣成本后绝对年化 > 0 且超额 > 0，研发区和保留集都满足才标 production，否则 observe。
   - 删旧三池和 lgbm 代码；更新 README / RULES / ARCHITECTURE；生成首期股池并用 MCP 复核。

"两个都要赚钱"不能保证；不达标的池如实标 observe 继续跟踪。

## 七、验证
- 单元测试：
  - 烂股规则按历史时点判定，例如 ST 不用当前快照。
  - 事件面板在公告日前不可见。
  - 指标与 akquant.talib 对数。
  - 状态机：止损、移动止盈、最长持有、10 只上限、排队、行业上限、择时、涨跌停顺延。
  - yaml 校验。
- `C:/Users/merak/anaconda3/envs/qlib/python.exe -X utf8 -m pytest tests -q` 全部通过。
- `-m stockanalysis check --pool short` / `--pool long`：选股与回测逐日持仓一致。
- `-m stockanalysis event-study` / `points-grid`：研发区报告。
- `-m stockanalysis select --pool short,long`：两池各 ≤ 10 只，简报含买卖点和长桥体检栏。

## 八、复用
`features/fundamental.build_fundamental`、`data/fundamental`（缓存 / 重试 / 代码规范化）、`research/metrics`（rank_ic / summarize_ic / topk_excess）、`registry.Registry`、`selection/filters.py` 现有规则、`backtest/engine.run_backtest`、`backtest/metrics.segment_metrics`、`research/consistency.py`、`data/sync.py`（qlib 数据自动更新）。

## 九、阶段 8~9：数据层重整与选股规则优化（2026-10-17 增补）

2026-10-17 数据体检与选股方式评审后，用户的决定：
- 数据：修错误 + 重组目录。暂不补充新数据，风格指数、准确市值、历史行业留作以后。
- 选股优化，四项都做，全部只在研发区（≤ 2022-12-31）验证：
  - long：大类上限、EP + BP 估值、业绩恶化预警。审查后补了两项代码问题（事件类卖点每日检查、单票权重上限），估值口径扩为 EP / 3 年平均 EP / EP+BP 三选一；
  - short_reversal：开盘成交、行业上限；
  - 烂股：市值门槛对比；
  - 实盘跟踪报告，同时停止 breakout 每日出池。
- 顺序：先数据、后规则。阶段 8 完成并核对选股、回测结果不变后，再进入阶段 9。

体检结论：
- 数据本身完整、正确。
- 实时数据有三处滞后或失败，程序不提示：
  - 10-16 两融只有沪市；
  - baostock 只到 10-15；
  - 东财快照一直断连，`exclude_st` 只好退回用财报名称判断。
- 中证1000 缺 2025-04-16 一天，基准收益因此丢了两天。
- 存放乱：
  - akshare 数据分散在 `fundamental/` 和 `events/`，命名不一致；
  - 快照每天新增一份；
  - 公告 3577 个日文件，读一遍 8 秒、1.87GB；
  - baostock 5481 个文件，只用 `is_st`，读一遍 34 秒；
  - `paths.py` 说明过时。

执行分支 `refactor/data-and-rules`（自 main），每个小步单独提交。

### 阶段 8：数据层重整（选股、回测结果必须不变）

#### 目标目录

```
data/
  pool/  reports/                     不动
  raw/                                可重下载的原始缓存（不入库），统一 <数据集>/<单元>.parquet
    akshare/   yjbb xjll fhps yysj yjyg gdhs/<报告期>  margin notice/<yyyymmdd>  lhb lhb_jg/<yyyymm>
               release/<yyyy>  ggcg repurchase spot/<yyyymmdd>（快照只留最新）
    baostock/  daily/<qlib_code>  industry/<yyyymmdd>（只留最新）
    longbridge/market_temp/<yyyy>
  panel/                              由 raw 生成的紧凑表（不入库，可重建）
    is_st.parquet  risk_notice.parquet  （各带 .meta.json）
```

原始单元文件不合并：缓存是否完整按文件修改时间判断（`fundamental.cache_valid`、`events._valid`），合并会破坏这套逻辑。

#### 改动

1. **`paths.py`**
   - 新增 `RAW_DIR / AKSHARE_DIR / BAOSTOCK_DIR / LONGBRIDGE_DIR / PANEL_DIR`；
   - 删除 `FUND_DIR / EVENTS_DIR`；
   - 重写顶部说明。
2. **路径改动**
   - `data/fundamental.py`、`data/events.py`、`data/baostock_data.py`、`data/longbridge_cli.py`、`features/fundamental.py`（fhps 的 glob）改用新路径；
   - fundamental 的文件名去掉 `<kind>_` 前缀，`_load_periods` 改用 `p.stem`。
3. **快照只留最新**：`events._save_snapshot(df, dataset)` 写入今天的文件并删除旧的。`fetch_ggcg / fetch_repurchase / fetch_spot / fetch_industry` 共用。
4. **两融不完整不落盘**：`fetch_margin` 在沪深任一边为空时抛出 `IncompleteError`。`update_events` 已有失败记录，日志写"未公布 / 不完整，下次重抓"。
5. **新增 `data/panel.py`**
   - `build_if_stale(name, src_files, builder, extra_meta)`：meta 记录源文件数、最大 mtime 和额外键（如风险关键词正则）。源或额外键变了就重建，否则直接读。
   - `is_st_panel()`：生成 int8 宽表。
   - `risk_notice_panel()`：只保留命中 `features/risk.RISK_TITLE`、且不命中 `RISK_TITLE_EXCLUDE` 的公告。正则仍只定义在 risk.py。
   - 调用方：`features/risk.st()` 改读 is_st panel；`risk.risk_events()` 改读 risk_notice panel，按 start 截取。
   - `update-data` 结束时重建 panel。
6. **去掉 `exclude_st`**
   - 删除 `filters.exclude_st`、`ctx["st_codes"]`、`signal.build_ctx` 中的取值、`reference.current_st_codes`；
   - 修改 `pool_defaults.yaml`、`RULES.md` 第 2 节、`tests/test_filters.py`；
   - `name_map` 保留快照优先、财报名称兜底。
7. **基准缺日**：`backtest/points_engine._bench_returns` 先对收盘价做 ffill 再算收益（缺日收益记 0，次日拿到完整涨跌），仍打印缺日告警。
8. **新增 `data/freshness.py`：数据截止日**
   - 对比信号日 T，列出 qlib 日历、baostock（is_st panel 末日）、公告、两融（到 T-1 即正常）、龙虎榜、市场温度、各快照日期、数据版本；滞后的标 ⚠。
   - 接入两处：
     - 简报元信息下加"数据截止"，有 ⚠ 时展开成表（`selection/points_daily.py`）；
     - `update-data` 结束时打印，并新增 `update-data --status`，只看状态、不下载。
9. **一次性迁移**
   - 脚本放在 scratchpad，不入库；用 `os.replace` 移动文件，保留 mtime。迁移内容：fundamental、events、baostock、market_temp、industry 移到新位置，快照只留最新，删除空的旧目录。
   - 迁移前后核对文件数、行数、mtime。
10. **其他**：`.gitignore` 改为忽略 `data/raw/`、`data/panel/`；README、ARCHITECTURE、各 data 模块顶部的布局说明同步修改。

#### 阶段 8 验证

- 新增单元测试：
  - 快照删旧；
  - 两融单边为空时抛异常且不落盘；
  - `build_if_stale`：源变化、正则变化时重建；
  - is_st panel 与逐股读取结果相同，risk_notice panel 与原 `risk_events` 结果相同；
  - freshness 的 ⚠ 判断，以及两融 T-1 视为正常；
  - 基准 ffill。
- 全部测试通过，pyflakes 无告警；grep 确认没有残留的 `FUND_DIR / EVENTS_DIR / exclude_st / current_st_codes`。
- 补数据：后台跑 `update-data --only baostock`，补 10-16；再跑 `--only events --events margin`，确认如实报告深市"不完整"。
- 结果不变：
  - `select --pool all --date 2026-10-16`：持仓与次日计划和已入库 CSV 一致；
  - long 研发区回测年化仍为 15.68%（基准修复只影响超额，按实际变化记录）；
  - `check` 0 差异。
- 记录改造前后的读取耗时与内存。

### 阶段 9：选股规则优化

#### 9.0 先做：实盘记录不可改写（其余各项改 yaml 的前提）

问题：每日选股都按**当前** yaml 从 `live_start` 重新推演。阶段 9 一改参数，2026-10-15 以来的"实盘"持仓就会被改写；如果实盘跟踪也用重新推演来算，就成了自己验证自己。

做法：
- **实盘的唯一依据是已入库的每日 CSV**，即当天实际发布的持仓和计划。
- 每次规则改动开一个新的跟踪段：
  - 改 yaml 的同时把 `live_start` 设为改动生效日；
  - 在 `points.live_log` 记录各段的起止日期和 config_hash。
- 旧段的 CSV 不动，`track` 把各段按时间拼接起来。

#### 验收规则（事先定好，分两类）

研发区只跑 `points-grid --starts 6`，与当前冻结参数（基线）比较。

**一次只改一项，与基线比较**：通过的几项再一起组合确认一次，不做全因子网格，以免比较次数太多、把噪声当成改进。

**A 类：信号质量类**（估值口径、业绩预警卖点）
- long 5 年只有 22 笔交易，组合层面的差异多半是噪声。
- 所以先在**候选集**上检验：每天满足买点的约 140 只股票等权，或用事件研究；样本大，要求 5 年中至少 4 年更好。
- 候选集通过后，再看组合层面：合成年化不低于基线 1 个百分点以上，滚动验证两折都不差于基线。

**B 类：风控类**（大类上限、单票权重上限、行业上限）
- 按风险指标判断：
  - 最差起点年化、最大回撤、单票最大贡献占比要改善；
  - 年化下降不超过 1 个百分点。
- 不要求收益提高。2020~2022 的收益主要来自煤炭股集中持仓，风控类改动降低收益是正常的。

**通用**
- 相邻参数也要满足，看邻域，不按单个冠军取参数。
- 通过后才改 yaml，并按 9.0 开新跟踪段。
- 保留集只跑一次作参考，报告注明"已看过，不作独立检验"。状态（production / observe）不因此改变。

#### 9.1 long

**① 事件类卖点改为每日检查**（B 类；代码里发现的问题，优先做）
- 现状：`positions.sell_reasons` 中的条件卖点（`forecast_negative`、`industry_expensive`）和 `rank_exit` 只在月末检查日判断。业绩预减公告后，最多要等约 20 个交易日才卖出。这是"卖点偏迟"的一大原因，新增的预警卖点也会有同样的延迟。
- 做法：
  - `PointsSpec` 增加 `sell_daily: [条件名]`，列在其中的条件每日检查，其余仍按 freq；
  - long 设为 `sell_daily: [forecast_negative]`，估值和 rank 类卖点仍按月检查。
  - 对比：月末检查 vs 每日检查。

**② 单票权重上限**（B 类；代码里发现的问题）
- 现状：`points_engine.account` 只在买入时按"净值 / 10"建仓，之后从不再平衡。long 平均持有 592 天，兖矿一只涨了 354%，可能占到净值的三成以上。"10 只等权"只在建仓那一刻成立，单票集中的风险比行业集中更大。
- 做法：
  - `PointsSpec.max_weight`：检查日超过上限的减到目标权重；减仓卖出的部分由下一次买入补足。
  - 只改记账，不改模拟器的持仓集合，前缀一致性检查不受影响。
  - 简报持仓表增加"当前权重"一列。
  - 网格取 max_weight ∈ {不设, 20%, 25%}。

**③ 大类上限**（B 类）
- 东财行业名带"Ⅱ"，就是申万二级，因此用申万官方的二级 → 一级映射，不手工拼大类：
  - 新增 `config/sw_l1.yaml`，未映射的行业自成一类；
  - 新增 `PointsSpec.sector_max`，照 `ind_count` 的方式计数。
- 要事先说明效果有限：
  - 申万一级里"银行"和"非银金融"是两类，当前 5 只金融股（3 银行 + 2 非银）在 sector_max=3 下不会被拦；
  - long 真正集中的是"高股息 / 低估值"风格（银行、煤炭、电力），大类上限只能缓解一部分。
- 网格取 sector_max ∈ {2, 3, 不设}；另加一个可选合并组 `大金融`（银行 + 非银金融），作为对照。

**④ 估值口径**（A 类）
- 原方案只有 EP + BP。但 BP 解决不了原来想解决的"周期股盈利高点时 EP 最便宜"问题，而且会偏向 PB < 1 的银行和地产。2021~2022 年地产就是高 BP 陷阱。
- 改为在行业内比较三种口径：
  - EP（基线）；
  - 3 年平均 EP：用近 3 个年报 EPS 的均值，平滑周期，直接针对盈利高点的问题；
  - EP + BP 各半。
- 实现：`_industry_ep_rank` 增加 `basis: ep | ep3y | ep_bp`；`industry_cheap / industry_expensive` 透传，默认 `ep`，旧配置结果不变。
- `ep3y` 由 `features/fundamental` 按年报计算，按披露日次日可见。

**⑤ 业绩恶化预警**（A 类，在 ① 之后做）
- 新条件 `profit_deteriorate {yoy}`：最近已披露报告期的净利润同比 ≤ yoy。
- 注意：
  - 净利润同比是年初累计口径；
  - 年报和半年报通常已有业绩预告在先，这个条件主要在一季报、三季报上提前。
- 列入 `sell_daily`；网格取 {不用, -30%, -50%}。

**测试**
- `sell_daily` 在非检查日也会触发；
- max_weight 减仓后记账自检通过；
- `sector_max` 状态机；
- `basis=ep` 与旧结果完全相同；
- `ep3y` 和 `profit_deteriorate` 都按披露日次日可见。

#### 9.2 short_reversal：开盘成交 + 行业上限

- **开盘成交**
  - 先做事件研究，验证前提：反转信号触发后，T+1 的隔夜收益（T 日收盘到 T+1 开盘）和 T+1 日内收益分别是多少。只有隔夜收益明显为正，"开盘买"才有意义；否则跳过这一项。
  - 实现：`DEAL_PRICES` 增加 `open`，fill 取 `ctx["open"]`。
  - 买和卖都按开盘价成交，口径一致：持有满 5 天、止损都在次日开盘卖出。
  - 可行性：开盘价达到涨停价（按前收盘和板块涨跌幅计算）的买不进，达到跌停价的卖不出。
  - 开盘集合竞价的冲击更大，open 方案的滑点从千 1 提到千 2 比较。
  - 网格对比 close / vwap / open。
- **行业上限**（B 类）：网格取 industry_max ∈ {2, 3, 不设}。
- **测试**：开盘涨停买不进、开盘跌停卖不出、开盘成交的记账。

#### 9.3 烂股：市值门槛对比

- **只按尾部风险判断，不按收益判断**：
  - 研发区小市值跑赢，是 2018~2022 微盘股行情的结果；
  - 2024 年起退市新规收紧（市值退市门槛提高、ST 范围扩大），小市值的尾部风险在研发区里体现不出来。
  - 按研发区收益放宽门槛，等于去追小盘因子。
- 做法：
  - 用 `quality` 命令对比 15 亿、20 亿、30 亿三档的剔除组 / 保留组，看 60 日大跌率、退市 / 戴帽比例；
  - 收益只列出，不作为依据。
- 结论：只有放宽后保留组的大跌率不上升，才改 `pool_defaults`；否则维持 30 亿，只出报告。
- 市值仍用"净利润 / EPS 推算股本"的近似，报告中注明。

#### 9.4 实盘跟踪报告 + 停 breakout

- **新命令 `track --pool all`**（`cli/track.py`，接入 `cli/__init__.COMMANDS`），每个池输出 `data/reports/track_<pool>.md`：
  - **实盘净值来自已入库的 CSV**：逐日持仓和计划买卖，按 T+1 收盘价和成本记账；各跟踪段（9.0）按时间拼接；
  - 对照项：用当前配置重新推演同一区间，两者的差异来自数据修订或规则改动，分开列出；
  - 与回测预期对比：研发区、保留集的月度收益分布，实盘落在第几分位；
  - short_reversal 增加一条提示：实盘超额低于预期区间下沿时，建议降为 observe。
- **short_breakout**：改为 `enabled: false`，状态仍是 observe，配置和历史股池保留；README、RULES 说明它已停止每日出池，作为对照的结论保留。

#### 阶段 9 顺序

1. 9.0 跟踪段 + 9.4 track、停 breakout：先把实盘记录固定下来。
2. 9.1 ①（事件类卖点每日检查）、②（单票权重上限）：代码里发现的问题。
3. 9.1 ③ ④ ⑤、9.2、9.3：逐项研究，一次只改一项。
4. 组合确认 → 改 yaml，开新跟踪段 → 保留集参考跑一次。

#### 阶段 9 交付

- 网格报告：`data/reports/points_grid_{long,short_reversal}_v4.md`、市值门槛对比报告。
- 采纳的参数改进 yaml；RULES 第 3、4、5、7 节同步更新。
- 首个实盘跟踪报告。
- 进度表与日志更新；测试全过；是否合并 main 由用户确认，不推送远端。

### 阶段 8~9 复用

- `research/points_grid`：多起点、滚动验证、邻域判断。
- `backtest/runner.backtest_pool / backtest_window`。
- `research/quality_check`：剔除效果统计。
- `selection/positions` 中的 `ind_count`：行业上限计数，大类上限照此实现。
- `features/fundamental` 的 `bp / netprofit_yoy / _ttm / _latest_visible`：已有，ep3y 照此计算，不用新数据。
- `research/event_study`：用来检验候选集和隔夜收益。
- `points_engine.account` 的记账自检：max_weight 改动之后仍要通过。
- `features/risk.RISK_TITLE`：风险关键词唯一定义处。
- `fundamental.cache_valid`、`events._valid`：缓存完整性判断，不改。
- `sync.data_version`：数据版本。

---

## 进度跟踪

状态：☐ 未开始 · ◐ 进行中 · ☑ 完成 · ⛔ 等待用户决定。每完成一项即更新本表，并随该阶段提交。

| 阶段 | 任务 | 状态 | 提交 | 备注 |
|------|------|------|------|------|
| 0 | 计划入库、建分支 `refactor/rule-pools` | ☑ | | |
| 1 | `data/baostock_data.py`：逐日 ST / 停牌 / 估值 / 换手（全市场，断点续传） | ☑ | ea26909 | 多进程下载 5481 只，2026-10-17 完成，失败 0；换网络后可直连 |
| 1 | 财报真实披露日、业绩预告 / 快报 | ☑ | | 改用 akshare：`stock_yysj_em` 实际披露日（2013+）、`stock_yjyg_em` / `stock_yjkb_em`（带公告日）；可见日 = 披露日次日；修复报告期缓存永不更新的问题 |
| 1 | `data/events.py`：两融、龙虎榜、股东户数、解禁、增减持 / 回购、公告排雷 | ☑ | | 立案调查公告归在"其他"类，须抓全部公告按标题识别 |
| 1 | `data/longbridge_cli.py`：市场温度历史、实盘体检封装 | ☑ | | 市场温度单次区间上限 3 年，按年缓存 |
| 1 | `update-data` 命令接入新数据源；防未来单元测试 | ☑ | | `--only fundamental,events,baostock,market_temp`，`tests/test_data_layer.py` |
| 1 | 全量历史下载（后台） | ☑ | | 全部完成，各数据集 0 失败（2026-10-17） |
| 2 | 烂股剔除规则（exit）+ 解禁回避（entry）；`account_boards` 全局板块权限 | ☑ | | `features/risk.py` + 8 条过滤规则写入 pool_defaults；`tests/test_risk.py` |
| 2 | 研发区剔除效果统计（每年剔除只数、剔除组 vs 保留组后续收益） | ☑ | | 正式版 quality_report_long_final.md：剔除组平均超额更高但大跌率更高，定位为尾部风险控制 |
| 3 | `features/technical.py`（akquant.talib + qlib 表达式） | ☑ | a4a456d | akquant.talib 逐股计算、停牌缺口不传染；量能统一用成交额（qlib volume 口径不一致） |
| 3 | `features/events.py`、`features/regime.py` | ☑ | a4a456d | |
| 3 | `research/event_study.py` + `event-study` 命令；MCP 规则翻译清单 | ☑ | a4a456d | 翻译清单 = `features/conditions.py`（54 条，含阈值邻域）；2018H1 冒烟通过 |
| 3 | 研发区事件研究报告 | ☑ | 694d857 | 正式版 event_study_{long,short_reversal}_final.md，结论与初步版一致 |
| 4 | `selection/points.py` 买卖点规则注册表 | ☑ | | 规则直接引用条件库（CONDITIONS）+ yaml points 段，加载时校验条件名与参数 |
| 4 | `selection/positions.py` 状态机 `step_positions()` | ☑ | | `decide()` + `simulate()`；`tests/test_positions.py` |
| 4 | `multifactor` 优先分、`signal_points` 回测策略、schema / 配置 | ☑ | | 回测改为自记账（`backtest/points_engine.py`），见进度日志 |
| 4 | 选股输出（买入原因、卖出线、接近买点、长桥体检栏）；一致性检查 | ☑ | | `selection/points_daily.py`；选股与回测是同一个 simulate()，一致性由构造保证 + 记账逐日自检 |
| 5 | 中长线池 `long`：规则落地 + 研发区网格 + 滚动验证 | ☑ | e550214 | 网格 v1~v3 + 均价对照；按参数邻域定稿：月度、无移动止盈、rank_exit 0.3、industry_cheap 0.7 |
| 6 | 短线反转池 `short_reversal`：规则落地 + 研发区网格 + 滚动验证 | ☑ | e550214 | 网格 v1（81）+ v2（32，择时 / 均价）；定稿：急跌 / RSI 低位，持有 5 天，止损 8%；邻域不稳，结论偏弱 |
| 6 | 短线突破池 `short_breakout`（对照）：规则落地 + 研发区网格 + 滚动验证 | ☑ | e550214 | 网格 v1（24）全部为负；定稿：止损 5%、持有 10 天，observe 对照 |
| 7 | 冻结 → 保留集评估一次 → 定状态 | ☑ | e550214 | long / short_reversal production，short_breakout observe（RULES.md 第 5 节） |
| 7 | 删旧三池与 lgbm 代码；更新 README / RULES / ARCHITECTURE；首期股池 + MCP 复核 | ☑ | 65a0737 | ec49829 / 5d4a4d1 删除；文档改写；首期股池 2026-10-16 + 长桥体检复核；check 前缀一致性 0 差异 |
| 8 | 8.1 计划入库、建分支 `refactor/data-and-rules` | ☑ | | |
| 8 | 8.2 目录与命名（raw/akshare、raw/baostock、raw/longbridge、panel）+ 迁移；8.3 快照只留最新、两融不完整不落盘 | ☑ | | 迁移 12025 个文件，行数 / mtime 全部一致；删除 2 个旧快照 |
| 8 | 8.4 panel 派生表（is_st、risk_notice） | ☑ | | 与旧读取逐格相同；is_st 30 秒 → 0.4 秒，风险公告 8 秒 / 1.87GB → 0.1 秒 / 7MB（首次构建 30 秒 / 15 秒） |
| 8 | 8.5 去掉 exclude_st；基准缺日 ffill；数据截止日（简报 + `update-data --status`） | ☑ | | 重抓两融发现 10-16 沪市当时也不完整（1675 → 2000 行），新规则正确拒存 |
| 8 | 8.6 文档；核对选股 / 回测结果不变 | ☑ | | 10-16 三池持仓与计划与已入库一致；long 研发区 15.68% / 超额 15.53% 不变；check 0 差异；baostock 补到 10-16，失败 0 |
| 9 | 9.0 跟踪段（live_log）+ 9.4 `track` 命令、停 short_breakout | ☑ | | 跟踪段存于 `data/pool/<池>/segments.json`（存档完整配置）；改规则须 `track --new-segment`；首份报告与已发布股池一致 |
| 9 | 9.1 ① 事件类卖点每日检查（sell_daily）② 单票权重上限（max_weight） | ◐ | | 代码与测试完成，待研发区网格 |
| 9 | 9.1 ③ 大类上限（申万一级）④ 估值口径（ep / ep3y / ep_bp）⑤ 业绩恶化预警 | ◐ | | 代码与测试完成（config/sw_l1.yaml、sector_max / sector_groups、basis、profit_deteriorate），待研究 |
| 9 | 9.2 short_reversal 开盘成交 + 行业上限 | ◐ | | `deal_price: open` 代码与测试完成，待隔夜收益检验与网格 |
| 9 | 9.3 市值门槛对比（只看尾部风险） | ☐ | | |
| 9 | 组合确认 → 改 yaml、开新跟踪段 → 保留集参考一次；RULES 更新 | ☐ | | |

### 进度日志

- 2026-10-16：计划确认，建分支，开始阶段 1。
- 2026-10-16：阶段 1 数据层代码完成并开始后台下载；阶段 2 烂股规则落地。发现与调整：
  - 业绩报表里的股票简称是**当前**名称（15 年里只有 1 只股票名称变过），不能用于历史 ST，历史 ST 必须用 baostock isST；
    baostock 数据到位前，用"主板 ±5% 涨跌停"近似（只覆盖主板）。
  - 财报真实披露日改用 akshare 预约披露接口（实际披露时间），不依赖 baostock 季频接口。
  - 板块权限改为 global.yaml 的 `account_boards`（`account` 已是资金额，不复用）。
  - 两个新池 `config/pools/long.yaml` / `short.yaml` 先以 `enabled: false` 占位，阶段 4 接入买卖点框架后启用。
  - 2018H1 冒烟：ST / 小营收亏损 / 风险公告命中组后续超额为负、大跌率高；市值 < 30 亿的命中组反而跑赢，全量结果出来后重点复核。
- 2026-10-16：阶段 3 代码完成（全量研究待数据），阶段 4 买卖点框架完成。设计调整：
  - **买卖点池的回测不走 qlib 策略**：改为纯函数模拟器 `selection/positions.simulate()` + 自记账 `backtest/points_engine.py`
    （T+1 收盘成交、涨停买不进 / 跌停卖不出 / 停牌顺延、佣金 + 印花税 + 滑点、最低佣金）。原因：带止损 / 移动止盈的持仓状态
    要和 qlib 的成交结果逐日对账，容易两边漂移；同一个模拟器同时用于每日选股和回测，结果按构造一致，记账时逐日自检持仓。
    报告列与 qlib report 相同，沿用 `segment_metrics`。未建模：整手取整、成交量上限（单只约净值 1/10，流动性门槛以上影响小）。
  - 每日选股从 `points.live_start` 起用模拟器推演到当日，不读写上期状态文件；简报价格一律显示不复权价。
  - 冒烟发现初始短线规则"缩量回踩买入 + 跌破 MA20 卖出"互相矛盾（回踩股多在 MA20 下方，次日即卖），需在网格中解决。
- 2026-10-16：初步事件研究（研发区 2018~2022，主板，两融 / 公告数据未下完）。要点：
  - 本地中证全指 SH000985 缺 667 个交易日，不能做基准；长线池改用中证800（SH000906），短线池改用中证1000（SH000852）。
  - 追涨类条件在 5 / 20 / 60 日全部为负且逐年稳定为负：放量突破、均线多头、MACD 零轴上金叉、BOLL 升穿上轨、近期持续上升、
    放量、"主力流入"（CMF / 尾盘强势）、龙虎榜机构净买入、追高。A 股短线以反转为主，与旧研究一致。
  - 反转类（急跌、超跌、跌穿布林下轨、RSI 低位金叉）平均超额为正，但逐年只有 1/5 年为正，不稳定。
  - 稳定为正（4~5/5 年）：行业内便宜、高股息 / 分红连续、业绩预增、股东户数大降、小盘成长；
    连续因子：股息率 / 分红连续性 / EP / 现金流质量 IC 5/5 年为正，波动率 / 近 20 日涨停次数 / 近 20 日涨幅 5/5 年为负。
  - 格雷厄姆 / 绝对低估值在 2018~2022 略负（价值陷阱），"行业内便宜"为正。
  - 站上均线（MA60 / MA120）做买点条件为负；择时指标与后续收益无单调关系（宽度极低时反而反弹）。
- 2026-10-16：用户决定（初步事件研究后）：
  - 中长线持有池：质量价值 + 分红 + 预期（行业内便宜 + 分红连续为必要条件，业绩预增 / 股东户数大降 / 高股息 / 高 ROE 至少一项；
    不用均线择时；卖点：行业内变贵 / 业绩预减 / 优先分跌出 / 大幅回撤）。
  - 短线建两个池：`short_reversal`（反转 + 事件，推荐方案）与 `short_breakout`（追涨突破，按用户原设想保留作对照，
    加强过滤：大盘在 60 日线上、不追高、质量门槛）。共三个池，各 10 只。
  - `--set` 改为叶子值整体替换（字典不再与原字典合并），参数网格才能替换 / 清空条件组。
- 2026-10-16：长线池网格（研发区 2018~2022，基准中证800，公告数据未下完）：
  - v1（单起点）年化 -0.5%~14%，但 5 年只有 11~40 笔交易、平均持有 200~600 天，结果很依赖建仓时选中的 10 只；
    同参数仅检查频率不同就差 4 个百分点，滚动验证 F2 选中方案 2022 年 -2.2%（全部方案中位数 +8.7%）。
  - 因此网格加了多起点（`--starts`，每隔 42 个交易日一个建仓日，合成净值）。v2（6 起点）：最好的几组年化约 9.6%~10.4%、
    超额约 9%~10%、最大回撤约 -23%~-27%、各起点年化 9%~17%（都为正）；共同特征：**不设或放宽移动止盈**（15% 移动止盈明显变差）、
    rank_exit 0.3~0.5、月度或周度检查。滚动验证：F1 选中方案 2021 年 +10.8%（中位数 +22.8%），F2 选中方案 2022 年 +1.0%（中位数 +7.4%）
    ——按 2018~2020 / 2018~2021 选出的"最优"在下一年都不如中位数，说明方案间差异多为噪声，不宜按网格冠军定稿。

### 暂停点（2026-10-16 23:50）与恢复步骤

当前状态：分支 `refactor/rule-pools`，测试 122 个全过；三个规则池均 `enabled: false`（未定稿，不会被 `select --pool all` 运行）。
旧指数池 hs300 / zz500 / zz1000 仍为 enabled（尚未停用，阶段 7 再处理）。

后台仍在运行（可断点续传，中断后重跑同一命令即可补齐）：
- `update-data --only events --events gdhs,release,ggcg,repurchase,lhb,margin`（两融，预计约 02:00 完成），日志 `data/reports/logs/update_events.log`
- `update-data --only events --events notice`（全部公告，预计约 04:30 完成），日志 `data/reports/logs/update_notice.log`

恢复时按顺序：
1. 确认两个下载完成；失败单元重跑：`python -X utf8 -m stockanalysis update-data --only events`（已完成的跳过）。
2. baostock：本机经 Clash Meta TUN 时大响应卡住，需给 `baostock.com` 加 DIRECT 规则后运行
   `python -X utf8 -m stockanalysis update-data --only baostock`（历史 ST / 换手率；未完成前历史 ST 用涨跌停近似）。
3. 正式版体检与事件研究（数据齐全后）：`quality --pool long`、`event-study --pool long`、`event-study --pool short_reversal`，
   对比 `*_prelim.md` 的结论是否变化。⛔ 向用户汇报烂股剔除效果（阶段 2 遗留）。
4. 长线池定参数：用 `--starts 6` 网格，选"参数邻域内表现都不差"的方案而非单个冠军（倾向：不设移动止盈或 25%、rank_exit 0.3~0.5、
   月度检查）；补测 buy 必要条件的阈值（industry_cheap 0.5/0.7/0.8、dividend_steady 2/3/4）和优先分因子组合。⛔ 汇报。
5. 短线两池网格（`short_reversal`、`short_breakout`，各 `--starts 6`）：止损 / 止盈 / 最长持有 / 买点组合；
   短线另用 T+1 均价成交做对照（尚未实现：points_engine 成交价目前固定为收盘价）。⛔ 汇报。
6. 阶段 7：冻结 → 保留集只评估一次 → 定状态 → 停用旧三池 → 写 README / RULES / ARCHITECTURE → 设置 `live_start` 并启用。

- 2026-10-17：第一次跑长线 v3 网格（36 方案）和短线反转 v1 网格（81 方案）时，两个网格和 4 进程 baostock 下载同时进行，
  05:56 两个网格都因内存不足崩溃（长线做到 12/36，短线做到 54/81）。改为：网格按信号键分组，只保留当前一组的 PointSignal；
  不计算简报用的面板；多起点切片不复制数组；每跑完一个方案就把结果写入 `data/reports/logs/points_grid_<池>.partial.csv`。
  两个网格改为先后跑（e23e146）。公告接口在节假日会报 KeyError '代码'，现在当作当天无公告（之前算作失败单元，共 81 个，重跑即补）。
- 2026-10-17：长线 v3 网格（月度检查，6 起点）与短线反转 v1 网格（81 方案，6 起点）完成：
  - 长线：近 5 年分红次数要求 2 / 3 / 4 次，结果完全相同；每日满足买点的候选中位数约 142 只，这条门槛不起筛选作用，
    实际起作用的是优先分排序。选出的是高股息价值组合（银行 / 电力 / 煤炭 / 石化），平均持有约 1.5 年，
    5 年每个起点平均只有约 9 笔交易，单只兖矿能源 +354% 贡献很大。邻域结论：不设移动止盈；rank_exit 0.3 优于 0.5；
    industry_cheap 0.7~0.8 优于 0.5（年化约 10.3%~10.4%，起点最差 9.7%，最大回撤约 -27%）。
    forecast_negative 卖点多在已大跌后才触发（-30%~-44%），卖点偏迟。
  - 短线反转：81 个方案中 25 个年化为正；最好约年化 6%、回撤约 -30%，年化成本约 4%~5%。
    2018~2021 选出的方案在 2022 年为 -18.9%，与全部方案中位数 -19.5% 相当，2022 年整体失效。
    四项买点组合（急跌 / 超跌 / 布林下轨 / RSI 低位）均值 -12.6%，最差；止盈不起作用；max_hold=20 略好。
  - 下一轮：短线反转 v2（加择时与均价成交对照）、短线突破 v1，先后在后台运行。
- 2026-10-17：短线反转 v2（32 方案）与短线突破 v1（24 方案）完成：
  - 突破池 24 个方案全部年化为负（均值约 -7%），与事件研究中追涨类条件为负的结论一致。
  - 反转池加择时后没有变好：不择时平均 +0.6%；指数站上 MA60 才开仓，回撤从约 -36% 降到 -25%，但年化变成 -1.8%；
    宽度 ≥ 30% 才开仓最差（-9.1%）。反转在弱市中反而有效，择时会错过这些时段。
  - 发现网格 bug：信号键没有包含 deal_price，均价成交方案复用了收盘价成交的 PointSignal，所以 vwap 和收盘价
    两列结果相同、对照无效（已修复，加了测试）。现在对反转池最好的 4 个方案和长线推荐参数重跑均价对照。
  - 均价成交对照（修复后）：长线推荐参数收盘价成交 10.41% / 均价成交 10.59%，影响可以忽略。短线反转最好的两组：
    B（急跌 + RSI 低位）持有 5 天，收盘价 5.6% / 均价 5.7%；C（六项任一）持有 20 天，收盘价 6.3% / 均价 5.2%。
    但 B 持有 20 天是 -3.2%，C 持有 5 天是 -6.2%，最好的参数换个持有天数就转为亏损，说明结果不稳，接近噪声。
- 2026-10-17：用户决定"都保留，先按计划完成整个项目，后面有需要再改"。阶段 7 完成：
  - 冻结参数（long：月度检查、不设移动止盈、rank_exit 0.3、industry_cheap 0.7；short_reversal：急跌 / RSI 低位，
    持有 5 天，止损 8%，不设止盈；short_breakout：止损 5%，不设移动止盈，持有 10 天）。
  - 保留集 2023-01 ~ 2026-10 一次性评估（单一起点 / 6 起点合成）：long +12.2% / +13.0%（超额 +7.6% / +8.8%）；
    short_reversal +8.2% / +7.4%（超额 +2.0% / +1.8%）；short_breakout -6.5% / -6.7%。按事先定的规则：
    long、short_reversal 标 production，short_breakout 标 observe。short_reversal 的参数邻域不稳，在 RULES / README 中注明。
  - 删除旧三池、LightGBM / linear 训练代码与 Alpha158 特征；README / RULES / ARCHITECTURE 改写。
  - 收尾时发现并修复：窗口最后一天从不被标为检查日 → 月度 long 池实盘月末不会买入或条件卖出（回测不受影响）。
    `check` 对买卖点池改为前缀一致性检查（数据截止到 T vs 延续到之后），三池 0 差异。
  - live_start 2026-10-15，首期股池基于 2026-10-16 数据生成（long 10 只：银行 3、保险 1、券商 1 等；short_reversal 10 只
    均为 10-15 美国 CPI 超预期后的急跌股，其中有色 4 只，同受一次宏观冲击；short_breakout 空仓）。
  - 遗留：公告下载完成后补跑 `quality` / 正式版 `event-study`，并重跑保留集回测核对风险公告规则（评估时公告约到 2024 年上半年）。
- 2026-10-17：公告下载完成并补齐失败单元（各事件数据集 0 失败）。正式版 `quality` / `event-study` 完成，结论与初步版一致
  （行业内便宜 5/5 年为正；分红 / 业绩预增 / 股东户数大降 4/5；追涨类 0~1/5；反转类 1~2/5）。烂股剔除组后续平均超额高于保留组，
  但大跌率更高，定位为尾部风险控制（RULES 第 6 节）。补齐公告后重跑保留集：long +13.2% / +8.5%，short_reversal +9.7% / +3.4%，
  short_breakout -5.1% / -10.4%，状态不变。用户同意合并到 main。
- 2026-10-17：数据体检：数据本身完整正确（qlib 与 baostock 价格 / 成交额 / 停牌逐格对得上，各缓存单元齐全）；
  实时数据有三处滞后或失败且不提示（两融 10-16 只有沪市、baostock 只到 10-15、东财快照断连），中证1000 缺 2025-04-16；
  存放分散、快照每天累积、公告 / baostock 小文件读取慢（8 秒 1.87GB / 34 秒）。选股方法评审后用户决定：先做阶段 8 数据层重整，
  再做阶段 9 规则优化（全部只在研发区验证，验收规则见第九节）。
- 2026-10-17：阶段 8 完成。缓存迁到 data/raw（12025 个文件，行数 / mtime 一致），panel 派生表使 ST 与风险公告读取从
  30 秒 / 8 秒（1.87GB）降到 0.4 秒 / 0.1 秒；去掉东财快照 ST（exclude_st），ST 只用 baostock；基准缺日改为 ffill；
  简报与 `update-data --status` 显示数据截止日。重抓两融时发现 10-16 沪市当时也不完整（1675 → 2000 行），新规则拒存。
  结果核对：10-16 三池持仓 / 计划不变，long 研发区年化 15.68% 不变，前缀一致性 0 差异。
- 2026-10-17：阶段 9.0 / 9.4 实现。与计划的差异：实盘净值不解析每日 CSV，而是**按各段存档的完整配置重新推演并记账**
  （与每日选股同一实现，段间衔接现金与持仓），再逐日核对已入库 CSV 的持仓；这样改规则不会改写旧段，CSV 不一致只可能来自数据修订。
  第一段按 2026-10-15 建立（配置哈希因阶段 8 去掉 exclude_st 而变化，选股结果核对相同）。short_breakout `enabled: false`。
  9.1 / 9.2 的代码先全部完成（默认关闭，旧配置结果不变），研究按"一次只改一项"进行；网格报告新增"最大单票权重"列。
