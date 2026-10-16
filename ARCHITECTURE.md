# 代码架构（ARCHITECTURE）

## 怎么改（常见需求 → 改哪里）

| 需求 | 操作 | 改代码？ |
|------|------|---------|
| 增加股池 | 复制 `config/pools/_template.yaml` 为 `config/pools/<名>.yaml`，把 `name` 改成同名，填 universe / benchmark；`train --pool <名>` | 否 |
| 暂停 / 恢复股池 | 该池 yaml `enabled: false / true`（`--pool all` 跳过暂停的池） | 否 |
| 改所有池的池容 / 换仓（含调仓频率）/ 打分器 / 过滤 | 改 `config/pool_defaults.yaml` | 否 |
| 只改某个池 | 在该池 yaml 写同名项覆盖（如 `rebalance: {hold_thresh: 5}`、`sizing: {topk: 10}`） | 否 |
| 某池关闭一条默认过滤规则 | 该池 yaml `filters: {规则名: false}`（`in_universe` 不可关闭） | 否 |
| 改资金 / 成本 / 回测窗口 / 模型公共参数 | 改 `config/global.yaml` | 否 |
| 新增过滤规则 | `selection/filters.py` 写 `@FILTERS.register("名", scope="entry"/"exit")` 函数，yaml 的 `filters:` 声明 | 仅新增 |
| 新增打分模型 | `models/scorers.py` 写 `@SCORERS.register("名")` 类（`predict(pool, ctx)`），yaml `scorer.name` | 仅新增 |
| 新增回测策略 | `backtest/strategies.py` 写 `@STRATEGIES.register("名")` 工厂 `(signal, rebalance, **参数)` | 仅新增 |
| 新增补充特征 | `features/fundamental.py` 的 `build_fundamental` 加一项并加入 `FEATURES` | 仅新增 |
| 新增命令 | `cli/<命令>.py`（`HELP` / `add_arguments` / `run`），加入 `cli/__init__.py` 的 `COMMANDS` | 仅新增 |
| 临时实验 | `train / backtest / grid / check` 加 `--set 点路径=值`，`backtest / compare / grid` 加 `--tag 名`；模型按训练签名分目录，不覆盖正式版本 | 否 |

配置合并顺序（后者覆盖前者，dict 递归合并，加载即校验未知键 / 未注册名字 / 参数名）：

```
config/global.yaml → config/pool_defaults.yaml → config/pools/<name>.yaml → --scorer 整段替换 → --set 点路径覆盖
```

## 目录

```
config/
  global.yaml            系统级：资金、成本、撮合、回测窗口、模型公共参数、数据同步
  pool_defaults.yaml     所有池共用的选股规则：sizing / rebalance / signal / scorer / strategy / filters
  pools/<name>.yaml      单个池：name / enabled / status / universe / benchmark + 覆盖项（_ 开头为模板，不加载）
stockanalysis/
  __main__.py            python -X utf8 -m stockanalysis <命令>
  paths.py               所有目录位置
  registry.py            通用注册表：装饰器注册 + 签名校验（过滤规则 / 打分器 / 策略共用）
  runtime.py             内存监控
  config/    schema.py   PoolSpec（Sizing / RebalanceRule 分段）
             loader.py   池发现、按层合并、校验、配置 hash
  data/      market.py   qlib 初始化、交易日历、OHLCV、宽表、行情上下文
             sync.py     GitHub release 版本比对 → 断点续传下载 → 校验 → 原子替换 cn_data
             fundamental.py  akshare 基本面缓存（业绩报表 / 分红 / 现金流）
             reference.py    股票名称、当前 ST 名单
             universe.py     指数成分时点面板、板块判定、上市日（qlib instruments 文件）
  features/  labels.py   标签与前向收益口径（唯一定义）
             processors.py   原地截面标准化处理器
             alpha158.py     特征集 → qlib DataHandlerLP
             fundamental.py  补充基本面特征（可选，model.features）
  models/    scorers.py  SCORERS：lgbm / linear（滚动模型）、momentum / random（冒烟）
             estimators.py   模型构造与参数（FixedRoundLGBModel）
             windows.py      滚动窗口（含 purge 空档）
             store.py        训练签名、产物目录、读写
             training.py     滚动训练
             inference.py    推理（样本外预测 + 最近模型外推）
  selection/ filters.py  FILTERS：(ctx, **参数) -> 交易日×代码 布尔面板，scope=entry|exit
             signal.py   build_signal → Signal(score / entry / topk)
             sizing.py   池容规则
             rebalance.py    换仓规则（选股与回测共用）
             schedule.py     调仓日程：哪些交易日是信号日（选股与回测共用）
             selector.py     选股（纯逻辑）
             store.py        股池 CSV 与简报读写
             daily.py        每日选股流程
  backtest/  engine.py   池无关回测
             strategies.py   STRATEGIES：entry_aware_topk_dropout（调用 rebalance()）
             metrics.py      区间指标（复利口径）与区间切分
             runner.py       回测区间（保留集保护）与单池回测流程
  research/  metrics.py  RankIC / TopK 超额 / IC 汇总 / 特征相关（唯一定义）
             reports.py  Markdown 报告
             compare.py grid.py rounds.py consistency.py factor_check.py
  cli/       common.py   公共参数（--pool / --set / --show-holdout / --tag / 数据同步）
             select train backtest compare grid rounds factors check sync update_data
tests/       与包结构对应的单元测试
data/        pool/（入库）· models/ mlruns/ reports/ fundamental/（不入库）
```

依赖方向：`cli → research → backtest → selection → models → features → data → paths / registry / runtime`，下层不依赖上层。
`config` 被各层使用；唯一的反向引用是 `config/loader.py` 校验时延迟导入三个注册表（FILTERS / SCORERS / STRATEGIES），以便在加载配置时就报出未注册的名字。

## 数据流

```
GitHub release ──sync──> cn_data（qlib 行情）
                              │
        Alpha158（表达式引擎）──> DataHandlerLP（截面标准化）──> 滚动训练 LightGBM
                                                                      │  data/models/{pool}/lgbm_{签名}/
                                                                      ▼
行情上下文 ──> FILTERS 面板（exit / entry）                  样本外预测 / 最近模型外推
                │                                                     │
                └──────────> build_signal（分数.where(exit)、entry、topk）<───┘
                                         │
                ┌────────────────────────┴───────────────────────┐
     select：最后一行 + 上期股池 → rebalance()            backtest：整张面板 → EntryAwareTopkDropout
             → data/pool/{pool}/{T}.csv                             （内部同样调用 rebalance()）
```

## 防未来函数约定

- **指数成分时点**：只有当日在指数内的股票参与排名（`in_universe` 强制规则）。
- **信号对齐**：T 日收盘打分 → T+1 收盘成交（qlib 策略 shift=1、deal_price=close）。
- **标签**：`Ref($close,-6)/Ref($close,-1)-1`；训练/验证/测试之间各留 label_horizon+1 个交易日空档。
- **财报**：报告期数据从披露截止日（年报/一季报 4/30、半年报 8/31、三季报 10/31）起可见；分红从预案公告日起可见。
- **ST**：只有当前快照，仅作用于最新一天，历史回测不套用。
- **保留集**：回测、网格、训练报告、模型对比默认截止 `in_sample_end`；`--show-holdout` 仅用于参数冻结后的最终评估。

## 价格与单位约定

- qlib `$close` 为复权价，真实成交价 = `$close / $factor`；每股财务值需 × 报告期末 `$factor` 才能与复权价相除。
- 本数据 `$volume` 单位为"手"（且已复权），成交量约束按 `$amount × 1000 / $close` 折算。
- `$amount` 单位为千元。

## 内存约定

- 本机提交内存上限约 32G（页面文件仅 600MB）。中证1000 特征约 276 万行 × 158 列。
- qlib 原版处理器整表复制会 OOM，已改用原地处理器 `CSRobustZScoreInplace`（需 `drop_raw=True`）。
- 多池训练逐池单独进程运行（`train --pool zz1000`）；`QLIB_KERNELS=2`（`__main__.py` 默认）。
- 报告在 `data/reports/`、模型在 `data/models/`、`data/mlruns/`（均不入库）；股池 CSV 在 `data/pool/`（入库）。

## 已发现并修复的问题（2026-10 重构期间，避免重蹈覆辙）

| 问题 | 后果 | 修复 |
|------|------|------|
| 成分股取区间并集，未按当日成分屏蔽 | 用"未来才纳入指数"的股票选股（前视偏差） | `in_universe` 强制规则（成分时点面板） |
| 选股与回测两套过滤、两套换仓逻辑 | 选股结果与回测持仓不一致 | 统一 `build_signal` + `rebalance()`，`check` 命令逐日验证 |
| 准入规则套在分数上 | 持仓股涨停 / 成交额偶低即被卖出 | 规则分 entry / exit，准入只限制新买入 |
| `$volume` 单位为"手" | 成交量约束严 100 倍，卖单部分成交 | 按 `$amount × 1000 / $close` 折算 |
| qlib 默认涨跌停时买卖全禁 | 涨停不能卖、跌停不能买（与实盘不符） | `forbid_all_trade_at_limit=False`，只按方向限制 |
| qlib 风险指标默认简单累加 | 回撤可超过 -100% | 改用复利口径（`mode="product"`） |
| 旧股息率因子用复权价做分母 | 截面比例随复权因子扭曲 | 每股值 × 报告期末 `$factor` ÷ 复权收盘价 |
| 中证1000 特征处理整表复制 | 超出提交内存上限（OOM） | 原地截面标准化处理器 + `drop_raw` + 逐池单独进程 |
| LightGBM 早停在第 1 棵树 | 分数只有几个取值、头部并列、按代码顺序截取 | 固定 200 棵树、不早停（`FixedRoundLGBModel`） |
| 训练签名包含整个 model 段 | 改 lgbm 参数导致 linear 模型找不到 | 签名只含公共参数 + 本模型参数 |
| GitHub 下载中途断流 | 数据包不完整、sha256 校验失败 | 断点续传 + 重试，未完成部分保留到下次 |


## 规则池（买卖点）架构（2026-10 重构，见 REFACTOR_PLAN.md）

`long` / `short_reversal` / `short_breakout` 三个池：A 股全市场 → 烂股剔除 → 板块权限 → 买点入池 / 卖点出池，各 ≤ 10 只。

```
akshare / baostock / 长桥 CLI ──update-data──> data/fundamental/ data/events/ data/baostock/
                                                      │
features/  risk.py（烂股判定）  technical.py（指标，akquant.talib）  events.py（事件面板）  regime.py（择时）
                 │                         │                                  │
selection/filters.py（烂股 = exit 规则）   features/conditions.py（CONDITIONS：MCP 平台规则翻译 + 阈值邻域）
                 │                         │
                 └──> selection/points.py：build_point_signal()（买点 / 卖点 / 可成交 / 优先分 / 择时面板）
                                           │
                       selection/positions.py：simulate()（逐日：执行昨日计划 → 更新持仓 → 判断卖点 → 补位）
                         │                                           │
     select：points_daily.py 从 points.live_start 推演到 T          backtest：points_engine.py 记账（含成本）
             → data/pool/{pool}/{T}.csv + 简报（卖出线、计划、体检）     research：event-study / points-grid
```

| 需求 | 操作 |
|------|------|
| 新增选股条件 | `features/conditions.py` 写 `@register(名, 来源, kind, grid)` 函数；先 `event-study --only 名` 检验 |
| 改某池买卖点 | 池 yaml 的 `points` 段（buy / buy_any / avoid / sell + 止损类参数），加载时校验条件名与参数 |
| 改板块权限 | `config/global.yaml` 的 `account_boards`（所有池的 boards 规则默认取它） |
| 改烂股规则 | `config/pool_defaults.yaml` 的 filters（exit 规则，已持有的也会被卖出） |
| 规则参数实验 | `points-grid --pool X --vary 键=值1,值2 --starts 6`（多起点合成，只看研发区） |

防未来约定（规则池补充）：
- 财报可见日 = 实际披露日（akshare 预约披露）次日，无记录时为披露截止日；较早报告期晚于较新报告期披露时不倒退。
- 事件可见日：两融 T+1、龙虎榜 T 日盘后、股东户数 / 增减持 / 回购 / 业绩预告 / 公告为公告日次日；解禁日程事先公开。
- 历史 ST 用 baostock `isST`；业绩报表里的股票简称是**当前**名称，不能用于历史 ST。
- 行业为当前分类（东财行业），历史回测有轻微前视。
- 成交：T 日收盘信号 → T+1 收盘（或 `deal_price: vwap` 均价）成交；止损类按收盘价判断，不是盘中触价。

重构中发现的数据问题：

| 问题 | 后果 | 处理 |
|------|------|------|
| 本地中证全指 SH000985 缺 667 个交易日 | 基准收益与择时指标失真 | 长线基准改中证800、短线改中证1000；记账发现基准缺数据会告警 |
| 业绩报表 / 现金流报告期缓存永不更新 | 披露未完时抓到的半张表一直沿用 | `cache_valid()`：截止日 + 7 天后抓取才视为完整，否则每日重抓 |
| baostock 原生接收循环在断开时空转 | 下载"卡住"不报错 | 替换 send_msg：超时 + 断开即报错 + 重新登录重试 |
| 立案调查公告归在"其他"类 | 按公告类型筛不到 | 抓全部公告，按标题关键词识别 |
