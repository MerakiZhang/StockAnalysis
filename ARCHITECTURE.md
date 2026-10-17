# 代码架构（ARCHITECTURE）

## 怎么改（常见需求 → 改哪里）

| 需求 | 操作 | 改代码？ |
|------|------|---------|
| 增加股池 | 复制一个现有池为 `config/pools/<名>.yaml`，`name` 改成同名，改 benchmark / scorer / points | 否 |
| 暂停 / 恢复股池 | 该池 yaml `enabled: false / true`（`--pool all` 跳过暂停的池） | 否 |
| 改某池买卖点 | 池 yaml 的 `points` 段（buy / buy_any / avoid / sell + stop_loss / take_profit / trailing_stop / max_hold / rank_exit 等），加载时校验条件名与参数 | 否 |
| 改优先分 | 池 yaml 的 `scorer.factors`（因子名 → 权重，负权重 = 越小越好） | 否 |
| 改板块权限 | `config/global.yaml` 的 `account_boards`（所有池的 boards 规则默认取它） | 否 |
| 改烂股 / 准入规则 | `config/pool_defaults.yaml` 的 filters（exit = 已持有的也卖出；entry = 只限制新买入） | 否 |
| 某池关闭一条默认过滤规则 | 该池 yaml `filters: {规则名: false}`（`in_universe` 不可关闭） | 否 |
| 改资金 / 成本 / 回测窗口 | 改 `config/global.yaml` | 否 |
| 新增选股条件 | `features/conditions.py` 写 `@register(名, 来源, kind, grid, desc)` 函数；先 `event-study --only 名` 检验 | 仅新增 |
| 新增过滤规则 | `selection/filters.py` 写 `@FILTERS.register("名", scope="entry"/"exit")` 函数，yaml 的 `filters:` 声明 | 仅新增 |
| 新增打分器 | `models/scorers.py` 写 `@SCORERS.register("名")` 类（`predict(pool, ctx)`），yaml `scorer.name` | 仅新增 |
| 新增命令 | `cli/<命令>.py`（`HELP` / `add_arguments` / `run`），加入 `cli/__init__.py` 的 `COMMANDS` | 仅新增 |
| 规则参数实验 | `points-grid --pool X --vary 键=值1,值2 --starts 6`（多起点合成，只看研发区）；单次用 `backtest --set 点路径=值 --tag 名` | 否 |

配置合并顺序（后者覆盖前者，dict 递归合并；`--set` 的叶子值为 dict 时整体替换；加载即校验未知键 / 未注册名字 / 参数名）：

```
config/global.yaml → config/pool_defaults.yaml → config/pools/<name>.yaml → --scorer 整段替换 → --set 点路径覆盖
```

## 目录

```
config/
  global.yaml            系统级：资金、账户板块权限、成本、撮合、回测窗口、数据同步
  pool_defaults.yaml     所有池共用：烂股剔除 / 准入 filters、占位优先分
  pools/<name>.yaml      单个池：name / enabled / status / universe / benchmark / scorer / points（_ 开头为模板，不加载）
stockanalysis/
  __main__.py            python -X utf8 -m stockanalysis <命令>
  paths.py               所有目录位置
  registry.py            通用注册表：装饰器注册 + 签名校验（过滤规则 / 打分器 / 策略 / 条件共用）
  runtime.py             内存监控
  config/    schema.py   PoolSpec、PointsSpec（买卖点参数）
             loader.py   池发现、按层合并、--set、校验、配置 hash
  data/      market.py   qlib 初始化、交易日历、OHLCV / vwap / 复权因子、行情上下文
             sync.py     GitHub release 版本比对 → 断点续传下载 → 校验 → 原子替换 cn_data
             universe.py     全市场在市区间、板块判定、上市日
             fundamental.py  akshare 业绩报表 / 分红 / 现金流 / 实际披露日 / 业绩预告（按报告期缓存）
             events.py       akshare 两融 / 龙虎榜 / 股东户数 / 解禁 / 增减持 / 回购 / 全部公告
             baostock_data.py  逐日 ST / 停牌 / 估值 / 换手（多进程下载）+ 行业快照
             longbridge_cli.py 长桥 CLI：市场温度历史、实盘体检（资金流 / 评级 / 新闻 / 公告）
             reference.py    股票名称
             cache.py        原始缓存约定：快照只留最新一份
             panel.py        data/panel 派生表（源文件或关键词变化自动重建）
             freshness.py    各数据源截止日 vs 信号日（简报"数据截止"、update-data --status）
  features/  labels.py   前向收益口径（唯一定义）
             fundamental.py  基本面逐日面板（按披露日可见）、行业映射
             risk.py     烂股判定（历史 ST、市值、连续亏损、风险公告、解禁等）
             technical.py    技术指标（akquant.talib 逐股计算；量能用成交额）
             events.py   事件 → 逐日面板（公告日次日可见）
             regime.py   择时：宽度、涨跌停家数、指数均线、市场温度
             conditions.py   CONDITIONS：选股条件库（MCP 平台规则翻译 + 本项目条件）与 CONTINUOUS 连续因子
  models/    scorers.py  SCORERS：multifactor（截面排名加权优先分）、momentum / random（冒烟）
  selection/ filters.py  FILTERS：(ctx, **参数) -> 交易日×代码 布尔面板，scope=entry|exit
             points.py   build_point_signal → PointSignal（买点 / 卖点 / 可成交 / 优先分 / 择时面板）
             positions.py    simulate()：买卖点状态机（选股与回测共用）
             points_daily.py 买卖点池每日选股（持仓、卖出线、次日计划、接近买点、长桥体检）
             signal.py   行情上下文 + 过滤面板（exit / entry 合成）
             schedule.py     信号日日程（daily / weekly / monthly）
             store.py        股池 CSV 与简报读写
             daily.py        每日选股流程（按池类型分流）
  backtest/  points_engine.py  买卖点池自记账回测（成本、涨跌停、停牌顺延、逐日自检）
             metrics.py      区间指标（复利口径）与区间切分
             runner.py       回测区间（保留集保护）与单池回测流程（按池类型分流）
  research/  event_study.py  条件事件研究（5 / 20 / 60 日超额、胜率、逐年）
             points_grid.py  买卖点参数网格 + 多起点合成 + 滚动验证
             quality_check.py  烂股剔除效果体检
             consistency.py  防未来检查（前缀一致性）
             metrics.py reports.py
  cli/       common.py   公共参数（--pool / --set / --show-holdout / --tag / 数据同步）
             select backtest quality event_study points_grid check sync update_data
tests/       与包结构对应的单元测试
data/        pool/（入库）· reports/ raw/{akshare,baostock,longbridge}/ panel/（不入库）
```

依赖方向：`cli → research → backtest → selection → models → features → data → paths / registry / runtime`，下层不依赖上层。
`config` 被各层使用；唯一的反向引用是 `config/loader.py` 校验时延迟导入注册表（FILTERS / SCORERS / CONDITIONS），
以便在加载配置时就报出未注册的名字和参数。

## 数据流（买卖点池）

```
GitHub release ──sync──> cn_data（qlib 行情）
akshare / baostock / 长桥 CLI ──update-data──> data/raw/ ──> data/panel/（is_st、风险公告）
                                                      │
features/  risk.py（烂股判定）  technical.py（指标）  events.py（事件面板）  fundamental.py  regime.py（择时）
                 │                         │
selection/filters.py（exit / entry）   features/conditions.py（CONDITIONS）   models/scorers.py（优先分）
                 │                         │                                      │
                 └──> selection/points.py：build_point_signal()  <───────────────┘
                                           │
                       selection/positions.py：simulate()（逐日：执行昨日计划 → 更新持仓 → 判断卖点 → 排队补位）
                         │                                           │
     select：points_daily.py 从 points.live_start 推演到 T          backtest：points_engine.py 记账（含成本）
             → data/pool/{pool}/{T}.csv + 简报（卖出线、计划、体检）     research：event-study / points-grid
```

买卖点池的回测不走 qlib 策略：同一个纯函数模拟器同时用于每日选股和回测，选股结果按构造等于回测在 T 日的持仓；
记账的报告列与 qlib report 口径相同，沿用 `segment_metrics`。旧版 topk 换仓路径（池容 / 换仓规则 / qlib 回测策略）
已于 2026-10-17 删除，需要时见 git 历史。

## 防未来函数约定

- **在市时点**：只有当日在市的股票参与（`in_universe` 强制规则）；上市满一年才可买入。
- **信号对齐**：T 日收盘信号 → T+1 收盘（或 `deal_price: vwap` 均价）成交；止损类按收盘价判断，不是盘中触价。
- **财报**：可见日 = 实际披露日（akshare 预约披露）次日，无记录时为披露截止日（年报 / 一季报 4/30、半年报 8/31、三季报 10/31）；
  较早报告期晚于较新报告期披露时不倒退；分红从预案公告日起可见。
- **事件**：两融 T+1、龙虎榜 T 日盘后、股东户数 / 增减持 / 回购 / 业绩预告 / 公告为公告日次日；解禁日程事先公开。
- **ST**：历史与当日都用 baostock `isST`（当日未到时沿用前一日，简报"数据截止"标出）；业绩报表里的股票简称是**当前**名称，不能用于历史 ST。
- **行业**：当前分类（东财行业），历史回测有轻微前视。
- **保留集**：回测、网格默认截止 `in_sample_end`（2022-12-31）；`--show-holdout` 仅用于参数冻结后的最终评估。

## 价格与单位约定

- qlib `$close` 为复权价，真实成交价 = `$close / $factor`；每股财务值需 × 报告期末 `$factor` 才能与复权价相除。
  简报一律显示不复权价，卖出线按 T 日复权因子折算。
- 本数据 `$volume` 单位为"手"（且已复权）；`$amount` 单位为千元。量能条件一律用成交额。

## 内存约定

- 本机提交内存上限约 32G（页面文件仅 600MB）。全市场一个 PointSignal 约数百 MB（交易日 × 5000+ 只 × 二十余个面板）。
- `points-grid` 按信号键分组依次跑，内存里只保留当前一组 PointSignal，不计算简报用面板；每跑完一个方案写
  `data/reports/logs/points_grid_<池>.partial.csv`。网格不要与 baostock 多进程下载或另一个网格同时运行。
- Windows 下脚本若触发 qlib / joblib 多进程，入口必须有 `if __name__ == "__main__":`，否则子进程反复导入主模块。

## 已发现并修复的问题（2026-10，避免重蹈覆辙）

| 问题 | 后果 | 修复 |
|------|------|------|
| 成分股取区间并集，未按当日成分屏蔽 | 用"未来才纳入"的股票选股（前视偏差） | `in_universe` 强制规则（时点面板） |
| 选股与回测两套过滤、两套换仓逻辑 | 选股结果与回测持仓不一致 | 选股与回测共用同一实现 `simulate()`，`check` 做前缀一致性（防未来）检查 |
| 准入规则套在分数上 | 持仓股涨停 / 成交额偶低即被卖出 | 规则分 entry / exit，准入只限制新买入 |
| `$volume` 单位为"手" | 旧 qlib 回测的成交量约束严 100 倍 | 量能一律改用成交额 `$amount` |
| qlib 默认涨跌停时买卖全禁 | 涨停不能卖、跌停不能买 | 只按方向限制（买卖点模拟器：涨停不买、跌停不卖） |
| qlib 风险指标默认简单累加 | 回撤可超过 -100% | 复利口径（`mode="product"`） |
| 股息率用复权价做分母 | 截面比例随复权因子扭曲 | 每股值 × 报告期末 `$factor` ÷ 复权收盘价 |
| GitHub 下载中途断流 | 数据包不完整 | 断点续传 + 重试 |
| 本地中证全指 SH000985 缺 667 个交易日 | 基准收益与择时指标失真 | 长线基准中证800、短线中证1000；记账发现基准缺数据告警 |
| 业绩报表 / 现金流报告期缓存永不更新 | 披露未完时抓到的半张表一直沿用 | `cache_valid()`：截止日 + 7 天后抓取才视为完整 |
| baostock 原生接收循环在断开时空转 | 下载"卡住"不报错 | 替换 send_msg：超时 + 断开即报错 + 重新登录重试 |
| 立案调查公告归在"其他"类 | 按公告类型筛不到 | 抓全部公告，按标题关键词识别 |
| akshare 公告接口节假日报 KeyError | 节假日被记为失败单元、每次重抓 | 当作当日无公告 |
| 网格同时持有多组 PointSignal | 内存不足崩溃 | 按信号键分组、只留当前一组 |
| 网格信号键未含 `deal_price` | 均价成交方案复用收盘价成交的信号，对照无效 | 信号键加入 `deal_price`（有测试） |
| LightGBM 早停在第 1 棵树（旧模型池，已删除） | 分数并列、按代码顺序截取 | 当时改固定树数；2026-10-17 随旧池一起删除 |
