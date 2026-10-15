# 重构计划：三大指数股池（沪深300 / 中证500 / 中证1000）+ 模块化架构

**阅读顺序**：第 0 节（定了什么）→ 第 1 节（规则规格，实现时的唯一依据）→
第 2 节（结构与接口）→ 第 3 节（分阶段执行，每阶段可独立验收、独立提交）。

**约定**：
- `PY` = `C:/Users/merak/anaconda3/envs/qlib/python.exe -X utf8`（qlib 环境，见 `quant_env.txt`）；
- 每个阶段在分支 `refactor/index-pools` 上单独提交，阶段验收不通过不进入下一阶段；
- 标 **⛔ 决策点** 的地方需与用户讨论后再继续。

---

## 0. 目标与已定决策

| 项 | 决策 |
|----|------|
| 股池 | hs300（csi300/SH000300）、zz500（csi500/SH000905）、zz1000（csi1000/SH000852）；三池独立资金、独立回测。中证2000 放弃（本地无成分与指数行情） |
| 选股 | 当日指数成分打分 → 准入/持有规则过滤 → 池容 = max(5, ceil(1% × 准入通过只数)) → 按换仓规则在上期股池基础上更新；日频信号 |
| 数据 | 本地 qlib cn_data（来源 chenditc/investment_data release）；每次选股/回测前比对 GitHub 最新版本，不一致先更新；不追求对齐最新交易日 |
| 补充数据 | akshare 基本面缓存（`src/data/fundamental.py`）沿用，仅补 Alpha158 没有的基本面特征 |
| 工具链 | qlib 自带：表达式引擎、DataLoader/DatasetH、RollingGen、Recorder、回测引擎、模型库；不用 akquant |
| 打分 | qlib `LGBModel` 主力 + qlib `LinearModel` 基线，walk-forward 滚动训练；手写加权打分（`scoring.combine` + `factor_weights`）退役，IC 检验保留作特征体检 |
| 模块化 | 加股池 = 加 yaml；加过滤规则 = 一个带装饰器的函数 + yaml 声明；改某池 = 只改该池 yaml；换模型 = 改 yaml 的 scorer |
| 同口径 | 选股与回测共用同一套打分、过滤、换仓实现；选股输出 = 回测同日持仓 |
| 样本划分 | ≤2022-12-31 为研发区（所有调参、选模型只看这段）；2023+ 为保留集，参数冻结后只评估一次 |

**现状中需要被修正的问题（本次重构一并解决）**：
1. 选股与回测口径不一致：选股有 ST/停牌/涨跌停过滤而回测没有；选股每期取前 topk，回测却是 TopkDropout（每期只换 n_drop 只）。
2. `list_instruments(universe, start, end)` 返回区间内**曾经**的成分并集，未按每日成分资格屏蔽 → 前视偏差。
3. 估值缓存只覆盖 csi800（1260 只，`update_data.py` 硬编码），基本面缓存最早 2017，不足以支撑 csi1000 和 2013 起的训练窗。
4. `engine.run_backtest` 硬编码 stable/short 分支，成本按池名取值，`ACCOUNT=1e8` 硬编码。

---

## 1. 规则规格

### 1.1 选股流程（每个交易日收盘后，按数据最新交易日 T 执行）

**第 1 步 打分**：scorer 对 T 日指数成分（`in_universe` 为真）全部打分。
训练也用全指数成分（样本更多）；板块等限制只影响可选集合，不影响模型。

**第 2 步 过滤**：规则分两类，作用不同：

| 类别 | 含义 | 不通过时 |
|------|------|---------|
| **exit（持有规则）** | 还能不能继续持有 | 分数置 NaN → 已持有的强制调出 |
| **entry（准入规则）** | 能不能新买入 | 不能新进池；**已持有的不因此被卖出** |

原因：qlib `TopkDropoutStrategy` 中分数为 NaN 的持仓会排到末尾被优先卖出
（`signal_strategy.py`：`pred_score.reindex(current_stock_list).sort_values()`）。
若把准入规则套在分数上，持仓股某天涨停或成交额偶然偏低就会被误卖。

**第 3 步 池容**：`topk_T = max(min_topk, ceil(top_pct × entry 规则在 T 日通过的只数))`；
yaml 写了 `topk:` 则以其为准。hs300/zz500 主板口径下恒为 5，zz1000 约 8。

**第 4 步 换仓**（纯函数 `rebalance()`，选股与回测共用）：
```
输入：holdings（上期股池及各自已持有天数）、score_T、entry_T、topk_T、n_drop、hold_thresh
1. forced_out = 持仓中 score_T 为 NaN 的（exit 规则不通过）——不计入 n_drop
2. kept = holdings - forced_out，按 score_T 降序
3. candidates = entry_T 为真、score_T 非 NaN、不在 kept 中的股票，按 score_T 降序，
   取前 (n_drop + topk_T - len(kept)) 只
4. comb = kept ∪ candidates 按 score_T 降序；
   dropped = kept 中落在 comb 末尾 n_drop 名内、且已持有天数 ≥ hold_thresh 的
5. buy = candidates 前 (len(dropped) + topk_T - len(kept)) 只
6. new_holdings = kept - dropped + buy
无上期股池（首次）：new_holdings = candidates 前 topk_T 只。
```
与 qlib `TopkDropoutStrategy.generate_trade_decision` 逻辑一致，区别只在第 1、3 步。
回测中因停牌/跌停卖不出、涨停买不进的单子由 qlib exchange 处理；选股输出对应标注
"计划调出/调入，若不可成交则顺延"。

**第 5 步 输出**：新股池（等权）+ 新进/调出（注明原因：exit 规则名 / 排名末位）/留存 +
每只的 T 日排名 + "纯排名前 topk" 参考列表。简报标注
"基于 T 日收盘数据，供其后首个交易日参考，数据滞后 N 天"及 data_version。

### 1.2 过滤规则与各池默认值

| 规则 | scope | hs300 | zz500 | zz1000 | 说明 |
|------|-------|-------|-------|--------|------|
| in_universe | exit | ✓ | ✓ | ✓ | 强制，不可关闭；用成分文件 (start,end) 区间生成 |
| exclude_st | exit | ✓ | ✓ | ✓ | **只作用于最新一天**（只有当前快照）；历史回测不套用，防前视偏差 |
| boards | entry | [main] | [main] | [main] | main/cyb/kcb/bse |
| min_list_days | entry | 180 | 180 | 180 | |
| exclude_suspend | entry | 20 | 20 | 20 | 近 N 日有停牌（volume 为 NaN 或 0） |
| exclude_limit | entry | ✓ | ✓ | ✓ | 当日涨停买不进 |
| min_amount | entry | — | 5e7 | 5e7 | **近 20 日日均**成交额（元）；阈值在研发区调定 |
| chase_limit / falling_knife | entry | — | — | — | 原短线池专用，默认不启用 |

可选扩展（预留，不在本次实现）：行业集中度上限属组合构建约束，需在策略层实现。

### 1.3 回测规格

- 策略 `entry_aware_topk_dropout`：继承 `TopkDropoutStrategy`，`generate_trade_decision`
  内调用 `rebalance()`；topk 逐信号日读取。原 `ShortPoolStrategy` 保留注册。
- 信号对齐沿用现约定：T 日分数 → T+1 收盘成交（`shift=1`、`deal_price=close`）。
  `scores_on_rebalance_dates` 增加 `daily`（恒等）分支；weekly/monthly 池的 exit 检查只在信号日生效。
- 成本：佣金 + 印花税 + 滑点，global 默认、池内可覆盖。
- 账户规模 `account` 移入 global.yaml（按实际资金设置）；`exchange_kwargs` 加 `volume_threshold`；
  boards 含 cyb/kcb 时按板块给涨跌停表达式（qlib cn 默认 `limit_threshold=0.095`）。
- 报告：全区间 + 分段 + 研发区/保留集，列出超额年化、IR、超额回撤、日均换手；
  头部写 data_version、池配置 hash、scorer 及其 recorder id。

### 1.4 模型规格（lgbm / linear 共用）

| 项 | 规格 |
|----|------|
| 特征 | `NestedDataLoader(Alpha158DL, StaticDataLoader(补充基本面因子))`；补充因子不写入 cn_data |
| 标签 | 5 日：`Ref($close,-6)/Ref($close,-1)-1`；20 日对照：`Ref($close,-21)/Ref($close,-1)-1`；learn 处理器 `DropnaLabel` + `CSRankNorm` |
| 特征处理 | infer：`CSZScoreNorm` + `Fillna`（截面处理器，无需 fit，可跨滚动窗口复用 handler） |
| 滚动 | 每年末重训（可配 quarterly）；train 4 年（可配 3~5）→ valid 6 个月（LGBM 早停）→ test 下一区间 |
| purge | train→valid、valid→test 之间各留 `label_horizon + 1` 个交易日 |
| 工具 | `qlib.workflow.task.gen.RollingGen` / `qlib.contrib.rolling`（handler 缓存）+ `R` recorder（`data/mlruns/`，不入库）+ `SigAnaRecord`（IC/RankIC） |
| 复现 | 固定随机种子；recorder 记录 data_version 与池配置 hash |
| 推理 | 选股时加载该池最近一期模型推理，不重训；最近一期早于设定周期则提示先训练 |
| 输出 | 日频预测宽表（index=交易日, columns=代码），可选 EMA 平滑（`score_ema` 参数） |

### 1.5 数据同步规格

- `github_latest_version()`：GitHub API `releases/latest`（实施时核实 asset 命名）；支持 `proxy`。
- `local_version()`：读 `cn_data/.version`。
- `sync_if_needed(strict=False)`：版本不同 → 下载到临时目录 → 解压到 `cn_data.new` →
  校验（calendars/instruments/features 存在；新日历末日 ≥ 旧日历末日；csi300/500/1000 成分文件存在）→
  `cn_data` 改名 `cn_data.bak`（只留一代）→ `cn_data.new` 改名 `cn_data` → 写 `.version`。
  下载/校验/改名任一失败：告警并继续用旧数据；`strict=True` 时中止。
- `--adopt <tag>`：把现有数据标记为某版本（本地目前无 `.version`，否则首跑即全量下载）。
- 调用时机：脚本入口最先执行，必须早于 `init_qlib()` 和 `listing_dates()` 等缓存。
- 任何派生数据不得写入 cn_data 目录。

---

## 2. 目标结构与接口

### 2.1 目录

```
config/
  global.yaml              # cost / account / data(同步) / model 公共参数
  pools/{hs300,zz500,zz1000}.yaml
src/
  registry.py              # 通用 Registry：装饰器注册 + inspect.signature 参数校验
  config.py                # list_pools() / load_pool(name) -> PoolSpec（global 深合并 + 校验）
  data/
    market.py fundamental.py   # 沿用；fundamental 补每股净资产列
    sync.py                # 新：数据同步
  factors/
    technical.py fundamental.py registry.py   # 沿用（FACTORS = 补充因子来源）
    derived_valuation.py   # 新：由 yjbb EPS/BPS + 原始价推导 PE_TTM/PB
    alpha158.py            # 新：构造 NestedDataLoader / DatasetH 配置
  selection/
    universe.py            # 沿用纯函数 + 新增 membership_panel()
    filters.py             # 新：@register_filter(scope=...)，返回 date×code 布尔面板
    signal.py              # 新：build_signal(pool, start, end) -> Signal
    rebalance.py           # 新：calc_topk() / rebalance()
    pool.py                # 改：股池读写（按池分目录）、简报
  models/
    scorers.py             # 新：@register_scorer：momentum/random（冒烟）、lgbm、linear
    rolling.py             # 新：滚动训练、预测拼接、模型加载
  backtest/
    strategies.py          # 改：@register_strategy；新增 EntryAwareTopkDropout
    engine.py              # 改：池无关
  eval.py                  # 不变
scripts/
  update_data.py           # 改：universe 取所有池并集；start-year 默认 2012
  run_selection.py run_backtest.py run_factor_check.py run_sensitivity.py   # 改：--pool 动态
  run_train.py             # 新：滚动训练
  run_model_compare.py     # 新：lgbm vs linear A/B
  check_consistency.py     # 新：选股逐日回放 vs 回测持仓
tests/                     # 新：pytest 单元测试
data/pool/{pool}/{tag}.csv / {tag}_brief.md   # 按池分目录；旧 CSV 移入 data/pool/legacy/
```

### 2.2 池 yaml 示例（`config/pools/zz1000.yaml`）

```yaml
name: zz1000
universe: csi1000
benchmark: SH000852
top_pct: 0.01
min_topk: 5
# topk: 8                 # 可选：显式写死，优先于 1% 规则
rebalance: daily          # daily / weekly / monthly
strategy:
  name: entry_aware_topk_dropout
  n_drop: 1
  hold_thresh: 3
scorer:
  name: lgbm              # lgbm / linear / momentum / random
  label_horizon: 5
  score_ema: null
filters:
  in_universe: {}
  exclude_st: {}
  boards: {boards: [main]}
  min_list_days: {days: 180}
  exclude_suspend: {days: 20}
  exclude_limit: {}
  min_amount: {amount: 5.0e7, window: 20}
backtest: {start: '2018-01-01', end: null, in_sample_end: '2022-12-31'}
cost: {slippage: 0.002}   # 覆盖 global.cost 中的同名项
```

### 2.3 核心接口

```python
# src/config.py
list_pools() -> list[str]
load_pool(name: str) -> PoolSpec            # 深合并 global，校验注册名与参数名，失败即抛错

# src/selection/filters.py
@register_filter("min_amount", scope="entry")
def min_amount(ctx: dict, amount: float, window: int = 20) -> pd.DataFrame  # bool 面板

# src/selection/signal.py
@dataclass
class Signal:
    score: pd.DataFrame   # 已套 exit 规则
    entry: pd.DataFrame   # entry 规则交集
    topk: pd.Series       # 逐日池容
build_signal(pool: PoolSpec, start: str, end: str) -> Signal

# src/selection/rebalance.py
calc_topk(n_entry: int, top_pct: float, min_topk: int, fixed: int | None) -> int
rebalance(holdings: dict[str, int], score: pd.Series, entry: pd.Series,
          topk: int, n_drop: int, hold_thresh: int) -> RebalanceResult   # 新持仓/调入/调出及原因

# src/models/scorers.py
@register_scorer("lgbm")
class LGBScorer:
    def predict(self, pool: PoolSpec, start: str, end: str) -> pd.DataFrame

# src/backtest/engine.py
run_backtest(signal: Signal, pool: PoolSpec, start: str, end: str) -> dict
```

---

## 3. 分阶段执行

### P0 准备与基线留存
- 建分支 `refactor/index-pools`。
- 用现有代码跑 `PY scripts/run_backtest.py --score momentum --tag baseline_momentum`，
  保存 `data/reports/backtest_report_baseline_momentum.md` 作为回归对照。
- **验收**：基线报告存在；`git status` 干净。

### P1 配置层与注册表
- 新增 `src/registry.py`、`config/global.yaml`、`config/pools/*.yaml`（按 2.2）。
- 重写 `src/config.py`（`list_pools`/`load_pool`/`PoolSpec`/深合并/校验）；
  旧 `load_config()` 暂留给未迁移脚本，P8 删除。
- 测试 `tests/test_config.py`：深合并、未知规则名报错、未知参数名报错、`list_pools` 自动发现。
- **验收**：`PY -m pytest tests/test_config.py` 通过；复制一份 yaml 改名后 `list_pools()` 能发现。

### P2 数据同步
- 新增 `src/data/sync.py`（按 1.5），命令行 `PY -m src.data.sync [--adopt TAG] [--check]`。
- 在 `run_selection.py` / `run_backtest.py` 入口最先调用；加 `--skip-data-check`、`--strict-data`。
- `update_data.py`：新鲜度校验降为提示。
- 测试 `tests/test_sync.py`：mock GitHub 响应，验证版本相同跳过、不同触发、下载失败降级、
  校验失败不替换（在临时目录模拟 cn_data）。
- **验收**：先 `--adopt` 当前版本，再运行显示"版本一致"；单元测试通过；
  手工确认失败路径下 `cn_data` 未被改动。

### P3 过滤与信号层
- `universe.membership_panel()`；`filters.py` 实现 1.2 全部规则（面板化，声明 scope）。
- `scorers.py` 先实现 `momentum` / `random` 两个合成 scorer（冒烟用）。
- `signal.py` 的 `build_signal()`；`rebalance.py` 的 `calc_topk()`。
- 测试 `tests/test_filters.py`：构造小面板验证每条规则；`in_universe` 对纳入前日期为 False；
  `exclude_st` 只影响最后一天。
- **验收**：单元测试通过；三池 `build_signal(momentum)` 跑通，打印每池逐日 topk 分布
  （hs300/zz500 应恒为 5，zz1000 约 8）。

### P4 换仓、回测、选股端到端
- `rebalance()`（按 1.1 第 4 步）+ `tests/test_rebalance.py`：首次建仓、n_drop 上限、
  hold_thresh、exit 强制调出不计 n_drop、entry 不通过的持仓不被卖出、topk 变化。
- `EntryAwareTopkDropout` 策略；`engine.run_backtest` 池无关化（成本/账户/volume_threshold 取自 PoolSpec）。
- 重写 `run_backtest.py`（`--pool` 动态、`--scorer` 覆盖、`--tag`）与 `run_selection.py`
  （读上期股池 → `rebalance()` → 输出按池分目录，CSV 含 `entry_date`）。
- 新增 `check_consistency.py`：从同一初始股池出发逐日回放 20 个交易日，比对选股与回测持仓。
- **验收**：
  - 三池 `--scorer momentum` 回测与选股跑通；
  - `check_consistency.py` 三池全部一致（停牌/跌停导致的差异逐条列出原因）；
  - 与 P0 基线对比，差异只来自有意修改的口径（成分时点、过滤、换仓规则），在报告中逐项说明。

### P5 ML 可行性试验（纯 Alpha158）
- `alpha158.py`（先只用 Alpha158DL）、`models/rolling.py`、`lgbm`/`linear` scorer、`run_train.py`。
- 三池滚动训练（按 1.4），回测 2018 起，输出 RankIC（`SigAnaRecord`）与回测报告。
- 内存：按池串行，handler 缓存；峰值超过 ~12G 时 `QLIB_KERNELS=1`。
- **验收**：三池两种 scorer 均产出连续预测与回测报告；recorder 中可查到每个滚动窗口。
- **⛔ 决策点**：研发区 RankIC / 超额不理想时，与用户讨论是否调整方向，再进入 P6。

### P6 基本面特征
- `fundamental.py` 补"每股净资产"列；`update_data.py` 的 `--start-year` 默认 2012、
  估值范围取所有池 universe 并集；重拉 yjbb/xjll/fhps。
- `derived_valuation.py`：PE_TTM（累计 EPS 换算 TTM）、PB = 原始价（close/factor）/ BPS；
  抽样与百度估值缓存核对。
- 补充因子经 `StaticDataLoader` 并入 `NestedDataLoader`。
- `run_factor_check.py` 改为按池：三池 × 5日/20日 IC 体检 → `data/reports/factor_report_{pool}.md`。
- **验收**：基本面特征在各池覆盖率报告（按年统计非 NaN 比例）；若无法回溯到 2013，
  在报告中写明可用起始年，并相应推迟首个测试年；IC 体检报告生成。

### P7 A/B 对比与调参（只用研发区）
- `run_model_compare.py`：同池同区间 lgbm vs linear（可选第三个 `weighted` 对照，复用旧加权打分，仅对照不进生产）
  → `data/reports/model_compare_{pool}.md`。
- `run_sensitivity.py` 网格：n_drop × hold_thresh × score_ema × rebalance（daily/weekly），
  另调 `min_amount`。
- 参数冻结后**一次性**跑 2023+ 保留集，单独列示。
- **⛔ 决策点**：逐池与用户确定 scorer 与参数，写入池 yaml。

### P8 收尾
- 删除 `src/selection/scoring.py`、`factor_weights`、旧 `load_config()`、`config/selector.yaml`、
  脚本中 stable/short 分支；旧股池 CSV 移入 `data/pool/legacy/`。
- 更新 README / RULES（新规则与基线）/ ARCHITECTURE（扩展指南、数据同步、
  防未来函数约定新增：成分时点、标签对齐、purge gap、ST 只作用最新一天）。
- **验收**：`PY -m pytest tests` 全部通过；三池 `run_selection` / `run_backtest` 用生产 scorer 跑通。

---

## 4. 风险与应对

| 风险 | 应对 |
|------|------|
| 池容仅 5/5/~8 只，集中度高 | 报告重点看超额回撤；行业上限作为预留扩展 |
| 日频 ML 分数噪声大、换手吞噬 alpha | hold_thresh、score_ema、daily vs weekly 纳入网格；报告列换手与成本占比 |
| LightGBM 小池子过拟合 | linear 基线对照，样本外跑不过则退回 linear |
| 基本面数据无法回溯到 2013 | 推迟首个测试年，或该时段只用 Alpha158 |
| 内存（Alpha158 × ~1800 只 × 10 年） | handler 缓存、按池串行、float32、必要时 QLIB_KERNELS=1 |
| GitHub 访问不稳定 / 数据替换中断 | 代理配置；原子替换 + `cn_data.bak`；失败降级继续 |
| 数据每天变化导致结果不可复现 | 报告、模型、简报写 data_version 与配置 hash |
| 回测资金规模失真 | account 按实际设置 + volume_threshold |

## 5. 总验收标准

1. `run_selection.py --pool hs300|zz500|zz1000` 生成股池 CSV + 简报：池容符合 1.1 第 3 步；
   全部股票满足 exit 规则，新进股票满足 entry 规则；单日调入 ≤ n_drop（首次建仓除外）。
2. 数据同步：自动版本检查；落后时下载并原子替换；一致时零改动；`--skip-data-check`、`--adopt` 可用；
   失败时旧数据完好。
3. 模块化：新增测试池（复制 yaml 改 universe）不改代码即可选股与回测；
   新增过滤规则只需"装饰器函数 + yaml 声明"。
4. 同口径：`check_consistency.py` 三池通过。
5. 三池 × 5日/20日 IC 体检报告生成。
6. lgbm 与 linear 各跑通三池全量回测，A/B 报告生成；选择依据仅来自研发区，保留集单独列示；
   逐池 scorer 经讨论落定。
7. 所有报告与简报含 data_version 与配置 hash；`tests/` 全部通过。
8. README / RULES / ARCHITECTURE 与新结构一致。

---

## 6. 执行进度（截至 2026-10-15，分支 `refactor/index-pools`）

### 阶段状态

| 阶段 | 状态 | 要点 |
|------|------|------|
| P0~P5 | ✅ | 配置/注册表、数据同步、过滤与信号、换仓与回测同口径、Alpha158 + LightGBM 滚动训练 |
| P6 | ✅ | 补充基本面特征已实现（`model.features` 可选），**不采用**：RankIC 持平但 TopK 头部收益三池均下降 |
| P7 | ✅（模型待修复，见下） | 模型对比、换手网格、参数冻结后保留集一次性评估；⛔ 用户决定：**仅 hs300 为 production**，zz500/zz1000 为 observe |
| P8 | ✅ | 退役手写打分与旧配置；池 `status` 字段；旧股池归档 `data/pool/legacy/`；README/RULES/ARCHITECTURE 重写 |
| 模型修复 | 🔶 进行中 | 见"当前问题" |

### 已定决策（详见 RULES.md）

- 打分 lgbm（Alpha158，学习率 0.05）；换仓 n_drop=1、hold_thresh=10、日频、不做分数 EMA；三池同一套参数。
- 保留集保护：所有研究脚本默认截止 2022-12-31，`--show-holdout` 才看 2023+。
- 保留集一次性评估（2026-10-15，配置 hash hs300 5625b6bd43 / zz500 15ce5b536a / zz1000 1825cd175f）：
  hs300 +7.0%/IR 0.43，zz500 -1.5%/IR -0.09，zz1000 +2.6%/IR 0.14；RankIC 保持研发区 2/3 以上，但中证500/1000 Top5 头部失效。
- 5 只组合回测噪声很大，方案比较以 RankIC 与 TopK 未来超额为主（`run_model_compare.py` 已内置）。

### 当前问题：LightGBM 早停退化（发现于首次生成正式股池时）

- qlib LGBModel 总是早停；学习率 0.05 时部分窗口第 1 棵树后验证损失即不再下降 → 模型只有 1 棵树，
  分数仅几个取值、头部大量并列、按代码顺序截取（hs300 2026 窗口去重比例 7%，并列 ~12 只；hs300/zz500 2022 窗口也退化）。
  **因此保留集评估中 hs300 2026 部分来自退化模型，上表保留集结果并不干净。**已生成的 2026-10-14 股池已删除未提交。
- ⛔ 用户决定：改为固定树数（不早停），树数只在研发区选定，重训后对保留集做"第二次查看"（报告中注明不再是干净样本外）。
- 已完成：`FixedRoundLGBModel`（`model.lgbm.early_stopping_rounds: 0` 启用，含测试）；`scripts/run_lgbm_rounds.py`
  （研发区各窗口训练一次 400 棵，按前 k 棵评估）。
- 研发区树数结果（RankIC / Top5 / Top20 未来 5 日超额 %/期；对照：早停版 hs300 0.051/0.39/0.30，zz500 0.060/0.25/0.16）：

  | 树数 | hs300 | zz500 | zz1000 |
  |------|-------|-------|--------|
  | 25 | .0525 / .367 / .284 | .0601 / .097 / .155 | 未跑完 |
  | 50 | .0568 / .427 / .326 | .0625 / .171 / .177 | |
  | 100 | .0599 / .438 / .373 | .0653 / .229 / .239 | |
  | 200 | .0610 / .425 / .365 | .0653 / .330 / .252 | |
  | 400 | .0599 / .384 / .339 | .0624 / .301 / .248 | |

  初步：100~200 棵最好，400 开始回落；均优于早停版。

### 下次从这里继续

1. 跑 zz1000 树数研究（内存大，单独进程）：
   `QLIB_KERNELS=2 python -X utf8 scripts/run_lgbm_rounds.py --pool zz1000`
2. 三池统一选定树数（倾向 200，视 zz1000 结果），写入 `config/global.yaml`：
   `model.lgbm: {learning_rate: 0.05, early_stopping_rounds: 0, num_boost_round: N}`
3. 逐池重训：`QLIB_KERNELS=2 python -X utf8 scripts/run_train.py --pool <池> --kinds lgbm`
4. 研发区复核：`run_model_compare.py --pool all`、`check_consistency.py --pool all`；确认去重比例≈1、无头部并列。
5. 保留集第二次查看：`run_model_compare.py --pool all --variant "lgbm=scorer.name=lgbm" --show-holdout --tag final2`，
   报告与 RULES.md 中注明"第二次查看，非干净样本外"；与用户确认各池 production/observe 是否调整。
6. 更新 RULES.md 基线表；`run_selection.py --pool all` 生成首批正式股池并提交 `data/pool/`。
