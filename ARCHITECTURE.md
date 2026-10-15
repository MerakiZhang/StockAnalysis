# 代码架构（ARCHITECTURE）

## 模块图

```
config/global.yaml + config/pools/*.yaml      # 规则唯一来源；池 yaml 深合并覆盖 global
        │
src/config.py → PoolSpec                       # 发现池、深合并、注册名/参数名校验、配置 hash
src/registry.py                                # 通用注册表：装饰器注册 + 签名校验
        │
src/data/
  sync.py        GitHub release 版本比对 → 下载 → 校验 → 原子替换 cn_data（失败保留旧数据）
  market.py      qlib 初始化与行情读取
  fundamental.py akshare 基本面缓存（业绩报表 / 分红 / 现金流）
        │
src/factors/
  base.py           行情宽表 load_wide、行情上下文 market_ctx
  alpha158.py       Alpha158 + 标签 → DataHandlerLP；原地截面标准化处理器
  supplementary.py  补充基本面特征（可选，model.features 开关）
        │
src/models/
  scorers.py   SCORERS：lgbm / linear（滚动模型）、momentum / random（冒烟）
  rolling.py   滚动窗口（含 purge 空档）、训练、样本外预测拼接、按训练签名存取、外推推理
        │
src/selection/
  universe.py   板块判定、成分时点面板、上市日、当前 ST
  filters.py    FILTERS：每条规则 (ctx, **params) -> date×code 布尔面板，scope=entry|exit
  signal.py     build_signal → Signal(score 已套 exit / entry / 逐日 topk)
  rebalance.py  calc_topk、rebalance()（选股与回测共用的换仓规则）
  pool.py       select()、股池 CSV 与简报
        │
src/backtest/
  strategies.py STRATEGIES：entry_aware_topk_dropout（调用 rebalance）/ topk_dropout / short_pool
  engine.py     池无关回测、分段指标（复利口径）
src/eval.py     IC 工具 + 内存监控
```

## 数据流

```
GitHub release ──sync──> cn_data（qlib 行情）
                              │
        Alpha158DL（表达式引擎）──> DataHandlerLP（截面标准化）──> 滚动训练 LGBModel
                                                                      │  data/models/{pool}/{kind}_{签名}/
                                                                      ▼
行情上下文 market_ctx ──> FILTERS 面板（exit / entry）      样本外预测 / 最近模型外推
                │                                                     │
                └──────────────> build_signal（分数.where(exit)、entry、topk）<─┘
                                         │
                ┌────────────────────────┴───────────────────────┐
     run_selection：最后一行 + 上期股池 → rebalance()     run_backtest：整张面板 → EntryAwareTopkDropout
                → data/pool/{pool}/{T}.csv                            （内部同样调用 rebalance()）
```

## 扩展指南

| 修改 | 操作 | 改代码？ |
|------|------|---------|
| 增加股池 | 复制 `config/pools/hs300.yaml` 改 name/universe/benchmark，`run_train.py --pool 新池` | 否 |
| 改某池规则（池容、过滤、板块、换仓、成本、状态） | 只改该池 yaml，研发区重跑验证 | 否 |
| 增加过滤规则 | `filters.py` 写 `@FILTERS.register("名", scope="entry"/"exit")` 函数，池 yaml 的 `filters:` 声明 | 仅新增 |
| 换打分模型 | 池 yaml `scorer.name`（lgbm / linear）；模型参数放 `model.lgbm` 等，改后重训 | 否 |
| 增加打分模型 | `models/scorers.py` 写 `@SCORERS.register` 类，实现 `predict(pool, ctx)` | 仅新增 |
| 增加回测策略 | `strategies.py` 写类 + `@STRATEGIES.register` 工厂函数 | 仅新增 |
| 增加补充特征 | `supplementary.py` 的 `build_supplementary` 增加一项并加入 `FEATURES` | 仅新增 |
| 临时实验 | 任意脚本 `--set 点路径=值 --tag 名`，模型按训练签名分目录，不覆盖正式版本 | 否 |

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
- 多池训练逐池单独进程运行；`QLIB_KERNELS=2`。
- 报告在 `data/reports/`、模型在 `data/models/`、`data/mlruns/`（均不入库）；股池 CSV 在 `data/pool/`（入库）。
