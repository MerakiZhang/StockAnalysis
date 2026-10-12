# 代码架构（ARCHITECTURE）

## 模块图

```
config/selector.yaml          # 唯一规则来源（池参数/因子权重/成本/过滤）
        │
src/config.py                 # 加载 + 校验（factor_weights 键必须在注册表中）
        │
src/data/                     # 数据层
  market.py                   #   qlib 初始化（幂等）与行情读取（OHLCV）
  fundamental.py              #   akshare 基本面缓存：yjbb/fhps/xjll/估值/spot
        │
src/factors/                  # 因子层（全部返回"越大越好"的宽表 date×code）
  base.py                     #   load_wide：行情透视成宽表（float32）
  technical.py                #   动量/波动/短线情绪因子（原始方向）
  fundamental.py              #   质量/成长/估值/股息 + 时点面板构建
  registry.py                 #   FACTORS = {name: builder(codes,start,end,ctx)}
        │                     #   market_ctx()：行情只加载一次，供因子与过滤共用
src/selection/                # 选股层
  universe.py                 #   板块/ST/次新/停牌/涨跌停/流动性过滤
  scoring.py                  #   MAD 去极值 → zscore → 加权合成（combine）
  pool.py                     #   stable_scores/short_scores、股池 CSV 与简报
        │
src/backtest/                 # 回测层（qlib 0.9.7）
  engine.py                   #   run_backtest / scores_on_rebalance_dates / wf_segments
  strategies.py               #   ShortPoolStrategy（每日止损 + 到期强平）
        │
src/eval.py                   # 因子 IC 检验（IC/分层/相关性/滚动IC）+ RSS 内存监控

scripts/                      # 入口脚本（全部 python -X utf8 运行）
  update_data.py              #   基本面缓存更新 + 数据新鲜度校验
  run_selection.py            #   生成当期股池（--pool stable|short）
  run_backtest.py             #   回测（--score/--pools/--only-factor/--weights/--boards/--tag）
  run_factor_check.py         #   单因子 IC 检验（--pool stable|short）
  run_sensitivity.py          #   topk × n_drop 敏感性网格
```

## 数据流

```
qlib 行情（本地 cn_data）──┐
                          ├─> factors.registry.market_ctx（宽表 ctx，仅加载一次）
akshare 基本面缓存 ────────┘            │
（data/fundamental/*.parquet）          ▼
                FACTORS 注册表按 config factor_weights 构建非零权重因子
                                        ▼
                selection.scoring.combine（MAD→zscore→加权）= 日频总分
                                        ▼
        ┌─ run_selection：当日截面 + universe.build_filter_mask → 股池 CSV/简报
        └─ run_backtest：engine.scores_on_rebalance_dates 折叠信号
                        → TopkDropout / ShortPoolStrategy → 分段指标报告
```

## 扩展指南

**加因子**：
1. 在 `src/factors/technical.py` 或 `src/factors/fundamental.py` 写函数，
   返回宽表（index=交易日, columns=代码），方向处理成"越大越好"；
2. 在 `src/factors/registry.py` 的 `FACTORS` 注册一行（builder 签名
   `(codes, start, end, ctx)`，行情从 `ctx["close"]` 等取，不要自行加载）；
3. 在 `config/selector.yaml` 的 `factor_weights` 配权重（0 = 不计算）；
4. 先跑 `run_factor_check.py` 过 IC 检验，再跑 `run_backtest.py` 验证。

**改规则**：只改 `config/selector.yaml`（权重/topk/过滤/成本），重跑
`run_backtest.py --score real`，对照 RULES.md 基线。

**加新池**：在 yaml 加一段池配置，`registry` 因子可直接复用；
回测引擎目前只认 stable/short 两池（`engine.run_backtest` 按池选策略），
新池需在 engine 中加分支。

## 防未来函数约定

- **财报数据**：报告期数据从 `available_date`（披露截止日：年报/一季报 4/30、
  半年报 8/31、三季报 10/31）起才向前填充；东财"最新公告日期"列不可用
  （是公司最新公告日，非该报告披露日）。
- **信号对齐**：qlib TopkDropoutStrategy 用 shift=1 取前一交易日分数下单；
  `scores_on_rebalance_dates()` 把信号放在调仓日的前一交易日。
- **短线入场价**：在 `post_exe_step` 钩子里按真实成交价记录，不用信号价。
- **IC 检验**的 `forward_returns`（未来收益）仅限研究用途，禁止进入选股信号。
- **已知近似**：历史 ST 状态用当前快照代替；历史行业归属用最新报告期。

## 内存约定

- 本机 32G：qlib 初始化默认 `QLIB_KERNELS=2`（并行加载），紧张时设为 1 串行。
- 行情一律经 `load_wide` 转 float32 宽表；ctx 机制保证同一窗口只加载一次。
- 重任务脚本在关键节点调 `src.eval.log_rss` 打印 RSS；超过 ~12G 应降
  kernels 或缩小加载范围。
- 报告输出到 `data/reports/`（不入库）；股池 CSV 在 `data/pool/`（入库）。
