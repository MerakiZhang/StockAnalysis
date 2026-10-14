"""回测策略实现与注册表 STRATEGIES。

注册项是工厂函数 `(signal: Signal, **params) -> qlib 策略实例`，params 来自池 yaml 的
strategy 段（name 以外的键）。新增策略 = 写策略类 + 一个 @STRATEGIES.register 工厂函数。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from qlib.backtest import Order
from qlib.backtest.decision import OrderDir, TradeDecisionWO
from qlib.contrib.strategy import TopkDropoutStrategy

from ..registry import Registry
from ..selection.rebalance import rebalance

STRATEGIES = Registry("strategy", skip=1)


def to_qlib_signal(scores: pd.DataFrame) -> pd.Series:
    """宽表（index=交易日, columns=代码）-> MultiIndex(datetime, instrument) Series。

    全 NaN 的行（非信号日）被丢弃，策略在这些日期取到 None，持仓不动。
    """
    s = scores.stack()  # pandas>=2 默认 dropna
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


class EntryAwareTopkDropout(TopkDropoutStrategy):
    """TopkDropout + 准入规则 + 逐日池容，买卖清单由 selection.rebalance.rebalance() 生成，
    与选股脚本同一实现。

    signal 已套持有规则（NaN → 持仓调出）；entry 为准入面板，topk_series 为逐日池容，
    三者按"信号日"（成交日的前一交易日，shift=1）取值。
    """

    def __init__(self, *, entry: pd.DataFrame, topk_series: pd.Series, **kwargs):
        kwargs.setdefault("topk", int(topk_series.max()) if len(topk_series) else 1)
        super().__init__(**kwargs)
        self.entry = entry
        self.topk_series = topk_series

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        sig_date = pd.Timestamp(pred_start_time).normalize()
        if pred_score is None or sig_date not in self.entry.index:
            return TradeDecisionWO([], self)

        current = copy.deepcopy(self.trade_position)
        bar = self.trade_calendar.get_freq()
        holdings = {c: current.get_stock_count(c, bar=bar) for c in current.get_stock_list()}
        res = rebalance(holdings, pred_score, self.entry.loc[sig_date],
                        int(self.topk_series.loc[sig_date]), self.n_drop, self.hold_thresh)

        sell_orders, buy_orders = [], []
        cash = current.get_cash()
        for code in res.sell_exit + res.sell_rank:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue  # 停牌/跌停卖不出，留待下一信号日
            order = Order(stock_id=code, amount=current.get_stock_amount(code=code),
                          start_time=trade_start_time, end_time=trade_end_time, direction=Order.SELL)
            if self.trade_exchange.check_order(order):
                sell_orders.append(order)
                trade_val, trade_cost, _ = self.trade_exchange.deal_order(order, position=current)
                cash += trade_val - trade_cost

        value = cash * self.risk_degree / len(res.buy) if res.buy else 0
        for code in res.buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue  # 停牌/涨停买不进，该名额空置到下一信号日
            price = self.trade_exchange.get_deal_price(stock_id=code, start_time=trade_start_time,
                                                       end_time=trade_end_time, direction=OrderDir.BUY)
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time,
                                                    end_time=trade_end_time)
            amount = self.trade_exchange.round_amount_by_trade_unit(value / price, factor)
            buy_orders.append(Order(stock_id=code, amount=amount, start_time=trade_start_time,
                                    end_time=trade_end_time, direction=Order.BUY))
        return TradeDecisionWO(sell_orders + buy_orders, self)


class ShortPoolStrategy(TopkDropoutStrategy):
    """短线池策略：TopkDropout 周度调仓 + 每日止损 / 强制轮换。

    - 每日先检查持仓：现价 / 入场价 - 1 <= stop_loss，
      或持有交易日数 >= max_hold_days，则生成 SELL 单（不可交易则跳过）。
    - 之后叠加父类调仓逻辑（仅信号日非空），同一 code 的卖单去重。
    - 入场价/入场日在 post_exe_step 钩子中按实际成交记录
      （execute_result 元素为 (order, trade_val, trade_cost, trade_price)，
      trade_price 即真实成交价）。
    """

    def __init__(self, *, stop_loss: float | None = -0.05, max_hold_days: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.stop_loss = stop_loss  # None = 禁用止损
        self.max_hold_days = max_hold_days
        self._entries: dict[str, tuple[float, int]] = {}  # code -> (入场价, 入场 step)
        self.stop_loss_count = 0
        self.max_hold_count = 0

    # ------------------------------------------------------------- hooks
    def post_exe_step(self, execute_result) -> None:
        super().post_exe_step(execute_result)
        if not execute_result:
            return

        # collect_data 末尾已 step()，当前 step 比成交日大 1
        entry_step = self.trade_calendar.get_trade_step() - 1
        for order, _trade_val, _trade_cost, trade_price in execute_result:
            if order.deal_amount <= 0:
                continue
            if order.direction == Order.BUY:
                self._entries[order.stock_id] = (float(trade_price), entry_step)
            else:  # SELL 成交后若已清仓，移除入场记录
                if self.trade_position.get_stock_amount(order.stock_id) <= 0:
                    self._entries.pop(order.stock_id, None)

    # --------------------------------------------------------- decision
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pos = self.trade_position

        # 1) 每日止损 / 到期检查
        force_sell: dict[str, str] = {}
        for code in list(self._entries):
            amount = pos.get_stock_amount(code)
            if amount <= 0:
                self._entries.pop(code, None)
                continue
            entry_price, entry_step = self._entries[code]
            price = self.trade_exchange.get_close(code, trade_start_time, trade_end_time)
            if price is None or not np.isfinite(price):
                price = pos.get_stock_price(code)  # 上一交易日收盘价兜底
            reason = None
            if self.stop_loss is not None and entry_price > 0 and price / entry_price - 1 <= self.stop_loss:
                reason = "stop_loss"
            elif trade_step - entry_step >= self.max_hold_days:
                reason = "max_hold"
            if reason is not None:
                force_sell[code] = reason

        orders = []
        for code, reason in force_sell.items():
            order = Order(
                stock_id=code,
                amount=pos.get_stock_amount(code),
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if not self.trade_exchange.check_order(order):
                continue  # 停牌/跌停卖不出，保留入场记录明日再试
            orders.append(order)
            if reason == "stop_loss":
                self.stop_loss_count += 1
            else:
                self.max_hold_count += 1

        # 2) 叠加周度调仓（非信号日父类返回空决策）
        decision = super().generate_trade_decision(execute_result)
        for order in decision.order_list:
            if order.direction == Order.SELL and order.stock_id in force_sell:
                continue  # 止损单已覆盖，去重
            orders.append(order)
        return TradeDecisionWO(orders, self)


# ------------------------------------------------------------------ 注册表
@STRATEGIES.register("entry_aware_topk_dropout")
def entry_aware_topk_dropout(signal, n_drop: int = 1, hold_thresh: int = 1):
    return EntryAwareTopkDropout(
        signal=to_qlib_signal(signal.score), entry=signal.entry, topk_series=signal.topk,
        n_drop=int(n_drop), hold_thresh=int(hold_thresh), method_sell="bottom", method_buy="top",
        # 只按方向限制：涨停不能买、跌停不能卖（qlib 默认涨跌停时买卖都禁止，过于保守）
        forbid_all_trade_at_limit=False,
    )


@STRATEGIES.register("topk_dropout")
def topk_dropout(signal, n_drop: int = 1, hold_thresh: int = 1, topk: int | None = None):
    """qlib 原版 TopkDropout（固定池容，不区分准入/持有规则），仅作对照。"""
    return TopkDropoutStrategy(
        signal=to_qlib_signal(signal.score.where(signal.entry)),
        topk=int(topk or signal.topk.median()), n_drop=int(n_drop), hold_thresh=int(hold_thresh),
        method_sell="bottom", method_buy="top", only_tradable=True,
    )


@STRATEGIES.register("short_pool")
def short_pool(signal, n_drop: int = 3, hold_thresh: int = 1, stop_loss: float | None = -0.05,
               max_hold_days: int = 10, topk: int | None = None):
    """原短线池策略：TopkDropout + 每日止损 / 到期强平（固定池容）。"""
    return ShortPoolStrategy(
        signal=to_qlib_signal(signal.score.where(signal.entry)),
        topk=int(topk or signal.topk.median()), n_drop=int(n_drop), hold_thresh=int(hold_thresh),
        stop_loss=None if stop_loss is None else float(stop_loss), max_hold_days=int(max_hold_days),
        method_sell="bottom", method_buy="top", only_tradable=True,
    )
