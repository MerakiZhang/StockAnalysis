"""回测策略实现与注册表 STRATEGIES。

注册项是工厂函数 `(signal: Signal, rebalance: RebalanceRule, **params) -> qlib 策略实例`：
rebalance 为池的换仓规则（选股与回测共用），params 为池 yaml strategy 段中 name 以外的键。
新增策略 = 写策略类 + 一个 @STRATEGIES.register 工厂函数，池 yaml 的 strategy.name 指向它。
"""
from __future__ import annotations

import copy

import pandas as pd

from qlib.backtest import Order
from qlib.backtest.decision import OrderDir, TradeDecisionWO
from qlib.contrib.strategy import TopkDropoutStrategy

from ..registry import Registry
from ..selection.rebalance import rebalance

STRATEGIES = Registry("strategy", skip=2)  # signal、rebalance 不由 yaml 提供


def to_qlib_signal(scores: pd.DataFrame) -> pd.Series:
    """宽表（index=交易日, columns=代码）-> MultiIndex(datetime, instrument) Series。

    全 NaN 的行（非信号日）被丢弃，策略在这些日期取到 None，持仓不动。
    """
    s = scores.stack()  # pandas>=2 默认 dropna
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


class EntryAwareTopkDropout(TopkDropoutStrategy):
    """TopkDropout + 准入规则 + 逐日池容，买卖清单由 selection.rebalance.rebalance() 生成，
    与选股（selection/selector.py）同一实现。

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


# ------------------------------------------------------------------ 注册表
@STRATEGIES.register("entry_aware_topk_dropout")
def entry_aware_topk_dropout(signal, rebalance):
    return EntryAwareTopkDropout(
        signal=to_qlib_signal(signal.score), entry=signal.entry, topk_series=signal.topk,
        n_drop=rebalance.n_drop, hold_thresh=rebalance.hold_thresh, method_sell="bottom", method_buy="top",
        # 只按方向限制：涨停不能买、跌停不能卖（qlib 默认涨跌停时买卖都禁止，过于保守）
        forbid_all_trade_at_limit=False,
    )


@STRATEGIES.register("signal_points")
def signal_points(signal, rebalance):
    """买卖点池：不经 qlib 策略，回测走 backtest/points_engine.py（与选股共用 selection/positions.py 模拟器）。"""
    raise RuntimeError("signal_points 池请用 backtest.points_engine.run_points_backtest()")
