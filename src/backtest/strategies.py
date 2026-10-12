"""回测策略实现。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.backtest import Order
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.strategy import TopkDropoutStrategy


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
