"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-11 10:03:20
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-11 11:29:09
Description: 策略
"""
import backtrader as bt
import pandas as pd
import logging
import numpy as np

class Multi_Weight_Strategy(bt.Strategy):
    params = (
        ("verbose", True),  # 是否打印交易日志
        ("show_log", True),
    )

    def __init__(self):
        self.weights = {}
        self.target_weights = {}
        self.sell_orders = {}  # 记录卖出订单信息，使用订单ID作为键
        self.reserved_cash = 0  # 已预留但尚未使用的资金
        self.expected_cash_inflow = 0  # 预计的卖出资金流入
        self.pending_orders = 0  # 记录待处理订单数量

        # 设置订单在收盘时执行（确保卖出资金立即释放）
        self.broker.set_coc(False)  # 关闭收盘执行
        self.broker.set_coo(True)  # 启用开盘执行

    def next(self):
        if self.params.verbose:
            print(f"\n策略权重生成日期: {self.datetime.date()}")

        # 重置预留资金和预计流入
        self.reserved_cash = 0
        self.expected_cash_inflow = 0
        self.pending_orders = 0  # 重置待处理订单计数

        # 清理已完成的卖出订单
        self._cleanup_completed_orders()

        # 确定所有资产的target_weights
        self._calculate_target_weights()

        total_value = self.broker.get_value()
        cash_value = self.broker.get_cash()

        # 验证总市值是否有效
        if not self._is_valid_value(total_value):
            print("警告: 系统计算的总市值无效，将使用手动计算值")
            total_value = self._calculate_total_value()

        print(
            f"日期: {self.datetime.date(0)} | "
            f"账户资金: ¥{cash_value:,.2f} | "
            f"总市值: ¥{total_value:,.2f}"
        )

        # 检查是否为最后一个交易日或数据不足
        if self._is_last_day():
            print(f"⚠️ 警告：{self.datetime.date(0)} 是最后一个交易日或数据不足，跳过交易逻辑")
            return

        # 执行卖出和买入逻辑
        self._execute_sell_orders(total_value)
        self._execute_buy_orders(total_value, cash_value)

    def _cleanup_completed_orders(self):
        """清理已完成的订单信息"""
        for order_id in list(self.sell_orders.keys()):
            order = self.sell_orders[order_id]['order']
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                del self.sell_orders[order_id]

    def _calculate_target_weights(self):
        """计算所有资产的目标权重"""
        self.target_weights.clear()  # 清空旧权重
        for data in self.datas:
            # 允许在第一个交易日计算权重，只要有当前数据
            if len(data) < 1 or data.open[0] <= 0:
                continue
            if len(data.w) > 0 and not pd.isnull(data.w[0]):
                self.target_weights[data] = data.w[0]

    def _is_last_day(self):
        """检查是否为最后一个交易日或数据不足"""
        for data in self.datas:
            # 检查是否有足够的数据用于交易决策
            if len(data) < 1:  # 至少需要当前交易日的数据
                return True
            # 检查是否为最后一个交易日
            if len(data) >= data.buflen():
                return True
        return False

    def _execute_sell_orders(self, total_value):
        """执行卖出订单逻辑"""
        """执行卖出订单逻辑"""
        for data, target_weight in self.target_weights.items():
            # 确保有当前交易日的数据
            if len(data) < 1 or data.open[0] <= 0:
                continue

            position = self.getposition(data)
            # 如果没有持仓，跳过卖出逻辑
            if position.size == 0:
                continue

            # 使用统一的价格计算当前市值和目标市值
            price = data.open[0]  # 使用当前开盘价(注意索引改为0)
            position_value = position.size * price
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight
            delta_value = target_value - position_value

            if delta_value < -1e-5:  # 需要卖出
                # 计算需要卖出的数量
                raw_size = abs(delta_value) / price

                # 调试输出
                if self.params.verbose:
                    print(
                        f"[调试] {data._name}: 持仓量={position.size}, 当前权重={current_weight:.2%}, 目标权重={target_weight:.2%}")
                    print(
                        f"[调试] {data._name}: 持仓市值={position_value:.2f}, 目标市值={target_value:.2f}, 差值={delta_value:.2f}")
                    print(f"[调试] {data._name}: 理论卖出数量={raw_size:.2f}")

                # 处理规则：
                # 1. 超过100股：取100股整数倍（向下取整）
                # 2. 不足100股：计算实际需要卖出的数量
                if raw_size >= 100:
                    order_size = int(raw_size // 100) * 100  # 100股整数倍
                else:
                    # 计算实际需要卖出的数量，而不是全部卖出
                    order_size = min(int(raw_size), position.size)

                # 确保不超卖
                order_size = min(order_size, position.size)

                if order_size > 0:
                    order = self.sell(data=data, size=order_size, exectype=bt.Order.Market)
                    # 使用订单ID作为键记录卖出订单信息
                    order_id = order.ref
                    self.sell_orders[order_id] = {
                        'order': order,
                        'data': data,
                        'size': order_size,
                        'price': price,
                        'expected_cash': order_size * price
                    }
                    self.expected_cash_inflow += order_size * price
                    self.pending_orders += 1  # 增加待处理订单计数

                    if self.params.verbose:
                        print(
                            f"[模型生成卖出] {data._name}: 提交日期 {self.datetime.date(0)},价格 {price},目标权重 {target_weight:.2%}, "
                            f"理论数量 {raw_size:.2f}, 实际数量 {order_size}股, 预计资金流入: {order_size * price:.2f}")

    def _execute_buy_orders(self, total_value, cash_value):
        """执行买入订单逻辑"""
        # 按目标权重降序排序，优先处理高权重标的
        sorted_targets = sorted(
            self.target_weights.items(),
            key=lambda x: x[1],  # 按权重值排序
            reverse=True  # 降序：从大到小
        )

        # 计算可用资金（当前现金 + 预计卖出流入 - 已预留资金）
        available_cash = cash_value + self.expected_cash_inflow

        for data, target_weight in sorted_targets:  # 遍历排序后的列表
            # 确保有当前交易日的数据
            if len(data) < 1:
                if self.params.verbose:
                    print(f"警告: {data._name} 缺少价格数据，跳过买入逻辑")
                continue

            # 获取有效价格
            price = self._get_valid_price(data)
            if price is None:
                if self.params.verbose:
                    print(f"警告: {data._name} 没有有效价格数据，跳过买入逻辑")
                continue

            position = self.getposition(data)
            position_value = position.size * price  # 使用有效价格
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight
            delta_value = target_value - position_value

            # 仅处理买入逻辑（delta_value > 1e-5）
            if delta_value > 1e-5:
                raw_size = delta_value / price
                order_size = int(raw_size // 100) * 100  # 100股整数倍

                if order_size > 0:
                    required_cash = order_size * price

                    # 使用预计可用资金（当前现金 + 预计卖出流入 - 已预留资金）
                    available_cash = cash_value + self.expected_cash_inflow - self.reserved_cash

                    if available_cash >= required_cash:
                        order = self.buy(data=data, size=order_size, exectype=bt.Order.Market)
                        self.reserved_cash += required_cash  # 预留资金增加
                        self.pending_orders += 1  # 增加待处理订单计数

                        # 日志记录
                        if self.params.verbose:
                            print(
                                f"[模型生成买入] {data._name}: 提交日期 {self.datetime.date(0)} "
                                f"价格 {price:.2f} 目标权重 {target_weight:.2%} "
                                f"实际数量 {order_size}股, 占用资金: {required_cash:.2f}, "
                                f"可用资金: {available_cash:.2f}"
                            )
                    else:
                        # 增强版资金不足提示
                        if self.params.verbose:
                            print(
                                f"资金不足: [{data._name}]需{required_cash:.2f}元 "
                                f"（权重优先级: {target_weight:.2%}） "
                                f"可用资金: {available_cash:.2f}元"
                            )
                else:
                    if self.params.verbose:
                        print(f"跳过微小交易: {data._name} 理论数量 {raw_size:.2f} < 100股")

    # 添加缺失的 _get_valid_price 方法
    def _get_valid_price(self, data):
        """获取有效价格，如果没有则返回None"""
        # 尝试按优先级获取不同类型的价格
        price_types = [
            ('close', data.close[0] if len(data.close) > 0 else None),
            ('current', data[0] if len(data) > 0 else None),
            ('open', data.open[0] if len(data.open) > 0 else None)
        ]

        for price_type, price in price_types:
            if price is not None and price > 0:
                if self.params.verbose and price_type != 'close':
                    print(f"提示: {data._name} 使用 {price_type} 价格: {price:.2f}")
                return price

        return None
    def notify_order(self, order):
        # 订单状态变化时的回调
        order_id = order.ref  # 获取订单ID

        if order.status in [order.Completed]:
            if order.isbuy():
                direction = "买入"
                # 订单成交后，减少预留资金
                self.reserved_cash -= order.executed.price * order.executed.size
                if self.reserved_cash < 0:  # 防止浮点数误差导致负值
                    self.reserved_cash = 0
            elif order.issell():
                direction = "卖出"
                # 移除已完成的卖出订单信息
                if order_id in self.sell_orders:
                    del self.sell_orders[order_id]

            print(
                f"实际{direction} {order.data._name} 执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 订单取消或拒绝时的处理
            if order.isbuy():
                # 释放预留资金
                self.reserved_cash -= order.size * order.price
                if self.reserved_cash < 0:
                    self.reserved_cash = 0
            elif order.issell() and order_id in self.sell_orders:
                # 减少预计流入
                self.expected_cash_inflow -= self.sell_orders[order_id]['expected_cash']
                del self.sell_orders[order_id]

        # 减少待处理订单计数
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.pending_orders -= 1
            # 当所有订单都处理完毕时，打印持仓情况
            if self.pending_orders == 0:
                self._print_current_positions()

    def _print_current_positions(self):
        """打印当前持仓情况"""
        total_value = self.broker.get_value()
        cash = self.broker.get_cash()

        # 如果总市值计算失败，尝试手动计算
        if not self._is_valid_value(total_value) or total_value == 0:
            total_value = self._calculate_total_value()

        print("\n===== 当前持仓情况 =====")
        print(f"日期: {self.datetime.date(0)} | 总市值: ¥{total_value:,.2f} | 可用资金: ¥{cash:,.2f}")
        print("代码\t\t持仓数量\t市值\t\t持仓占比")

        # 计算每个持仓的市值和占比
        for data in self.datas:
            position = self.getposition(data)
            if position.size != 0:
                price = self._get_latest_price(data)
                if price > 0:
                    market_value = position.size * price
                    weight = market_value / total_value if total_value > 0 else 0
                    print(f"{data._name}\t{position.size:,.0f}\t\t¥{market_value:,.2f}\t{weight:.2%}")

        # 打印现金占比
        cash_weight = cash / total_value if total_value > 0 else 0
        print(f"现金\t\t-\t\t¥{cash:,.2f}\t{cash_weight:.2%}")
        print("=======================\n")

    def _get_latest_price(self, data):
        """获取最新可用的价格"""
        # 优先使用收盘价，然后是当前价、开盘价
        if len(data.close) > 0 and data.close[0] > 0:
            return data.close[0]
        elif len(data) > 0 and data[0] > 0:
            return data[0]
        elif len(data.open) > 0 and data.open[0] > 0:
            return data.open[0]
        else:
            return 0

    def _is_valid_value(self, value):
        """检查值是否有效（不是None、NaN或零）"""
        if value is None:
            return False
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return False
        if value == 0:
            return False
        return True

    def _calculate_total_value(self):
        """手动计算总市值"""
        total_value = self.broker.get_cash()

        for data in self.datas:
            position = self.getposition(data)
            if position.size != 0:
                price = self._get_latest_price(data)
                if price > 0:
                    market_value = position.size * price
                    total_value += market_value

        return total_value





