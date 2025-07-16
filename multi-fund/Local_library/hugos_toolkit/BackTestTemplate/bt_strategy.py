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


# 策略模板


class SignalStrategy(bt.Strategy):

    params = (
        ("open_threshold", 0.301),
        ("close_threshold", -0.301),
        ("show_log", True),
    )

    def log(self, txt, dt=None, show_log: bool = True):
        # log记录函数
        dt = dt or self.datas[0].datetime.date(0)
        if show_log:
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):

        self.dataclose = self.data.close
        self.signal = self.data.GSISI
        self.order = None

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,  # 订单编号
                        order.executed.price,  # 成交价
                        order.executed.value,  # 成交额
                        order.executed.comm,  # 佣金
                        order.executed.size,  # 成交量
                        order.data._name,  # 股票名称
                    ),
                    show_log=self.p.show_log,
                )
            else:  # Sell
                self.log(
                    "SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                    ),
                    show_log=self.p.show_log,
                )

    def next(self):

        # 取消之前未执行的订单
        if self.order:
            self.cancel(self.order)

        if self.position:
            if (
                self.signal[0] <= self.params.close_threshold
                and self.signal[-1] <= self.params.close_threshold
            ):
                self.log("收盘价Close, %.2f" % self.dataclose[0], show_log=self.p.show_log)
                self.log(
                    "设置卖单SELL CREATE, %.2f信号为:%.2f,阈值为:%.2f"
                    % (self.dataclose[0], self.signal[0], self.params.close_threshold),
                    show_log=self.p.show_log,
                )
                self.order = self.order_target_value(target=0.0)

        elif (
            self.signal[0] >= self.params.open_threshold
            and self.signal[-1] >= self.params.open_threshold
        ):
            self.log("收盘价Close, %.2f" % self.dataclose[0], show_log=self.p.show_log)
            self.log(
                "设置买单 BUY CREATE, %.2f,信号为:%.2f,阈值为:%.2f"
                % (self.dataclose[0], self.signal[0], self.params.open_threshold),
                show_log=self.p.show_log,
            )
            self.order = self.order_target_percent(target=0.95)


class TopicStrategy(bt.Strategy):

    params = (
        ("show_log", True),
        ('percents', 0.2),
    )


    def __init__(self):
        self.lowest_stocks = []  # 用于存储收盘价格最低的5只股票


    def start(self):
        # 获取股票列表
        self.stocks = self.getdatanames()

    def next(self):
        # 每天开盘前执行
        if self.datas[0].open[0] > 0:  # 确保是开盘时间
            # 获取前一天的收盘价
            previous_closes = {stock:  self.getdatabyname(stock).close[-1] for stock in self.stocks}

            # 找出收盘价格最低的5只股票
            self.lowest_stocks = sorted(previous_closes, key=previous_closes.get)[:5]

            # 计算总投资金额
            total_percents = self.params.percents * self.broker.getcash()

            # 买入最低价格的5只股票
            for stock in self.lowest_stocks:
                if stock not in self.positions or self.positions[stock] == 0:
                    self.buy(stock, exectype=bt.Order.Market, size=total_percents / 5)

            # 卖出不在最低5只股票中的股票
            for stock in self.positions:
                if stock not in self.lowest_stocks:
                    self.sell(stock, exectype=bt.Order.Market)

    def stop(self):
        # 每天收盘后执行
        for stock in self.positions:
            self.close(stock)

class TopicStrategy345(bt.Strategy):
    params = (
        ("ranking_threshold", 5),
        ("show_log", True),
    )

    def __init__(self):
        # 初始化一个列表来存储当前持仓的数据源
        self.stocks = []
        # 初始化一个字典来存储每个数据源的目标持仓市值百分比
        self.target_positions = {}

    def next(self):
        # 获取当前日期的所有数据源的收盘价
        #ranks = [(data._name, data.rank[0]) for data in self.datas]
        #ranks =  [(self.data_names[data], data.rank[0]) for data in self.datas]
        ranks =  [self.data._name,self.data.rank]
        # 找出收盘价最高的5只股票
        top5 = sorted(ranks, key=lambda x: x[1], reverse=False)[:5]

        # 计算总资产
        total_assets = self.broker.getcash()

        # 设置目标持仓市值为总资产的20%
        for stock, rank in top5:
            # 计算目标持仓市值
            target_value = total_assets * 0.2

            # 获取当前持仓
            position = self.getposition(stock)

            # 如果当前持仓不足，买入股票
            if position is None or position.size < target_value / close:
                self.order_target_value(stock, target_value)
            # 如果当前持仓过多，卖出股票
            elif position.size > target_value / close:
                self.order_target_value(stock, target_value)
            # 记录目标持仓市值
            self.target_positions[stock] = target_value

    def log(self, txt, dt=None):
        ''' 用于记录日志的自定义方法 '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        ''' 订单执行通知 '''
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self.log(f'Order completed for {order.data._name}; executed at {order.executed.price}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order failed for {order.data._name}; status: {order.status}')


class LowRankStrategy(bt.Strategy):
    params = (
        ('buy_threshold', 5),  # 买入阈值
        ('stake', 0.1),  # 每只股票的仓位比例
        ("show_log", True),
    )

    def __init__(self):
        self.inds = {}  # 存储每只股票的指标数据
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]['prev_rank'] = d.rank(0)  # 交易日排名
            self.inds[d]['close'] = d.close(0)  # 交易日排名

    def next(self):
        for d, ind in self.inds.items():
            pos = self.getposition(d).size
            if pos > 0:
                # 当前有头寸
                if ind["prev_rank"][0]  > self.params.buy_threshold and ind["close"][0]>0:
                    # 卖出信号
                    self.order = self.order_target_percent(data=d, target=0.0)
                    print(f'Sell {d._name}, Size: {pos}, Prev Close: {ind["prev_rank"][0]:.2f}, date: {d.datetime.date(0):.2f}')
            else:
                # 当前无头寸
                if ind["prev_rank"][0] <= self.params.buy_threshold and ind["close"][0]>0:
                    # 买入信号
                    # size = int(self.broker.get_cash() * self.params.stake / d.close[0])
                    self.order = self.order_target_percent(data=d, target=self.params.stake)
                    print(f'Buy {d._name}, Size: 0.1, Prev rank: {ind["prev_rank"][0]:.2f}, date: {d.datetime.date(0)}')
    def stop(self):
        print('Strategy completed')


class LowRankStrategy_new_20241107_1(bt.Strategy):
    params = (
        ('buy_threshold', 1),  # 买入阈值
        ('max_exposure', 0.8),  # 最大仓位敞口
        ("show_log", True),
    )

    def __init__(self):
        self.inds = {}  # 存储每只股票的指标数据
        self.add_timer(when=bt.Timer.SESSION_END)  # 添加定时器，用于在每个交易时段结束时执行操作
        self.whitelist = pd.read_csv('/mnt/list.csv', usecols=[0]).squeeze().tolist()
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]['prev_rank'] = d.rank(0)  # 交易日信号
            self.inds[d]['close'] = d.close(0)  # 交易日信号
            self.inds[d]['buy_date'] = None  # 初始化买入日期为None
            self.inds[d]['purchase_price'] = None  # 记录买入价格

    def next(self):
        # 计算当前可用资金
        cash = self.broker.get_cash()

        # 统计满足买入条件的股票数量
        buy_count = sum(1 for d, ind in self.inds.items() if ind["prev_rank"][0] >= self.params.buy_threshold
                        and self.getposition(d).size == 0 and d._name in self.whitelist)

        if buy_count > 0:
            # 计算每只股票的买入金额
            buy_amount = cash * self.params.max_exposure / buy_count

            for d, ind in self.inds.items():
                pos = self.getposition(d).size
                if pos == 0 and ind["prev_rank"][0] >= self.params.buy_threshold and d._name in self.whitelist:
                    # 买入信号
                    self.order = self.order_target_value(data=d, target=buy_amount)
                    ind['buy_date'] = self.data.datetime.date(0)  # 记录买入日期
                    ind['purchase_price'] = d.close[0]  # 记录买入价格
                    print(f'Buy {d._name}, Amount: {buy_amount:.2f}, Prev rank: {ind["prev_rank"][0]:.2f}, date: {d.datetime.date(0)}')
                elif pos > 0:
                    current_price = d.close[0]
                    if ind["prev_rank"][0] < self.params.buy_threshold:
                        # 检查是否持有 10 天以上
                        if ind['buy_date'] and (self.data.datetime.date(0) - ind['buy_date']).days >= 5:
                            # 卖出信号
                            self.order = self.order_target_percent(data=d, target=0.0)
                            print(f'Sell {d._name}, Size: {pos}, Prev rank: {ind["prev_rank"][0]:.2f}, date: {d.datetime.date(0)}')
                        else:
                            print(f'Hold {d._name}, Not enough holding period, date: {d.datetime.date(0)}')
                    elif ind['purchase_price'] and (current_price / ind['purchase_price'] - 1) <= -0.05:
                        # 亏损 5%卖出条件
                        self.order = self.order_target_percent(data=d, target=0.0)
                        print(f'Sell {d._name} due to 5% loss, Size: {pos}, Prev rank: {ind["prev_rank"][0]:.2f}, date: {d.datetime.date(0)}')

    def timer(self):
        # 你可以在这里执行定时任务，例如记录日志、检查持仓等
        pass

    def stop(self):
        print('Strategy completed')

class LowRankStrategy_new_20241107_2(bt.Strategy):
    params = (
        ('buy_threshold', 1),  # 买入阈值
        ('max_exposure', 1),  # 最大仓位敞口
        ("show_log", True),
    )

    def __init__(self):
        self.inds = {}  # 存储每只股票的指标数据
        self.add_timer(when=bt.Timer.SESSION_END)  # 添加定时器，用于在每个交易时段结束时执行操作
        self.whitelist = pd.read_csv('C:/temp/important/whitelist.csv', usecols=[0]).squeeze().tolist()
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]['prev_rank'] = d.rank(0)  # 交易日信号
            self.inds[d]['close'] = d.close(0)  # 交易日信号
            self.inds[d]['buy_date'] = None  # 初始化买入日期为None
            self.inds[d]['purchase_price'] = None  # 记录买入价格

    def next(self):
        # 计算当前可用资金
        cash = self.broker.get_cash()

        # 统计满足买入条件的股票数量
        buy_count = sum(1 for d, ind in self.inds.items() if ind["prev_rank"][0] >= self.params.buy_threshold
                        and self.getposition(d).size == 0 and d._name in self.whitelist)

        if buy_count > 0:
            # 计算每只股票的买入金额
            buy_amount = cash * self.params.max_exposure / buy_count

            for d, ind in self.inds.items():
                pos = self.getposition(d).size
                if pos == 0 and ind["prev_rank"][0] >= self.params.buy_threshold:
                    if d._name in self.whitelist:
                        # 买入信号
                        self.order = self.order_target_value(data=d, target=buy_amount)
                        ind['buy_date'] = self.data.datetime.date(0)  # 记录买入日期
                        ind['purchase_price'] = d.close[0]  # 记录买入价格
                        logging.info(f'Buy {d._name}, Amount: {buy_amount:.2f}, Prev rank: {ind["prev_rank"][0]:.2f}')
                    else:
                        logging.info(f'Intercepted buy for {d._name}, not in whitelist')
                elif pos > 0:
                    current_price = d.close[0]
                    if ind["prev_rank"][0] < self.params.buy_threshold:
                        # 检查是否持有 10 天以上
                        if ind['buy_date'] and (self.data.datetime.date(0) - ind['buy_date']).days >= 5:
                            # 卖出信号
                            self.order = self.order_target_percent(data=d, target=0.0)
                            logging.info(f'Sell {d._name}, Size: {pos}, Prev rank: {ind["prev_rank"][0]:.2f}')
                        else:
                            logging.info(f'Hold {d._name}, Not enough holding period, date: {d.datetime.date(0)}')
                    elif ind['purchase_price'] and (current_price / ind['purchase_price'] - 1) <= -0.05:
                        # 亏损 5%卖出条件
                        self.order = self.order_target_percent(data=d, target=0.0)
                        logging.info(f'Sell {d._name} due to 5% loss, Size: {pos}, Prev rank: {ind["prev_rank"][0]:.2f}')

    def timer(self):
        # 你可以在这里执行定时任务，例如记录日志、检查持仓等
        pass

    def stop(self):
        logging.info('Strategy completed')

class LowRankStrategy_new(bt.Strategy):
    # 配置日志文件
    logging.basicConfig(filename='C:/temp/important/strategy.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    params = (
        ('buy_threshold', 1),  # 买入阈值
        ('max_exposure', 0.95),  # 最大仓位敞口
        ("show_log", True),
        ('min_cash_threshold', 50000)  # 最低资金阈值
    )

    def __init__(self):
        self.inds = {}  # 存储每只股票的指标数据
        self.add_timer(when=bt.Timer.SESSION_END)  # 添加定时器，用于在每个交易时段结束时执行操作
        self.whitelist = pd.read_csv('C:/temp/important/whitelist.csv', usecols=[0]).squeeze().tolist()
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]['prev_rank'] = d.rank(0)  # 交易日信号
            self.inds[d]['close'] = d.close(0)  # 交易日信号
            self.inds[d]['buy_date'] = None  # 初始化买入日期为None
            self.inds[d]['purchase_price'] = None  # 记录买入价格

    def next(self):
        # 计算当前可用资金
        cash = self.broker.get_cash()

        # 统计满足买入条件的股票数量
        buy_count = sum(1 for d, ind in self.inds.items() if ind["prev_rank"][0] >= self.params.buy_threshold
                        and self.getposition(d).size == 0 )

        if buy_count > 0:
            # 检查可用资金是否低于阈值

            # 计算每只股票的买入金额
            buy_amount = cash * self.params.max_exposure / buy_count

            for d, ind in self.inds.items():
                pos = self.getposition(d).size
                if pos == 0 and ind["prev_rank"][0] >= self.params.buy_threshold:
                    if d._name in self.whitelist:
                        if cash > self.params.min_cash_threshold:
                           self.buy_stock(d, buy_amount, ind)
                        else:
                            logging.info(f'Hold {d._name}, date: {d.datetime.date(0)},Not enough cash to buy, current cash: {cash:.2f}')
                    else:
                        logging.info(f'Intercepted buy for {d._name}, not in whitelist')
                elif pos > 0:
                    current_price = d.close[0]
                    if ind["prev_rank"][0] < self.params.buy_threshold:
                        if (self.data.datetime.date(0) - ind['buy_date']).days >= 5:
                            self.sell_stock(d, pos, ind)
                        else:
                            logging.info(f'Hold {d._name}, Not enough holding period, date: {d.datetime.date(0)}')
                    ##############强制止损在-5%到-10%之间
                    elif (current_price / ind['purchase_price'] - 1) <= -0.8:
                        logging.info(
                            f"Sell {d._name}, date: {d.datetime.date(0)}, Purchase Price: {ind['purchase_price']:.2f}, Current Price: {current_price:.2f}, Loss exceeded 5%")
                        self.sell_stock(d, pos, ind)

    def buy_stock(self, data, amount, ind):
        self.order = self.order_target_value(data=data, target=amount)
        ind['buy_date'] = self.data.datetime.date(0)  # 记录买入日期
        ind['purchase_price'] = data.close[0]  # 记录买入价格
        logging.info(f'Buy {data._name},time {self.data.datetime.date(0) },Amount: {amount:.2f}, Prev rank: {ind["prev_rank"][0]:.2f}')

    def sell_stock(self, data, size, ind):
        self.order = self.order_target_percent(data=data, target=0.0)
        logging.info(f'Sell {data._name},time {self.data.datetime.date(0) }, Size: {size}, Prev rank: {ind["prev_rank"][0]:.2f}')

    def timer(self):
        # 你可以在这里执行定时任务，例如记录日志、检查持仓等
        pass

    def stop(self):
        logging.info('Strategy completed')
        # 获取根日志记录器
        logger = logging.getLogger()
        # 关闭所有处理器
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)


import backtrader as bt
import pandas as pd


class Multi_Weight_Strategy(bt.Strategy):
    params = (
        ("verbose", True),  # 是否打印交易日志
        ("show_log", True),
    )

    def __init__(self):
        self.weights = {}
        self.target_weights = {}
        # 设置订单在收盘时执行（确保卖出资金立即释放）
        self.broker.set_coc(False)  # 关闭收盘执行
        self.broker.set_coo(True)  # 启用开盘执行 [1,9](@ref)

    def next(self):
        if self.params.verbose:
            print(f"\n策略权重生成日期: {self.datetime.date()}")


        # 确定所有资产的target_weights
        self.target_weights.clear()  # 清空旧权重
        for data in self.datas:
            if len(data) < 1 or data.open[1] <= 0:
                continue
            if len(data.w) > 0 and not pd.isnull(data.w[0]):
                self.target_weights[data] = data.w[0]

        total_value = self.broker.get_value()
        cash_value = self.broker.get_cash()
        print(
            f"日期: {self.datetime.date(0)} | "
            f"账户资金: ¥{cash_value:,.2f} | "
            f"总市值: ¥{total_value:,.2f}"
        )
        # 关键修改：判断是否为最后一个交易日（避免索引越界）
        # 使用主数据（data0）进行判断
        if len(self.data0) == self.data0.buflen():
            print(f"⚠️ 警告：{self.datetime.date(0)} 是最后一个交易日，跳过交易逻辑避免索引错误")
            return



        # ================= 第一阶段：卖出订单处理（关键修改开始）=================
        for data, target_weight in self.target_weights.items():
            if len(data) < 1 or data.open[1] <= 0:
                continue

            position = self.getposition(data)
            position_value = position.size * data.open[1]
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight
            delta_value = target_value - position_value

            if delta_value < -1e-5:  # 需要卖出
                price = data.open[1]
                # 原始计算值（可能含小数）
                raw_size = abs(delta_value) / price

                # 处理规则：
                # 1. 超过100股：取100股整数倍（向下取整）
                # 2. 不足100股：全部卖出（A股规则要求零股必须一次性卖出）[6](@ref)
                if raw_size >= 100:
                    order_size = int(raw_size // 100) * 100  # 100股整数倍
                else:
                    order_size = position.size  # 零头全部卖出

                # 确保不超卖
                order_size = min(order_size, position.size)

                if order_size > 0:
                    self.sell(data=data, size=order_size, exectype=bt.Order.Market)
                    if self.params.verbose:
                        print(f"[卖出] {data._name}: 成交日期 {self.datetime.date(0)},价格 {price},目标权重 {target_weight:.2%}, "
                              f"理论数量 {raw_size:.2f}, 实际数量 {order_size} "
                              f"({'零股全卖' if order_size < 100 else '100整数倍'})")
        # ================= 卖出订单处理（关键修改结束）=================
            # 2. 强制Broker处理订单（使卖出成交）
        #self.broker.run()  # 立即执行挂单

        # -------------------------------
        # 第二阶段：处理所有买入订单（按目标权重降序执行）
        # -------------------------------

        # 按目标权重降序排序，优先处理高权重标的
        sorted_targets = sorted(
            self.target_weights.items(),
            key=lambda x: x[1],  # 按权重值排序
            reverse=True  # 降序：从大到小
        )

        for data, target_weight in sorted_targets:  # 遍历排序后的列表
            if len(data) < 1 or data.open[1] <= 0:
                continue

            position = self.getposition(data)
            position_value = position.size * data.open[1]
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight
            delta_value = target_value - position_value

            # 仅处理买入逻辑（delta_value > 0）
            if delta_value > 1e-5:
                price = data.open[1]
                raw_size = delta_value / price
                order_size = int(raw_size // 100) * 100  # 100股整数倍

                if order_size > 0:
                    required_cash = order_size * price
                    cash_available = self.broker.get_cash()

                    if cash_available >= required_cash:
                        self.buy(data=data, size=order_size, exectype=bt.Order.Market)
                        # 日志记录
                        if self.params.verbose:
                            print(
                                f"[买入] {data._name}: 提交日期 {self.datetime.date(0)} "
                                f"价格 {price:.2f} 目标权重 {target_weight:.2%} "
                                f"实际数量 {order_size}股"
                            )
                    else:
                        # 增强版资金不足提示（含标的优先级信息）
                        if self.params.verbose:
                            print(
                                f"资金不足: [{data._name}]需{required_cash:.2f}元 "
                                f"（权重优先级: {target_weight:.2%}） "
                                f"可用资金: {cash_available:.2f}元"
                            )
                else:
                    if self.params.verbose:
                        print(f"跳过微小交易: {data._name} 理论数量 {raw_size:.2f} < 100股")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                direction = "买入"
            elif order.issell():
                direction = "卖出"
            print(f"{direction} {order.data._name} 执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}")

######################################
##################################
##################
class Multi_Weight_Strategy_new(bt.Strategy):
    params = (
        ("verbose", True),
        ("show_log", True),
        ("commission", 0.0003),  # 添加佣金参数
    )

    def __init__(self):
        self.target_weights = {}
        self.sell_orders = {}  # 跟踪卖出订单
        self.buy_orders = {}  # 跟踪买入订单
        self.commission_rate = self.p.commission  # 保存佣金率

        # 设置订单在开盘时执行
        self.broker.set_coo(True)  # 启用开盘执行

    def next(self):
        # 清除已执行的订单
        self.sell_orders = {d: o for d, o in self.sell_orders.items() if o.status < o.Completed}
        self.buy_orders = {d: o for d, o in self.buy_orders.items() if o.status < o.Completed}

        # 计算目标权重
        self.target_weights.clear()
        valid_datas = []  # 存储有有效数据的数据源
        price_data = {}  # 存储每个数据源的价格信息

        for data in self.datas:
            # 检查当前数据源是否有足够的数据
            if len(data) < 2:
                if self.p.verbose:
                    print(f"跳过数据不足的股票: {data._name} (长度: {len(data)})")
                continue

            # 检查下一个交易日是否有数据
            has_next_data = len(data) < data.buflen()

            # 获取价格数据并检查有效性
            next_open = data.open[1] if has_next_data else data.close[0]  # 最后交易日使用收盘价
            if pd.isna(next_open) or next_open <= 0:
                if self.p.verbose and has_next_data:
                    print(f"跳过无效价格的股票: {data._name} (价格: {next_open})")
                continue

            if hasattr(data, 'w') and len(data.w) > 0 and not pd.isnull(data.w[0]):
                self.target_weights[data] = data.w[0]
                valid_datas.append(data)
                price_data[data] = {
                    'next_open': next_open,
                    'has_next_data': has_next_data
                }

        if not valid_datas:
            if self.p.verbose:
                print(f"跳过无有效数据的交易日: {self.datetime.date()}")
            return  # 没有有效数据，跳过

        # 计算总资产和可用资金
        total_value = self.broker.get_value()
        cash_available = self.broker.get_cash()

        # 安全检查：确保总资产有效
        if pd.isna(total_value) or total_value <= 0:
            total_value = cash_available  # 回退到可用资金
            if self.p.verbose:
                print(f"警告: 总资产计算为NaN或零，使用可用资金替代: {total_value}")

        if self.p.verbose:
            print(f"\n日期: {self.datetime.date()}")
            print(f"总价值: {total_value:.2f}, 可用资金: {cash_available:.2f}")

        # 阶段1: 处理卖出订单
        for data in valid_datas:
            target_weight = self.target_weights[data]
            if data in self.sell_orders or data in self.buy_orders:
                continue  # 已有待处理订单

            position = self.getposition(data)
            if not position.size:
                continue  # 没有持仓

            price_info = price_data[data]
            next_open = price_info['next_open']
            has_next_data = price_info['has_next_data']

            position_value = position.size * next_open
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight

            # 如果当前权重大于目标权重，需要卖出
            if current_weight > target_weight + 0.001:
                delta_value = position_value - target_value
                price = next_open
                raw_size = delta_value / price

                # 处理A股100股整数倍的规则
                if raw_size >= 100:
                    order_size = int(raw_size // 100) * 100
                else:
                    order_size = position.size  # 零股全部卖出

                order_size = min(order_size, position.size)  # 确保不超卖

                if order_size > 0:
                    # 只有当有下一个交易日数据时才提交订单
                    if has_next_data:
                        order = self.sell(data=data, size=order_size, exectype=bt.Order.Market)
                        self.sell_orders[data] = order
                        action = "提交卖出"
                    else:
                        action = "计划卖出(最后交易日不执行)"

                    if self.p.verbose:
                        print(f"{action} {data._name}: {order_size}股, 价格: {price:.2f}")

        # 阶段2: 处理买入订单
        # 按目标权重降序排序，优先处理高权重标的
        sorted_targets = sorted(
            [(data, target_weight) for data, target_weight in self.target_weights.items()],
            key=lambda x: x[1],
            reverse=True
        )

        for data, target_weight in sorted_targets:
            if data not in valid_datas:
                continue  # 跳过无效数据
            if data in self.sell_orders or data in self.buy_orders:
                continue  # 已有待处理订单

            price_info = price_data[data]
            next_open = price_info['next_open']
            has_next_data = price_info['has_next_data']

            position = self.getposition(data)
            position_value = position.size * next_open if position.size else 0
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight

            # 如果当前权重小于目标权重，需要买入
            if current_weight < target_weight - 0.001:
                delta_value = target_value - position_value
                price = next_open

                # 安全检查：确保价格有效
                if pd.isna(price) or price <= 0:
                    if self.p.verbose:
                        print(f"跳过无效价格: {data._name} 价格 {price}")
                    continue

                raw_size = delta_value / price

                # 安全检查：确保计算结果有效
                if pd.isna(raw_size) or raw_size <= 0:
                    if self.p.verbose:
                        print(f"跳过无效计算: {data._name} raw_size={raw_size}")
                    continue

                # A股100股整数倍规则
                order_size = int(raw_size // 100) * 100

                if order_size > 0:
                    required_cash = order_size * price * (1 + self.commission_rate)  # 包含佣金

                    # 使用估计的可用资金（考虑已提交的卖出订单）
                    estimated_cash = cash_available
                    for sell_data, sell_order in self.sell_orders.items():
                        if sell_order.status < sell_order.Completed:
                            sell_position = self.getposition(sell_data)
                            if sell_position.size > 0:
                                # 从价格数据中获取对应股票的价格
                                if sell_data in price_data:
                                    sell_price = price_data[sell_data]['next_open']
                                    estimated_cash += sell_position.size * sell_price * (1 - self.commission_rate)

                    # 安全检查：确保估计资金有效
                    if pd.isna(estimated_cash) or estimated_cash <= 0:
                        estimated_cash = cash_available  # 回退到实际可用资金

                    if estimated_cash >= required_cash:
                        # 只有当有下一个交易日数据时才提交订单
                        if has_next_data:
                            order = self.buy(data=data, size=order_size, exectype=bt.Order.Market)
                            self.buy_orders[data] = order
                            action = "提交买入"
                        else:
                            action = "计划买入(最后交易日不执行)"

                        if self.p.verbose:
                            print(f"{action} {data._name}: {order_size}股, 价格: {price:.2f}, "
                                  f"估计资金: {estimated_cash:.2f}, 需要: {required_cash:.2f}")
                    else:
                        if self.p.verbose:
                            print(f"资金不足: {data._name} 需要 {required_cash:.2f}, 可用: {estimated_cash:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"买入 {order.data._name} 完成: 价格 {order.executed.price:.2f}, "
                      f"数量 {order.executed.size}, 费用 {order.executed.value:.2f}")
            elif order.issell():
                print(f"卖出 {order.data._name} 完成: 价格 {order.executed.price:.2f}, "
                      f"数量 {order.executed.size}, 收入 {order.executed.value:.2f}")

            # 从跟踪字典中移除已完成订单
            if order.data in self.sell_orders:
                del self.sell_orders[order.data]
            elif order.data in self.buy_orders:
                del self.buy_orders[order.data]

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"订单 {order.data._name} 取消/拒绝: {order.status}")

            # 从跟踪字典中移除失败订单
            if order.data in self.sell_orders:
                del self.sell_orders[order.data]
            elif order.data in self.buy_orders:
                del self.buy_orders[order.data]





