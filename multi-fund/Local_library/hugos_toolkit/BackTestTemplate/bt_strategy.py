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
        self.broker.set_coc(True)  # cheat-on-close

    def next(self):
        if self.params.verbose:
            print(f"\n日期: {self.datetime.date()}")

        # 确定所有资产的target_weights
        self.target_weights.clear()  # 清空旧权重
        for data in self.datas:
            if len(data) < 1 or data.close[0] <= 0:
                continue
            if len(data.w) > 0 and not pd.isnull(data.w[0]):
                self.target_weights[data] = data.w[0]

        total_value = self.broker.get_value()

        # -------------------------------
        # 第一阶段：优先处理所有卖出订单
        # -------------------------------
        for data, target_weight in self.target_weights.items():
            if len(data) < 1 or data.close[0] <= 0:
                continue

            position = self.getposition(data)
            position_value = position.size * data.close[0]
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight
            delta_value = target_value - position_value

            # 只处理卖出逻辑
            if delta_value < -1e-5:  # 需要卖出
                price = data.close[0]
                order_size = delta_value / price
                self.sell(data=data, size=abs(order_size), price=price)
                if self.params.verbose:
                    print(f"[卖出] {data._name}: 目标权重 {target_weight:.2%}, 数量 {abs(order_size):.2f}")

        # -------------------------------
        # 第二阶段：处理所有买入订单
        # -------------------------------
        for data, target_weight in self.target_weights.items():
            if len(data) < 1 or data.close[0] <= 0:
                continue

            position = self.getposition(data)
            position_value = position.size * data.close[0]
            current_weight = position_value / total_value if total_value > 0 else 0
            target_value = total_value * target_weight
            delta_value = target_value - position_value

            # 只处理买入逻辑
            if delta_value > 1e-5:  # 需要买入
                price = data.close[0]
                order_size = delta_value / price
                self.buy(data=data, size=order_size, price=price)
                if self.params.verbose:
                    print(f"[买入] {data._name}: 目标权重 {target_weight:.2%}, 数量 {order_size:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                direction = "买入"
            elif order.issell():
                direction = "卖出"
            print(f"{direction} {order.data._name} 执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}")




