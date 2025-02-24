import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/cn_data/"  # target_dir
import sys
import os
local_path = os.getcwd()
local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_weight_bt,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import Multi_Weight_Strategy
import pandas as pd
qlib.init(provider_uri=provider_uri, region=REG_CN)

test_period = ("2024-09-01", "2025-02-07")
ranked_data = pd.read_csv('c:\\temp\\new_industry_kfold_stock_20250219.csv',
                          parse_dates=['date'],
                          index_col=['date', 'code'])

ranked_data.rename_axis(index={'date': 'datetime'}, inplace=True)
# 筛选出 datetime 大于 '2024-09-01' 的所有数据行
raw_data = ranked_data.loc[(ranked_data.index.get_level_values('datetime') >= '2024-09-01'), :]
raw_data = raw_data.reset_index(level='code')
new_df=raw_data

bt_result = get_weight_bt(
    new_df,
    name="code",
    strategy=Multi_Weight_Strategy,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False},
    begin_dt=test_period[0],
    end_dt=test_period[1],
)

benchmark_old = ["SH000300"]
# data, benchmark = get_backtest_data(ranked_data, test_period[0], test_period[1], market, benchmark_old)
benchmark: pd.DataFrame = D.features(
    benchmark_old,
    fields=["$close"],
    start_time=test_period[0],
    end_time=test_period[1],
).reset_index(level=0, drop=True)
benchmark_ret: pd.Series = benchmark['$close'].pct_change()


trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()
TradeStatisticsAnalyzer = bt_result.result[0].analyzers._TradeStatisticsAnalyzer.get_analysis()
DailyPositionAnalyzer = bt_result.result[0].analyzers._DailyPositionAnalyzer.get_analysis()

OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()

trader_df = pd.DataFrame(trade_logger)
orders_df = pd.DataFrame(OrderAnalyzer)

algorithm_returns: pd.Series = pd.Series(
    bt_result.result[0].analyzers._TimeReturn.get_analysis()
)

#benchmark_new = benchmark[split:]
report = analysis_rets(algorithm_returns, bt_result.result, benchmark['$close'].pct_change(), use_widgets=True)

from plotly.offline import iplot
from plotly.offline import init_notebook_mode

init_notebook_mode()
for chart in report:
    iplot(chart)
#######################################################
#######################################################

#orders_df.groupby(['order_status']).size().reset_index(name='条数')









