import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/cn_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
from qlib.data import D
#import talib
from typing import List, Tuple, Dict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




######################
# 使用 .loc 方法选择特定日期的数据
#specific_date_data = raw_data.loc[(raw_data.index.get_level_values('datetime') == pd.Timestamp('2005-01-04'))]
#######################

def knn_stock_prediction(df, start_date, split_date):
    """
    使用 K-Nearest Neighbors (KNN) 算法进行股票价格预测，并返回包含 'instrument'、'datetime' 和 'Predicted_Signal' 的 DataFrame。

    参数:
    df (pandas.DataFrame): 包含股票数据的 DataFrame，索引包含 'instrument' 和 'datetime'
    start_date (str): 最初日期，格式为 'YYYY-MM-DD'
    split_date (str): 分割日期，格式为 'YYYY-MM-DD'

    返回:
    pandas.DataFrame: 包含 'instrument', 'datetime', 'Predicted_Signal' 的 DataFrame
    """
    # 删除缺失值
    df = df.dropna()

    # 替换列名中的特殊字符
    df.columns = [col.replace('$', '') for col in df.columns]

    # 只取最初日期之后的数据
    df = df[df.index.get_level_values('datetime') >= pd.Timestamp(start_date)]

    # 确保索引唯一且排序
    df = df[~df.index.duplicated(keep='first')].sort_index()

    instruments = df.index.get_level_values('instrument').unique()
    predicted_signals = []

    for instrument in instruments:
        instrument_df = df.xs(instrument, level='instrument').copy()  # 创建副本以避免 SettingWithCopyWarning
        # 检查数据量是否足够
        if len(instrument_df) < 2:
            print(f"Instrument: {instrument} has insufficient data. Skipping...")
            continue

        # 特征工程
        instrument_df['Open-Close'] = instrument_df['open'] - instrument_df['close']
        instrument_df['High-Low'] = instrument_df['high'] - instrument_df['low']

        X = instrument_df[['Open-Close', 'High-Low']]
        Y = np.where(instrument_df['close'].shift(-1) > instrument_df['close'], 1, -1)

        # 根据日期拆分数据集
        train_mask = instrument_df.index < pd.Timestamp(split_date)
        test_mask = instrument_df.index >= pd.Timestamp(split_date)

        X_train = X[train_mask]
        Y_train = Y[train_mask]
        X_test = X[test_mask]
        Y_test = Y[test_mask]

        # 检查训练集和测试集是否为空
        if len(X_train) == 0:
            print(f"Instrument: {instrument} has no training data. Skipping...")
            continue
        if len(X_test) == 0:
            print(f"Instrument: {instrument} has no test data. Skipping...")
            continue

        # 训练 KNN 模型
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, Y_train)

        # 计算准确率
        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

        print(
            f'Instrument: {instrument}, '
            f'Train_data Samples: {len(X_train)}, Train_data Accuracy: {accuracy_train:.2f}, '
            f'Test_data Samples: {len(X_test)}, Test_data Accuracy: {accuracy_test:.2f}'
        )

        # 生成预测信号
        instrument_df['Predicted_Signal'] = knn.predict(X)
        instrument_df['Predicted_Signal'] = instrument_df['Predicted_Signal'].replace(-1, 0)

        # 确保预测信号的索引唯一且排序
        instrument_df = instrument_df[~instrument_df.index.duplicated(keep='first')].sort_index()

        # 手动添加 'instrument' 列
        instrument_df['instrument'] = instrument

        # 选择需要的列并保留 'instrument' 和 'datetime' 作为普通列
        predicted_signal_df = instrument_df.reset_index()[['instrument', 'datetime', 'Predicted_Signal']]
        predicted_signals.append(predicted_signal_df)

    # 合并所有预测信号
    if predicted_signals:
        predicted_signals_df = pd.concat(predicted_signals, ignore_index=True)
    else:
        predicted_signals_df = pd.DataFrame(columns=['instrument', 'datetime', 'Predicted_Signal'])

    return predicted_signals_df




if __name__ == '__main__':
    test_period = ("2020-01-01", "2025-02-07")

    market = "all"
    # benchmark = "SZ16070"
    benchmark = "SH000300"
    # 获取test时段的行情原始数据
    stockpool: List = D.instruments(market=market)
    raw_data: pd.DataFrame = D.features(
        stockpool,
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=test_period[0],
        end_time=test_period[1],
    )

    raw_data.columns = [col.replace('$', '') for col in raw_data.columns]
    start_date = '2020-01-01'
    split_date = '2024-09-01'
    test = knn_stock_prediction(raw_data, start_date, split_date)
    test.to_csv("c:\\temp\\new_all_data_20250213.csv")


'''
##################################################################    
# 假设raw_data和weights_df_1是已经存在的数据框
merged_df = pd.merge(raw_data.reset_index(), weights_df_2, left_on=['instrument', 'datetime'], right_on=['code', 'date'], how='inner')

# 如果不需要合并过程中产生的临时列，可以删除
merged_df = merged_df.drop(['code', 'date','取值日期'], axis=1)
#merged_df = merged_df[['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume', '名称', 'weight','板块名称']]
# 可以将instrument和datetime设置回索引（如果需要）
merged_df = merged_df.set_index(['instrument', 'datetime'])
'''        








'''
    # 获取指数6月底跟12月底的权重数据，取数方式见“同花顺数据采集.py”
    #########################################################################
    #获取股票板块及总股本
    combined_df_new = pd.read_csv('c:\\temp\\combined_df_new_20250124.csv')
    combined_df_new = combined_df_new.drop(combined_df_new.columns[[0]], axis=1)
    weights_df = pd.read_csv("c:\\temp\\weights_df_20160616.csv")
    weights_df = weights_df.drop(weights_df.columns[[0]], axis=1)
    weights_df_1 = pd.read_csv("c:\\temp\\weights_df_1_20160617.csv")
    weights_df_1 = weights_df_1.drop(weights_df_1.columns[[0]], axis=1)

    new_weights_df = pd.concat([weights_df, weights_df_1], axis=0)

    new_weights_df.rename(columns={'p03563_f001': '取值日期',
                                   'p03563_f002': '代码',
                                   'p03563_f003': '名称',
                                   'p03563_f004': '权重'}, inplace=True)

    weights_df = new_weights_df
    weights_df['代码'] = weights_df['代码'].apply(lambda x: x[-2:] + x[:-3])
    # 假设 combined_df_new 和 weights_df 是你的数据框
    # 首先确保两个数据框的“代码”列都是字符串格式，以确保正确匹配
    combined_df_new['代码'] = combined_df_new['代码'].astype(str)
    weights_df['代码'] = weights_df['代码'].astype(str)

    # 根据“代码”列进行合并，使用左连接（left join）以保留 weights_df 中的所有行
    weights_df_1 = pd.merge(weights_df, combined_df_new[['代码', '板块名称']], on='代码', how='left')


    # 确保date列的格式与datetime列一致
    weights_df_1 = weights_df_1.rename(columns={
        "代码": "code",
        "权重": "weight"
    })
    weights_df_1['date'] = pd.to_datetime(weights_df_1['date'], format='%Y%m%d', errors='coerce')

    # 确保test中的datetime列也是datetime类型
    test['datetime'] = pd.to_datetime(test['datetime'])

    # 合并DataFrame
    merged_df = pd.merge(test, weights_df_1, left_on=['code', 'datetime'], right_on=['code', 'date'], how='left')

    # 只保留weight列有数据的行
    merged_df = merged_df.dropna(subset=['weight'])

    #######################################
    merged_df[merged_df['datetime'] == '2020-01-02']
    merged_df = merged_df[['code', 'datetime', 'close', 'Predicted_Signal','weight','板块名称']]
    merged_df = merged_df.rename(columns={
        "板块名称": "INDUSTRY_CODE",
        "Predicted_Signal":"rank"})
    #######################################

    # 使用 merge 函数将 combined_df_new 中的相关列合并到 ranked_data 中


    ranked_data_1 = merged_df.join(combined_df_new.set_index('代码'), on='code', how='left')

    ranked_data_1 = ranked_data_1.set_index(['code', 'datetime'])
    # ranked_data_2 = ranked_data_2.rename_axis(index={'datetime': 'date'})

    # 取字段
    ranked_data_2 = ranked_data_1[['close', 'rank', 'INDUSTRY_CODE', 'market_cap']]
    #
    ranked_data_2['NEXT_RET'] = ranked_data_2['close'].pct_change().shift(-1)
    #检测数据
    #ranked_data_2.loc[(ranked_data_2.index.get_level_values('datetime') == pd.Timestamp('2020-01-02'))]

    #只取规则里的数据

    ranked_data_3 = ranked_data_2.loc[(ranked_data_2.index.get_level_values('datetime') >= pd.Timestamp('2024-09-01'))]















#######################检测数据############################################
#test_set = test[test['Predicted_Signal'].isna()]
#grouped_counts = test_set.groupby('instrument').size().reset_index(name='count')


#test.xs('SH600764', level='instrument')['Predicted_Signal'].isna()
#[test.xs('SH600764', level='instrument')['Predicted_Signal'].isna()].reset_index()

#####


'''



