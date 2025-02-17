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


def factors_null_process(data: pd.DataFrame, columns_to_process=None) -> pd.DataFrame:
    # 删除行业缺失值
    data = data[data['INDUSTRY_CODE'].notnull()]

    # 变化索引，以行业为第一索引，股票代码为第二索引
    data_ = data.reset_index().set_index(['INDUSTRY_CODE', 'code']).sort_index()

    if columns_to_process is None:
        # 如果未指定处理的列，默认选择数值类型的列
        numeric_columns = data_.select_dtypes(include='number').columns
    else:
        # 筛选出指定要处理的列
        numeric_columns = [col for col in columns_to_process if col in data_.columns]

    # 用行业中位数填充数值列的缺失值
    def fillna_median(group):
        group[numeric_columns] = group[numeric_columns].fillna(group[numeric_columns].median())
        return group

    data_ = data_.groupby(level=0).apply(fillna_median)

    # 有些行业可能只有一两个个股却都为nan此时使用0值填充数值列
    data_[numeric_columns] = data_[numeric_columns].fillna(0)

    # 检查 INDUSTRY_CODE 是否已经是普通列
    if 'INDUSTRY_CODE' in data_.columns:
        # 如果已经是普通列，只对 'code' 索引进行 reset_index
        data_ = data_.reset_index(level=data_.index.names.index('code'))
    else:
        # 否则，对所有索引进行 reset_index，但不插入重复列
        index_levels = data_.index.names
        levels_to_reset = [data_.index.names.index(level) for level in index_levels if level not in data_.columns]
        data_ = data_.reset_index(level=levels_to_reset)

    # 设置 'code' 为索引并排序
    data_ = data_.set_index('code').sort_index()

    return data_.drop('date', axis=1)


def get_group(ser: pd.Series, N: int = 3, ascend: bool = True) -> pd.Series:
    '''默认分三组 升序'''
    ranks = ser.rank(ascending=ascend)
    label = ['G' + str(i) for i in range(1, N + 1)]

    return pd.cut(ranks, bins=N, labels=label)

if __name__ == '__main__':

######################step1 获取原始量价数据 ########################

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
#####################step2 权重，板块数据 ###########################
    # 获取股票板块及总股本
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
    weights_df_1 = pd.merge(weights_df, combined_df_new[['代码', '板块名称','总股本']], on='代码', how='left')

    # 确保date列的格式与datetime列一致
    weights_df_1 = weights_df_1.rename(columns={
        "代码": "code",
        "权重": "weight"
    })
    weights_df_1['date'] = pd.to_datetime(weights_df_1['date'], format='%Y%m%d', errors='coerce')
    #限定一下获取日期
    weights_df_2=weights_df_1[weights_df_1['date'] >= '2020-01-01']

#####################step3 限定原始量价数据 ###########################
    # 假设raw_data和weights_df_1是已经存在的数据框
    merged_df = pd.merge(raw_data.reset_index(), weights_df_2, left_on=['instrument', 'datetime'],
                         right_on=['code', 'date'], how='inner')

    # 如果不需要合并过程中产生的临时列，可以删除
    merged_df = merged_df.drop(['code', 'date', '取值日期'], axis=1)
    # merged_df = merged_df[['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume', '名称', 'weight','板块名称','总股本']]
    # 可以将instrument和datetime设置回索引（如果需要）
    merged_df = merged_df.set_index(['instrument', 'datetime'])

####################step4 限定原始量价进行计算 ############################
    start_date = '2020-01-01'
    split_date = '2024-09-01'
    test = knn_stock_prediction(merged_df, start_date, split_date)

    # 确保 df 的索引转换为列，方便合并操作
    merged_df_1 = merged_df.reset_index()

    # 合并数据
    merged_df_2 = pd.merge(merged_df_1, test, on=['instrument', 'datetime'], how='left')

    # 如果需要，可以将索引恢复原状
    merged_df_2 = merged_df_2.set_index(['instrument', 'datetime'])

    merged_df_2['market_cap']=merged_df_2['$close']*merged_df_2['总股本']
    merged_df_2 = merged_df_2.rename_axis(index={'instrument': 'code'})
    merged_df_2 = merged_df_2.rename_axis(index={'datetime': 'date'})
    merged_df_2 = merged_df_2.rename(columns={'板块名称': 'INDUSTRY_CODE'})
    merged_df_2 = merged_df_2.drop(columns=['名称', '总股本'])

    # 假设 merged_df_2 已经定义
    # 指定要处理的列
    columns_to_process = ['market_cap']
    merged_df_3 = merged_df_2.groupby(level='date').apply(lambda x: factors_null_process(x, columns_to_process))

    # 市值等量分三组
    k1 = [pd.Grouper(level='date'),
        pd.Grouper(key='INDUSTRY_CODE')]

    merged_df_3['GROUP'] = merged_df_3.groupby(
        k1)['market_cap'].apply(lambda x: get_group(x, 3))

############################################################################
import pandas as pd

# 假设 merged_df_3 已经定义并包含 'date', 'INDUSTRY_CODE', 'market_cap' 列

def get_group(ser: pd.Series, N: int = 3, ascend: bool = True) -> pd.Series:
    '''默认分三组 升序'''
    ranks = ser.rank(method='first', ascending=ascend)  # 使用 'first' 以确保排名唯一
    labels = ['G' + str(i) for i in range(1, N + 1)]
    return pd.cut(ranks, bins=N, labels=labels, include_lowest=True)

# 使用 transform 应用分组和分类
merged_df_3['GROUP'] = merged_df_3.groupby(['date', 'INDUSTRY_CODE'])['market_cap'].transform(
    lambda x: get_group(x, 3)
)

# 现在 merged_df_3 包含了一个新的 'GROUP' 列，表示每个 (date, INDUSTRY_CODE) 组合下的市值分组

merged_df_3.xs('2020-01-02', level='date').head()
#merged_df_3=pd.csv(“c:\\temp\\merged_df_3_20250214.csv")
#merged_df_3 = merged_df_3.set_index(['date','code'])

merged_df_3['Predicted_Signal'] = merged_df_3['Predicted_Signal'].fillna(0)
merged_df_3 = merged_df_3.rename(columns={'Predicted_Signal': 'SCORE'})
# 获取每组得分最大的
k2 = [pd.Grouper(level='date'),
      pd.Grouper(key='INDUSTRY_CODE'),
      pd.Grouper(key='GROUP')]

industry_kfold_stock = merged_df_3.groupby(
    k2)['SCORE'].apply(lambda x: x.idxmax()[1])

# 格式调整
industry_kfold_stock = industry_kfold_stock.reset_index()
industry_kfold_stock = industry_kfold_stock.set_index(['date', 'SCORE'])
industry_kfold_stock.index.names = ['date', 'code']

# 加入权重
industry_kfold_stock['weight'] = merged_df_3['weight']
merged_df_3['NEXT_RET'] = merged_df_3['$close'].pct_change().shift(-1)
industry_kfold_stock['NEXT_RET'] = merged_df_3['NEXT_RET']
# 储存,用于回测
#industry_kfold_stock.to_csv("c:\\temp\\industry_kfold_stock_20250217.csv")









