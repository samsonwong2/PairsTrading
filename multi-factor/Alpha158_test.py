import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/cn_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data import D
from typing import List, Tuple, Dict
from qlib.contrib.data.handler import Alpha158

test_period = ("2024-01-01", "2025-02-07")
market = "all"
benchmark = "SH000300"

dh = Alpha158(instruments='csi300',
              start_time=test_period[0],
              end_time=test_period[1],
              infer_processors={}
              )

fetch_factor = dh.fetch()

#######################取权重及行业数据##############################
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
weights_df_1 = pd.merge(weights_df, combined_df_new[['代码', '板块名称', '总股本']], on='代码', how='left')

# 确保date列的格式与datetime列一致
weights_df_1 = weights_df_1.rename(columns={
    "代码": "code",
    "权重": "weight"
})
weights_df_1['date'] = pd.to_datetime(weights_df_1['date'], format='%Y%m%d', errors='coerce')
# 限定一下获取日期
weights_df_2 = weights_df_1[weights_df_1['date'] >= '2020-01-01']
######################
##########合并数据#########
merged_df = pd.merge(fetch_factor.reset_index(), weights_df_2, left_on=['instrument', 'datetime'],
                     right_on=['code', 'date'], how='inner')

# 如果不需要合并过程中产生的临时列，可以删除
merged_df = merged_df.drop(['code', 'date', '取值日期'], axis=1)
# merged_df = merged_df[['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume', '名称', 'weight','板块名称','总股本']]
# 可以将instrument和datetime设置回索引（如果需要）
merged_df = merged_df.set_index(['instrument', 'datetime'])

merged_df = merged_df.rename(columns={'板块名称': 'INDUSTRY_CODE','总股本':'market_cap'})
merged_df = merged_df.rename_axis(index={'instrument': 'code'})
merged_df = merged_df.rename_axis(index={'datetime': 'date'})
merged_df = merged_df.drop(columns=['名称'])
factors = merged_df.drop(columns=['weight'])


def factors_null_process(data: pd.DataFrame) -> pd.DataFrame:
    # 删除行业缺失值
    data = data[data["INDUSTRY_CODE"].notnull()].copy()

    # 确保 code 存在于列中（如果 code 是索引层级）
    if "code" not in data.columns:
        data = data.reset_index(level="code")

    # 使用唯一临时名称重命名关键列
    temp_col = "TEMP_INDUSTRY_123"  # 唯一临时名称
    data = data.rename(columns={"INDUSTRY_CODE": temp_col})

    # 设置复合索引
    data_ = data.set_index([temp_col, "code"]).sort_index()

    # 行业中性化填充（使用行业中位数）
    data_ = data_.groupby(level=0).apply(
        lambda x: x.fillna(x.median(numeric_only=True))
    ).fillna(0)

    # 重置索引并删除临时列
    data_ = data_.reset_index(drop=False)
    # 恢复原始列名并删除临时列
    data_ = data_.rename(columns={temp_col: "INDUSTRY_CODE"})
    # 确保没有残留的临时列
    if temp_col in data_.columns:
        data_ = data_.drop(temp_col, axis=1)

    # 重新设置股票代码为索引
    data_ = data_.set_index("code").sort_index()

    # 删除冗余列（确保 date 列存在）
    if "date" in data_.columns:
        data_ = data_.drop("date", axis=1)

    return data_

# step2:构建绝对中位数处理法函数
def extreme_process_MAD(data: pd.DataFrame, num: int = 3) -> pd.DataFrame:
    ''' data为输入的数据集，如果数值超过num个判断标准则使其等于num个标准'''

    # 为不破坏原始数据，先对其进行拷贝
    data_ = data.copy()

    # 获取数据集中需测试的因子名
    feature_names = [i for i in data_.columns.tolist() if i not in [
        'INDUSTRY_CODE','market_cap']]

    # 获取中位数
    median = data_[feature_names].median(axis=0)
    # 按列索引匹配，并在行中广播
    MAD = abs(data_[feature_names].sub(median, axis=1)
              ).median(axis=0)
    # 利用clip()函数，将因子取值限定在上下限范围内，即用上下限来代替异常值
    data_.loc[:, feature_names] = data_.loc[:, feature_names].clip(
        lower=median-num * 1.4826 * MAD, upper=median + num * 1.4826 * MAD, axis=1)
    return data_

##step3:构建标准化处理函数
def data_scale_Z_Score(data: pd.DataFrame) -> pd.DataFrame:

    # 为不破坏原始数据，先对其进行拷贝
    data_ = data.copy()
    # 获取数据集中需测试的因子名
    feature_names = [i for i in data_.columns.tolist() if i not in [
        'INDUSTRY_CODE','market_cap']]
    data_.loc[:, feature_names] = (
        data_.loc[:, feature_names] - data_.loc[:, feature_names].mean()) / data_.loc[:, feature_names].std()
    return data_


# step4:因子中性化处理函数
def neutralization(data: pd.DataFrame) -> pd.DataFrame:
    '''按市值、行业进行中性化处理 ps:处理后无行业市值信息'''

    factor_name = [i for i in data.columns.tolist() if i not in [
        'INDUSTRY_CODE', 'market_cap']]

    # 回归取残差
    def _calc_resid(x: pd.DataFrame, y: pd.Series) -> float:
        result = sm.OLS(y, x).fit()

        return result.resid

    X = pd.get_dummies(data['INDUSTRY_CODE'])
    # 总市值单位为亿元
    X['market_cap'] = np.log(data['market_cap'] * 100000000)

    df = pd.concat([_calc_resid(X.fillna(0), data[i])
                    for i in factor_name], axis=1)

    df.columns = factor_name

    df['INDUSTRY_CODE'] = data['INDUSTRY_CODE']
    df['market_cap'] = data['market_cap']

    return df

## 其实可以直接pipe处理但是这里为了后续灵活性没有选择pipe化

# 去极值
factors1 = factors.groupby(level='date').apply(extreme_process_MAD)
factors1 = factors1.droplevel(2)
# 缺失值处理
#factors2 = factors1.groupby(level='date').apply(factors_null_process)
# 中性化
#factors3 = factors2.groupby(level='date').apply(neutralization)
# 标准化
factors4 = factors1.groupby(level='date').apply(data_scale_Z_Score)
factors4 = factors4.droplevel(0)


def lowdin_orthogonal(data):
    data_ = data.copy()
    cols = [col for col in data_.columns if col not in ['INDUSTRY_CODE', 'market_cap']]

    # 1. 检查 NaN/Inf
    if data_[cols].isnull().values.any() or np.isinf(data_[cols].values).any():
        # 可以选择填充或删除问题数据
        data_ = data_.dropna(subset=cols)
        data_ = data_.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    # 2. 提取数值矩阵
    F = data_[cols].values

    # 3. 计算协方差矩阵并添加正则化项
    M = F.T @ F
    M += 1e-8 * np.eye(M.shape[0])  # 正则化

    # 4. 特征值分解（使用 eigh 确保稳定性，因为 M 是实对称矩阵）
    a, U = np.linalg.eigh(M)

    # 5. 处理非正特征值
    a = np.maximum(a, 1e-10)  # 避免负值或零

    # 6. 计算 Lowdin 变换
    a_inv_sqrt = np.diagflat(1 / np.sqrt(a))
    S = U @ a_inv_sqrt @ U.T
    F_ortho = F @ S

    # 7. 更新数据
    data_.loc[:, cols] = F_ortho
    return data_

factors5 = factors4.groupby(level='date').apply(lowdin_orthogonal)
factors5.info()


# 构建计算横截面因子载荷相关系数均值函数
def get_relations(datas: pd.DataFrame) -> pd.DataFrame:
    relations = 0
    for trade, d in datas.groupby(level='date'):
        relations += d.corr()

    relations_mean = relations / len(datas.index.levels[0])

    return relations_mean

import matplotlib.pyplot as plt
import seaborn as sns
# 绘制因子正交前的相关性的热力图
fig = plt.figure(figsize=(26, 18))
# 计算对称正交之前的相关系数矩阵
relations = get_relations(factors4.iloc[:,:-2])
sns.heatmap(relations, annot=True, linewidths=0.05,
            linecolor='white', annot_kws={'size': 8, 'weight': 'bold'})

#绘制因子正交后的相关性热力图
fig=plt.figure(figsize=(26,18))
#计算对称正交之后的相关系数矩阵
relations= get_relations(factors5.iloc[:,:-2])
sns.heatmap(relations,annot=True,linewidths=0.05,
            linecolor='white',annot_kws={'size':8,'weight':'bold'})


#############################################################################
#############################################################################
market = "csi300"
# benchmark = "SZ16070"
benchmark = "SH000300"
# 获取test时段的行情原始数据
stockpool: List = D.instruments(market=market)
raw_data_3: pd.DataFrame = D.features(
    stockpool,
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time=test_period[0],
    end_time=test_period[1],
)

raw_data_3['NEXT_RET'] = raw_data_3['$close'].pct_change().shift(-1)


#factors5 = factors5.set_index(['date', 'code'])

raw_next_ret = raw_data_3[['NEXT_RET','$close']].reset_index()
# 将 factors5 和 raw_next_ret 的索引转为普通列
factors5_reset = factors5.reset_index()
raw_next_ret_reset = raw_next_ret.reset_index()
factors5_reset['date'] = pd.to_datetime(factors5_reset['date'], format='%Y-%m-%d')
raw_next_ret['datetime'] = pd.to_datetime(raw_next_ret['datetime'], format='%Y-%m-%d')
# 合并，根据 instrument/code 和 datetime/date 对齐
merged = pd.merge(
    factors5_reset,                   # 左表（包含 instrument, datetime 列）
    raw_next_ret,                    # 右表（已重命名为 date, code 列）
    left_on=['code', 'date'],  # 左表合并键
    right_on=['instrument', 'datetime'],          # 右表合并键
    how='left'                        # 或根据需求选择 'inner', 'outer'
)
merged
merged = merged.drop(['instrument', 'datetime'], axis=1)
merged = merged.set_index(['date', 'code'])
merged['market_cap']=merged['$close']*merged['market_cap']
merged = merged.drop(['$close'], axis=1)
factors5=merged


# 根据IR计算因子权重

# step1:计算rank_IC


def calc_rank_IC(factor: pd.DataFrame) -> pd.DataFrame:
    factor_col = [x for x in factor.columns if x not in [
        'INDUSTRY_CODE', 'market_cap', 'NEXT_RET']]

    IC = factor.groupby(level='date').apply(lambda x: [st.spearmanr(
        x[factor], x['NEXT_RET'])[0] for factor in factor_col])

    return pd.DataFrame(IC.tolist(), index=IC.index, columns=factor_col)


## step2: 计算IR权重
def IR_weight(factor: pd.DataFrame) -> pd.DataFrame:
    data_ = factor.copy()
    # 计算ic值，得到ic的
    IC = calc_rank_IC(data_)

    # 计算ic的绝对值
    abs_IC = IC.abs()
    # rolling为移动窗口函数,滚动12个月
    rolling_ic = abs_IC.rolling(12, min_periods=1).mean()
    # 当滚动计算标准差时，起始日期得到的是缺失值，所以算完权重后，起始日期的值任用原值IC代替
    rolling_ic_std = abs_IC.rolling(12, min_periods=1).std()
    IR = rolling_ic / rolling_ic_std  # 计算IR值
    IR.iloc[0, :] = rolling_ic.iloc[0, :]
    weight = IR.div(IR.sum(axis=1), axis=0)  # 计算IR权重,按行求和,按列相除

    return weight

# 获取权重
import scipy.stats as st
weights = IR_weight(factors5)

# 获取因子名称
factor_names = [name for name in factors5.columns if name not in [
    'INDUSTRY_CODE', 'market_cap', 'NEXT_RET']]

# 计算因子分数
factors5['SCORE'] = (factors5[factor_names].mul(weights)).sum(axis=1)

ranked_data_2 = factors5
# 按 'date' 分组，并使用 'market_cap' 的中位数进行回填
ranked_data_2['market_cap'] = ranked_data_2.groupby('date')['market_cap'].transform(lambda x: x.fillna(x.median()))

# 市值等量分三组
k1 = [pd.Grouper(level='date'),
      pd.Grouper(key='INDUSTRY_CODE')]

ranked_data_2['GROUP'] = ranked_data_2.groupby(
    k1)['market_cap'].apply(lambda x: get_group(x, 3))


def get_group(ser: pd.Series, N: int = 3, ascend: bool = True) -> pd.Series:
    '''默认分三组 升序'''
    ranks = ser.rank(method='first', ascending=ascend)  # 使用 'first' 以确保排名唯一
    labels = ['G' + str(i) for i in range(1, N + 1)]
    return pd.cut(ranks, bins=N, labels=labels, include_lowest=True)

# 使用 transform 应用分组和分类
ranked_data_2['GROUP'] = ranked_data_2.groupby(['date', 'INDUSTRY_CODE'])['market_cap'].transform(
    lambda x: get_group(x, 3)
)

#ranked_data_2.xs('2020-01-02', level='date').head()
ranked_data_2['SCORE'] = ranked_data_2['SCORE'].fillna(0)

# 获取每组得分最大的
k2 = [pd.Grouper(level='date'),
      pd.Grouper(key='INDUSTRY_CODE'),
      pd.Grouper(key='GROUP')]

industry_kfold_stock = ranked_data_2.groupby(
    k2)['SCORE'].apply(lambda x: x.idxmax()[1])

# 格式调整
industry_kfold_stock = industry_kfold_stock.reset_index()
industry_kfold_stock = industry_kfold_stock.set_index(['date', 'SCORE'])
industry_kfold_stock.index.names = ['date', 'code']

test1=weights_df_2[weights_df_2['date'] >= '2024-01-01']
merged_df_3=test1.set_index(['date', 'code'])

# 加入权重
industry_kfold_stock['weight'] = merged_df_3['weight']
industry_kfold_stock['NEXT_RET'] = ranked_data_2['NEXT_RET']

industry_kfold_stock = industry_kfold_stock.dropna()
industry_kfold_stock = industry_kfold_stock.drop(['index'], axis=1)


###########################
columns_to_add = ['$open', '$high', '$low', '$close', '$volume']
merged_df_subset = raw_data_3[columns_to_add]
merged_df_subset.columns = [col.replace('$', '') for col in merged_df_subset.columns]
# 通过索引合并（假设两个 DataFrame 均已正确设置 date 和 code 的多级索引）
new_industry_kfold_stock = industry_kfold_stock.join(merged_df_subset, how='left')
columns_to_add = ['open', 'high', 'low', 'close', 'volume','w']
new_industry_kfold_stock = new_industry_kfold_stock[columns_to_add]

new_industry_kfold_stock = new_industry_kfold_stock.droplevel(2)






