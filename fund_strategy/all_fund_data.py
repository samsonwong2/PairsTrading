import akshare as ak
import pandas as pd


fund_name_em_df = ak.fund_name_em()


#fund_exchange_rank_em_df.groupby("类型").size().reset_index(name="基金数量")
# 假设已加载 DataFrame 数据 fund_exchange_rank_em_df
# 筛选深交所（15/16/18 开头）
szse_funds = fund_name_em_df[
    fund_name_em_df["基金代码"].astype(str).str.startswith(("15", "16", "18"))
]

# 筛选上交所（50/51/52/56/58 开头）  ###501为lof的基金,A，C份额都有
sse_funds = fund_name_em_df[
    fund_name_em_df["基金代码"].astype(str).str.startswith(("508","510","511","512","513","515","516","517", "518", "52", "56", "588"))
]


all_funds = fund_name_em_df[
    fund_name_em_df["基金类型"].astype(str).str.
    startswith(("FOF-均衡型", "FOF-稳健型", "FOF-进取型","QDII-FOF","QDII-REITs","QDII-商品","QDII-普通股票","QDII-混合债",
                "QDII-混合偏股","QDII-混合平衡","QDII-混合灵活","QDII-纯债","REITs","Reits","指数型-其他","指数型-固收",
                "指数型-海外股票","指数型-股票"))
    &fund_name_em_df["基金代码"].astype(str).str.startswith(("15", "16", "18","50", "51", "52","53","56", "58"))
]

#场内交易基金排行榜
fund_exchange_rank_em_df = ak.fund_exchange_rank_em()


# 合并数据框
merged = fund_exchange_rank_em_df.merge(
    all_funds,
    on="基金代码",
    how="outer",
    indicator=True
)

# 筛选 `fund_exchange_rank_em_df` 独有基金（左侧独有）
unique_in_fer = merged[merged['_merge'] == 'left_only']

