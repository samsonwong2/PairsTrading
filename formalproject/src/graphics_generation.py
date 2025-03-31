# -*- coding: utf-8 -*-
"""
基金数据模式分析与自动报告生成系统
功能：
1. 加载QLib金融数据
2. 执行技术模式识别
3. 生成可视化图表
4. 自动邮件发送报告
创建日期：2025-03-31
作者：huangtuo
"""
import sys
import os
import datetime
import qlib
import pandas as pd
import matplotlib.pyplot as plt
from qlib.data import D
from qlib.constant import REG_CN
from send_mail_tool import send_mail_tool
from technical_analysis_patterns import (
    rolling_patterns,
    plot_patterns_chart,
    plot_patterns_chart1
)

# 系统路径配置
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
qlib.init(provider_uri="C:/Users/huangtuo/.qlib/qlib_data/fund_data/", region=REG_CN)

# 常量配置
REPORT_RECIPIENTS = ["tianfangfang1105@126.com", "huangtuo02@163.com"]
REPORT_SAVE_PATH = "C://temp//upload//"


def create_directory(path: str) -> bool:
    """
   创建指定目录（带路径规范化处理）
   参数：
       path - 目录路径字符串
   返回：
       True(创建成功)/False(已存在)
   """
    clean_path = os.path.normpath(path.strip())
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
        print(f"目录创建成功：{clean_path}")
        return True
    print(f"目录已存在：{clean_path}")
    return False


def generate_technical_analysis(code: str, name: str, days: int = 60) -> None:
    """
   执行技术模式分析并生成可视化报告
   参数：
       code - 基金代码
       name - 基金名称
       days - 分析周期（默认60天）
   """
    # 日期范围计算
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    date_str = end_date.strftime('%Y%m%d')

    # 数据获取与处理
    df = D.features([code],
                    fields=["$open", "$close", "$low", "$high"],
                    start_time=start_date.strftime('%Y%m%d'),
                    end_time=end_date.strftime('%Y%m%d')
                    ).reset_index(level=0, drop=True)
    df.columns = df.columns.str.replace('$', '')

    # 数据预处理
    analysis_data = df[-40:].copy()
    analysis_data.index = pd.to_datetime(analysis_data.index)
    analysis_data = analysis_data.astype('float64')

    try:
        # 添加预测数据点
        last_row = analysis_data.iloc[-1]
        analysis_data.loc[analysis_data.index[-1] + datetime.timedelta(days=1)] = last_row * 1.001
    except IndexError as e:
        print(f"数据索引异常：{str(e)}")
        return

    # 模式识别与可视化
    patterns = rolling_patterns(analysis_data['close'], window=12)
    report_path = os.path.join(REPORT_SAVE_PATH, f"{date_str}_pattern_graph")
    create_directory(report_path)

    # 生成双视图图表
    for view_type in ['overview', 'detail']:
        fig_path = os.path.join(report_path, f"{date_str}_{name}_{view_type}.jpg")
        plot_patterns_chart(
            analysis_data,
            patterns,
            show_labels=True,
            show_patterns=(view_type == 'detail'),
            title=name,
            save_path=fig_path
        )
        plt.close()


if __name__ == '__main__':
    # 加载基金代码清单
    fund_codes = pd.read_csv('c:\\temp\\upload\\codefundsecname.csv')

    # 批量处理所有基金
    for _, row in fund_codes.iterrows():
        generate_technical_analysis(
            code=row['code'].strip(),
            name=row['name'].strip()
        )

    # 发送汇总报告
    report_date = datetime.datetime.now().strftime('%Y%m%d')
    send_mail_tool(
        recipients=REPORT_RECIPIENTS,
        report_path=os.path.join(REPORT_SAVE_PATH, f"{report_date}_pattern_graph"),
        fund_list=fund_codes['code'].tolist()
    ).action_send()