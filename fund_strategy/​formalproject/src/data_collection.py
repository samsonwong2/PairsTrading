# -*- coding: utf-8 -*-
"""
基金数据采集与QLib格式转换工具
Created on 2025-03-31
@author: huangtuo
"""
import time
import random
import pandas as pd
import akshare as ak
import subprocess
import os

# 配置常量（根据实际路径修改）
CONFIG = {
    "codefundsecname_file": r'c:\temp\upload\codefundsecname.csv',
    "csv_output_dir": r'C:\Users\huangtuo\.qlib\qlib_data\fund_data\change_csv',
    "qlib_dir": r'C:\Users\huangtuo\.qlib\qlib_data\fund_data',
    "qlib_scripts_path": r'C:\qlib-main\scripts',
    "start_date": '20050101',
    "end_date": '20250331'
}


def fetch_and_save_data(config):
    """
    从akshare获取各类基金数据并存储为CSV格式
    参数说明：
        config - 包含路径配置的字典，需包含以下键：
            codefundsecname_file: 基金代码清单文件路径
            csv_output_dir: CSV输出目录
            start_date: 数据开始日期(YYYYMMDD)
            end_date: 数据结束日期(YYYYMMDD)
    """
    # 读取基金代码分类清单
    code_df = pd.read_csv(config['codefundsecname_file'])
    type_mapping = {
        'lof': ak.fund_lof_hist_em,
        'etf': ak.fund_etf_hist_em,
        'index': ak.stock_zh_index_daily_em
    }

    # 按类型遍历处理
    for fund_type, api_func in type_mapping.items():
        codes = code_df[code_df['type'] == fund_type]['code']
        print(f"正在处理{fund_type.upper()}基金，共{len(codes)}只")

        for symbol in codes:
            try:
                # 指数类特殊处理symbol格式
                symbol_param = symbol if fund_type == 'index' else symbol[2:]

                # 调用akshare接口获取数据
                df = api_func(
                    symbol=symbol_param,
                    period="daily",
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    adjust=""
                )

                # 空数据检查
                if df.empty:
                    raise ValueError(f"Empty data for {symbol}")

                # 数据标准化处理
                processed_df = (df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                                   '最高': 'high', '最低': 'low', '成交量': 'volume'})
                                .pipe(format_datetime)
                                .assign(code=symbol))

                # 保存CSV文件
                save_path = os.path.join(config['csv_output_dir'], f"{symbol}.csv")
                processed_df.to_csv(save_path, index=False)

            except Exception as e:
                print(f"[ERROR] {symbol}数据处理失败：{str(e)}")
            finally:
                time.sleep(random.randint(1, 10))  # 随机延迟防止反爬


def format_datetime(df):
    """统一日期格式处理"""
    return (df.assign(date=pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d'))
            .astype({'date': 'datetime64[ns]'}))


def dump_to_qlib(config):
    """
    将CSV数据转换为QLib二进制格式
    参数说明：
        config - 包含路径配置的字典，需包含：
            qlib_scripts_path: dump_bin.py脚本路径
            csv_output_dir: CSV文件目录
            qlib_dir: QLib数据存储目录
    """
    if not all(os.path.exists(p) for p in [config['qlib_scripts_path'],
                                           config['csv_output_dir'],
                                           config['qlib_dir']]):
        raise FileNotFoundError("关键路径配置错误，请检查config设置")

    command = [
        'python', f"{config['qlib_scripts_path']}/dump_bin.py", 'dump_all',
        '--csv_path', config['csv_output_dir'],
        '--qlib_dir', config['qlib_dir'],
        '--symbol_field_name', 'code',
        '--date_field_name', 'date',
        '--include_fields', 'open,high,low,close,volume'
    ]

    try:
        result = subprocess.run(command, check=True, text=True,
                                capture_output=True, encoding='utf-8')
        print("QLib数据转换成功！详细信息：\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"转换失败：{e.stderr}")
        raise


if __name__ == '__main__':
    # 第一阶段：数据采集
    fetch_and_save_data(CONFIG)

    # 第二阶段：QLib格式转换
    dump_to_qlib(CONFIG)