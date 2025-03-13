import logging
import time
import pandas as pd
import akshare as ak
import random
from logging.handlers import RotatingFileHandler
import os


# 初始化日志配置
def setup_logger():


    """配置日志系统（含滚动日志功能）"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置根日志级别

    # 创建格式化器（网页1/网页4推荐格式）
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器（INFO级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 创建日志目录（新增代码）
    log_dir = r'C:\temp\log\fund_crawler'
    os.makedirs(log_dir, exist_ok=True)

    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # 文件处理器（DEBUG级别，滚动日志）
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'fund_crawler.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()


def fetch_and_save_data(dir_name, start_date, end_date):
    """带日志记录的基金数据抓取"""
    try:
        logger.info("开始获取基金列表")
        codefundsecname = ak.fund_exchange_rank_em()

        max_date = codefundsecname['日期'].max()
        logger.debug(f"获取到最新基金数据日期：{max_date}")

        # 保存基金列表（网页6的文件操作日志）
        file_name = f'fund_list_{max_date.strftime("%Y%m%d")}.csv'
        codefundsecname.to_csv(file_name, index=False)
        logger.info(f"基金列表已保存至：{file_name}")

        total_count = len(codefundsecname)
        success_count = 0
        failed_codes = []

        for idx, (_, code) in enumerate(codefundsecname['基金代码'].items(), 1):
            try:
                logger.debug(f"开始处理基金 {code} ({idx}/{total_count})")

                # 数据抓取
                fund_df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=""
                )

                if fund_df.empty:
                    logger.warning(f"基金 {code} 无有效数据")
                    continue

                # 数据清洗
                fund_df = fund_df.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume',
                    '成交额': 'turnover', '涨跌幅': 'change_pct', '换手率': 'turnover_rate'
                })
                fund_df["date"] = pd.to_datetime(fund_df['date']).dt.strftime('%Y-%m-%d')
                fund_df["code"] = code

                # 保存数据
                file_path = f"{dir_name}/{code}.csv"
                fund_df.to_csv(file_path, index=False)
                logger.debug(f"基金 {code} 数据已保存至 {file_path}")
                success_count += 1

            except Exception as e:
                logger.error(f"处理基金 {code} 失败", exc_info=True)
                failed_codes.append(code)
            finally:
                delay = random.randint(1, 10)
                logger.debug(f"等待 {delay} 秒后继续")
                time.sleep(delay)

        # 汇总日志（网页2的断言结合实践）
        logger.info(f"任务完成：成功 {success_count}/{total_count}")
        if failed_codes:
            logger.warning(f"失败基金代码列表：{failed_codes}")

    except Exception as e:
        logger.critical("主程序异常终止", exc_info=True)
        raise


if __name__ == '__main__':
    try:
        logger.info("程序启动")
        start_date = '20050101'
        end_date = '20250312'
        csv_path = r'C:\Users\huangtuo\.qlib\qlib_data\all_fund_data\change_csv'

        fetch_and_save_data(csv_path, start_date, end_date)
        logger.info("程序正常退出")

    except Exception as e:
        logger.exception("未捕获的全局异常")