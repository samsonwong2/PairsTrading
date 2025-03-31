#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
邮件自动化发送工具
功能：支持多附件发送、异常处理、配置文件管理
版本：2.0 (2025-03-31)
"""
import json
import smtplib
import warnings
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 禁用无关警告
warnings.filterwarnings('ignore')


class EmailSender:
    """邮件发送核心类"""

    def __init__(self, config_path='C://config//config.json'):
        """
        初始化邮件发送器
        参数：
            config_path: 配置文件路径（包含SMTP服务器和认证信息）
        """
        self._load_config(config_path)

    def _load_config(self, path):
        """加载邮箱配置信息"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.sender = config.get('_user')  # 发件人邮箱
                self.password = config.get('qq')  # SMTP授权码
                self.smtp_server = "smtp.qq.com"  # QQ邮箱服务器
                self.smtp_port = 465  # SSL加密端口
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"配置加载失败: {str(e)}")

    def send_email(self, receivers, subject, content, attachments):
        """
        执行邮件发送
        参数：
            receivers  : 收件人列表（支持多个收件人）
            subject    : 邮件主题
            content    : 正文内容
            attachments: 附件路径列表（示例：['file1.png', 'report.pdf']）
        """
        # 创建邮件主体
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = ','.join(receivers)  # 多收件人处理

        # 添加文本内容（统一使用UTF-8编码）
        msg.attach(MIMEText(content, 'plain', 'utf-8'))

        # 添加附件处理
        for file_path in attachments:
            self._add_attachment(msg, file_path)

        # 建立SMTP连接并发送
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender, self.password)
                server.sendmail(self.sender, receivers, msg.as_string())
            print("邮件发送成功")
        except smtplib.SMTPException as e:
            print(f"SMTP服务异常: {str(e)}")

    def _add_attachment(self, msg, file_path):
        """添加单个附件（内部方法）"""
        try:
            with open(file_path, 'rb') as f:
                attachment = MIMEText(f.read(), 'base64', 'utf-8')
                attachment.add_header('Content-Type', 'application/octet-stream')
                attachment.add_header('Content-Disposition',
                                      'attachment',
                                      filename=f.name.split('/')[-1])
                msg.attach(attachment)
        except FileNotFoundError:
            print(f"警告：附件 {file_path} 不存在，已跳过")


if __name__ == '__main__':
    # ----------------------
    # 使用示例
    # ----------------------
    try:
        # 初始化邮件发送器
        mailer = EmailSender()

        # 配置邮件参数
        receivers = [""]
        attachments = [
            'C://temp//upload//Figure_1.png',
            'C://temp//upload//Figure_2.png'
        ]

        # 发送邮件
        mailer.send_email(
            receivers=receivers,
            subject="基金数据分析报告",
            content="附件包含最新分析图表，请查收。\n\n系统自动发送，请勿直接回复。",
            attachments=attachments
        )
    except Exception as e:
        print(f"邮件发送失败: {str(e)}")