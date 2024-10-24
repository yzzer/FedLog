import logging


def init_log_config():
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        datefmt='%Y-%m-%d %H:%M:%S',  # 设置时间格式
        handlers=[
            logging.StreamHandler()  # 输出到控制台
        ]
    )