import os
import logging
from logging.handlers import TimedRotatingFileHandler
import colorlog

# 1、定义日志根目录（相对当前脚本所在目录）
LOG_ROOT = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)

# 2、日志文件名称，按日期划分
LOG_FILE = os.path.join(LOG_ROOT, "fastapi_run.log")

# 3、控制台彩色格式
console_formatter = colorlog.ColoredFormatter(
    fmt="%(log_color)s%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red,bg_white",
    },
)

# 4、文件日志普通格式（文件不需要颜色代码）
file_formatter = logging.Formatter(
    fmt="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 5、文件处理器：按天切割日志，保留 30 天日志记录
file_handler = TimedRotatingFileHandler(
    filename=LOG_FILE, when="D", interval=1, backupCount=30, encoding="utf-8"
)
file_handler.setFormatter(file_formatter)

# 6、控制台处理器
console_handler = colorlog.StreamHandler()
console_handler.setFormatter(console_formatter)


# 7、组装 logger
def get_logger(name="fastapi_log"):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # 避免重复添加 handler
    if not log.handlers:
        log.addHandler(file_handler)
        log.addHandler(console_handler)
    return log


# 全局实例化导出
logger = get_logger()
