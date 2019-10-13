from logging import Formatter, handlers, StreamHandler, getLogger

class Logger:
    #定数宣言
    LOG_SET_LEVEL = 'DEBUG'         #対象とするログレベル（ファイルにも使用する）
    STREAM_LOG_SET_LEVEL = 'INFO'   #標準出力で表示するログレベル

    def __init__(self, name=__name__):
        self.logger = getLogger(name)
        self.logger.setLevel(self.LOG_SET_LEVEL)
        formatter = Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

        # stdout
        handler = StreamHandler()
        handler.setLevel(self.STREAM_LOG_SET_LEVEL)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # file
        handler = handlers.RotatingFileHandler(filename = 'logfile/logger.log')
        handler.setLevel(self.LOG_SET_LEVEL)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)