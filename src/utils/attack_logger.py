"""
Utility for logging
Reference: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
"""

import logging
from colorlog import ColoredFormatter


class AttackLogger:
    logger = None

    @staticmethod
    def get_logger():
        if AttackLogger.logger is None:
            LOG_LEVEL = logging.DEBUG
            LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
            logging.root.setLevel(LOG_LEVEL)
            formatter = ColoredFormatter(LOG_FORMAT)
            stream = logging.StreamHandler()
            stream.setLevel(LOG_LEVEL)
            stream.setFormatter(formatter)

            AttackLogger.logger = logging.getLogger('pythonConfig')
            AttackLogger.logger.setLevel(LOG_LEVEL)
            AttackLogger.logger.addHandler(stream)

            AttackLogger.logger.info('Logger initialized!')

            # AttackLogger.logger.debug("A quirky message only developers care about")
            # AttackLogger.logger.info("Curious users might want to know this")
            # AttackLogger.logger.warning("Something is wrong and any user should be informed")
            # AttackLogger.logger.error("Serious stuff, this is red for a reason")
            # AttackLogger.logger.critical("OH NO everything is on fire")

        return AttackLogger.logger
