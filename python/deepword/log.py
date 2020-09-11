import logging
import logging.config
from typing import Optional


class Logging(object):
    """
    Logging utils for classes
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: name for logging, default module_name.class_name
        """

        if name is None:
            cls_name = self.__class__.__name__
            module_name = self.__module__
            name = '{}.{}'.format(module_name, cls_name)
        self.__logger = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        self.__logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.__logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.__logger.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.__logger.error(msg, *args, **kwargs)
