########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os
import sys
import time


def logger_init_basic():
    # logging.basicConfig(level=logging.INFO, format="[%(asctime)s - %(levelname)1.1s] %(message)s")

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s - %(levelname)1.1s] %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)


def logger_init(log_path, model_type, feature_type, access_type):
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s - %(levelname)1.1s] %(message)s")

    filename = time.strftime('%Y%m%d_%H%M%S',
                             time.localtime(
                                 time.time())) + '_' + model_type + '_' + feature_type + '_' + access_type + '_log.txt'
    log_filename = os.path.join(log_path, filename)
    fh = logging.FileHandler(log_filename, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)


class MyLogger():
    def __init__(self, log_path, log_name):
        self.terminal = sys.stdout

        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s - %(levelname)1.1s] %(message)s")

        filename = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '_' + log_name
        log_filename = os.path.join(log_path, filename)
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.fh = fh

        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)
        self.console = console

    def write(self, message):
        if message is not None and message != '\n':
            logging.info(message)

    def flush(self):
        self.fh.flush()
        self.console.flush()
