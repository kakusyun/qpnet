import logging
import os
import sys
import time

from qpnet.utils.config import cfg


def setup_logging():
    logging.root.handlers = []
    fmt = "\033[32m[%(asctime)s %(name)s:%(lineno)3d]:\033[0m %(message)s"
    data_fmt = "%Y-%m-%d %H:%M:%S"
    logging_config = {
        "level": logging.INFO,
        "format": fmt,
        "datefmt": data_fmt
    }
    cur_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    if cfg.LOG.DEST == "file" and not os.path.exists(cfg.LOG.PATH):
        os.makedirs(cfg.LOG.PATH)

    # Log either to stdout or to a file
    if cfg.LOG.DEST == "stdout":
        logging_config["stream"] = sys.stdout
    else:
        logging_config["filename"] = os.path.join(cfg.LOG.PATH,
                                                  cur_time + '.log')
    # Configure logging
    logging.basicConfig(**logging_config)


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)
