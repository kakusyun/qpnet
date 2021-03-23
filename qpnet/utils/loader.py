import os
from datasets.preprocessing.relabeling_location_npy import (getInformation,
                                                            load_json)
from qpnet.utils import logging
from qpnet.utils.config import cfg
from collections import OrderedDict

logger = logging.get_logger(__name__)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def infer_files_loader(path):
    assert path is not None, 'No infer path!'
    x_path = []
    for root, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith(('jpeg', 'jpg', 'png')):
                x_path.append(os.path.join(root, f))
    return x_path


def data_loader(is_train=True):
    logger.info('All the sample paths are from json files.')
    x_train_path = None
    if is_train:
        train_json = os.path.join(cfg.DATA.PATH, cfg.DATA.NAME,
                                  f'{cfg.TRAIN.SPLIT}.json')
        train_annotations = load_json(train_json)
        logger.info(
            f'The total training samples are \033[1;31m{len(train_annotations)}\033[0m.'
        )
        x_train_path, _, _ = getInformation(train_annotations)

    val_json = os.path.join(cfg.DATA.PATH, cfg.DATA.NAME,
                            f'{cfg.TEST.SPLIT}.json')
    val_annotations = load_json(val_json)
    logger.info(
        f'The total val samples are \033[1;31m{len(val_annotations)}\033[0m.')
    x_val_path, _, _ = getInformation(val_annotations)

    if is_train:
        return x_train_path, x_val_path
    else:
        return x_val_path
