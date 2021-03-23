#!/usr/bin/env python3
import argparse
import os
import cv2
from yacs.config import CfgNode as CfgNode

# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C
# ------------------------------------------------------------------------------------ #
# preprocessing
# ------------------------------------------------------------------------------------ #
_C.DATA = CfgNode()
# dataset path
_C.DATA.PATH = 'datasets'
# dataset name
_C.DATA.NAME = 'math'
# classes
_C.DATA.CATEGORIES = ['foreground']
# Note: down-sampling rate
_C.DATA.DSR_X = 4
_C.DATA.DSR_Y = 4
# scale
_C.DATA.SQUARE_SIDE = 800
_C.DATA.RESIZE_H = cfg.DATA.SQUARE_SIDE
_C.DATA.RESIZE_W = cfg.DATA.SQUARE_SIDE
# short side and long side
_C.DATA.SHORT_SIDE = cfg.DATA.SQUARE_SIDE
_C.DATA.LONG_SIDE = cfg.DATA.SQUARE_SIDE

_C.DATA.CHANNEL = 3

_C.DATA.CLASS_NUM = 1

_C.DATA.CROSS_RATIO = 0.75
# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()
# Model type
_C.MODEL.TYPE = '.pth'
# Model backbone
_C.MODEL.BACKBONE = 'vgg16'
# output channels of VGG
_C.MODEL.VGG_OUT_CHANNEL = 512
# model path
_C.MODEL.PATH = 'models'
# feature map of CNN
_C.MODEL.FEATURE_MAP_H = cfg.DATA.RESIZE_H // cfg.DATA.DSR_Y
_C.MODEL.FEATURE_MAP_W = cfg.DATA.RESIZE_W // cfg.DATA.DSR_X

# current model path
_C.MODEL.CUR_PATH = os.path.join(
    cfg.MODEL.PATH, cfg.DATA.NAME,
    '_'.join([cfg.MODEL.BACKBONE,
              str(cfg.DATA.DSR_Y),
              str(cfg.DATA.DSR_X)]))

# ------------------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------------------ #
_C.TRAIN = CfgNode()
# TRAIN_DIR_SIGN
_C.TRAIN.SPLIT = "train"
# # Total mini-batch size
_C.TRAIN.BATCH_SIZE = 16
# True：output four quadrants and background
# False: output four quadrants, foreground and background
_C.TRAIN.QP_ONLY = False
# Note: if you only use part of training samples, set False
_C.TRAIN.USE_ALL = True
_C.TRAIN.NUM_IN_EPOCH = 1
# QP relabel
_C.TRAIN.LABEL_PATH = f'{cfg.DATA.PATH}/{cfg.DATA.NAME}/label_qp_{cfg.TRAIN.SPLIT}' \
                      f'_{cfg.DATA.DSR_Y}_{cfg.DATA.DSR_X}_{cfg.DATA.CROSS_RATIO}_{cfg.DATA.SQUARE_SIDE}'
# print frequency
_C.TRAIN.PRINT_ITER = 5
# ------------------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------------------ #
_C.TEST = CfgNode()
# TEST_DIR_SIGN
_C.TEST.SPLIT = "val"
# 结果存放根目录
_C.TEST.PATH = 'results'
# QP 再标注
_C.TEST.LABEL_PATH = f'{cfg.DATA.PATH}/{cfg.DATA.NAME}/label_qp_{cfg.TEST.SPLIT}' \
                     f'_{cfg.DATA.DSR_Y}_{cfg.DATA.DSR_X}_{cfg.DATA.CROSS_RATIO}_{cfg.DATA.SQUARE_SIDE}'
# # Total mini-batch size
_C.TEST.BATCH_SIZE = 16
# filter bounding box
_C.TEST.FILTER_BBOX = True
# output quadrant images
_C.TEST.QP_IMAGE = True
# output BBOX images
_C.TEST.BBOX_IMAGE = True
# filter bad models
_C.TEST.FILTER_MODEL = True
# filter standard
_C.TEST.GOOD = 0.995
_C.TEST.NORMAL = 0.99
# test dir
_C.TEST.CUR_PATH = os.path.join(
    cfg.TEST.PATH, cfg.DATA.NAME,
    '_'.join([cfg.MODEL.BACKBONE,
              str(cfg.DATA.DSR_Y),
              str(cfg.DATA.DSR_X)]))
# ------------------------------------------------------------------------------------ #
# Optimizer options
# ------------------------------------------------------------------------------------ #
_C.OPTIM = CfgNode()  #
# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 10
# the number of GPU
_C.OPTIM.GPU_NUM = 8
# thread
_C.OPTIM.NUM_WORKERS = 4
# learning rate
_C.OPTIM.LR = 0.0001 * 2 * cfg.OPTIM.NUM_WORKERS * cfg.TRAIN.BATCH_SIZE / cfg.OPTIM.GPU_NUM
# momentum
_C.OPTIM.MOMENTUM = 0.9
# decay weight
_C.OPTIM.WEIGHT_DECAY = 0.0001
# transfer learning
_C.OPTIM.TRANSFER = True
# ------------------------------------------------------------------------------------ #
# log
# ------------------------------------------------------------------------------------ #
_C.LOG = CfgNode()
# Log destination ('stdout' or 'file')
_C.LOG.DEST = "stdout"
# log file name
_C.LOG.PATH = 'logs'
# ------------------------------------------------------------------------------------ #
# others
# ------------------------------------------------------------------------------------ #
# Weights to start training from
_C.WEIGHTS = ''
# infer images in dirs
_C.INFER_PATH = ''
# use cudnn optimization
_C.CUDNN_BENCHMARK = True


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def setup_env(description="Config file options.", gpus='7'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("opts",
                        help="See qpnet/utils/config.py for all options",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    _C.merge_from_list(args.opts)
    cfg.freeze()
    from qpnet.utils import logging
    logging.setup_logging()
    logger = logging.get_logger(__name__)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info(f'CUDA_VISIBLE_DEVICES: {gpus}.')
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    logger.info(f'cv2 useOptimized is {cv2.useOptimized()}.')
