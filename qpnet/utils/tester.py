import copy
import time
import os
import numpy as np
from tqdm import tqdm
from qpnet.helper import sample_resize as sr
from qpnet.helper.writer import write_bbox, write_qp
from qpnet.utils import logging
from qpnet.utils import postprocessing as pp
from qpnet.utils.config import cfg
from qpnet.utils.loader import data_loader
from qpnet.utils.checkpoints import get_model
import torch
from qpnet.utils.builders import QPNet
from qpnet.utils.loader import copyStateDict
import torch.nn as nn
import torch.backends.cudnn as cudnn


logger = logging.get_logger(__name__)


# fast test, draw predicted bounding box, and save results to files
# mode = 'test' or 'infer'
def test(infer_path=None, mode='test'):
    logger.info(f'Start {mode}ing...')
    x_path = data_loader(is_train=False) if mode == 'test' else infer_path
    if mode == 'infer':
        assert infer_path is not None, 'No infer path.'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = QPNet().to(device)

    model_file = get_model() if cfg.WEIGHTS == '' else cfg.WEIGHTS
    net.load_state_dict(copyStateDict(torch.load(model_file, map_location=device)))
    logger.info(f'{model_file} is loaded.')

    if device != 'cpu':
        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            net = nn.DataParallel(net)
            cudnn.benchmark = cfg.CUDNN_BENCHMARK

    net.eval()

    predict_cls = np.zeros(
        (len(x_path), cfg.MODEL.FEATURE_MAP_H, cfg.MODEL.FEATURE_MAP_W, 1))
    predict_loc = np.zeros(
        (len(x_path), cfg.MODEL.FEATURE_MAP_H, cfg.MODEL.FEATURE_MAP_W, 5))

    x, info_s, info_p = sr.sampleResize(filepath=x_path,
                                        short_side=cfg.DATA.SHORT_SIDE,
                                        long_side=cfg.DATA.LONG_SIDE,
                                        channel=cfg.DATA.CHANNEL,
                                        resizeh=cfg.DATA.RESIZE_H,
                                        resizew=cfg.DATA.RESIZE_W,
                                        resizeX=True,
                                        sizeinfo=True)

    bbox = []
    logger.info('Predicting and postprocessing...')
    start = time.time()
    for i in tqdm(range(len(x_path))):
        img = x[i]
        input_x = torch.from_numpy(x[i]).permute(2, 0, 1).unsqueeze(0).to(device)
        output = net(input_x).view(cfg.MODEL.FEATURE_MAP_H, cfg.MODEL.FEATURE_MAP_W, 5)
        predict_cls[i] = output[:, :, :1].detach().cpu().numpy()
        predict_loc[i] = output.detach().cpu().numpy()

        box = pp.processLocation(img,
                                 cls_image=copy.deepcopy(predict_cls[i]),
                                 locations=copy.deepcopy(predict_loc[i]),
                                 dsr_x=cfg.DATA.DSR_X,
                                 dsr_y=cfg.DATA.DSR_Y,
                                 filterBox=cfg.TEST.FILTER_BBOX)
        bbox.append(box)

    cost_time = time.time() - start
    logger.info(f'The cost time for an image is {cost_time / len(x_path):.4f}.')

    ext_name = os.path.splitext(model_file)[1]
    file_name = os.path.split(model_file)[1]
    model_name = file_name[:-len(ext_name)]

    # 画小切块的类别
    if cfg.TEST.QP_IMAGE:
        logger.info('Drawing qp images...')
        write_qp(x=x,
                 x_path=x_path,
                 predict_cls=predict_cls,
                 predict_loc=predict_loc,
                 model_name=model_name,
                 mode=mode)

    if cfg.TEST.BBOX_IMAGE:
        logger.info('Drawing bbox images...')
        write_bbox(x=x,
                   x_path=x_path,
                   predict_cls=predict_cls,
                   predict_loc=predict_loc,
                   model_name=model_name,
                   mode=mode)

    bboxes = [sr.bboxesTransform_inverse(bboxes=box,
                                         size=info_s[i],
                                         pad=info_p[i],
                                         isWH=True) for i, box in enumerate(bbox)]

    return bboxes, x_path, model_name
