import numpy as np
from tqdm import tqdm
import os
from qpnet.utils import logging

logger = logging.get_logger(__name__)


def calculateCrossRatio_min(rec1, rec2):
    """
    computing cross ratio
    :param rec1: (x0, y0, w0, h0), which reflects
            (top, left, width, height)
    :param rec2: (x1, y1, w1, h1)
    :return: scala value of cross ratio
    It is special!!!
    rec1: grid
    rec2: bbox
    """
    set_grid_x = set(range(rec1[0], rec1[0] + rec1[2]))
    set_grid_y = set(range(rec1[1], rec1[1] + rec1[3]))
    set_bbox_x = set(range(rec2[0], rec2[0] + rec2[2]))
    set_bbox_y = set(range(rec2[1], rec2[1] + rec2[3]))

    special_cross_ratio_x = len(set_grid_x & set_bbox_x) / min(
        len(set_grid_x), len(set_bbox_x))
    special_cross_ratio_y = len(set_grid_y & set_bbox_y) / min(
        len(set_grid_y), len(set_bbox_y))
    special_cross_ratio = min(special_cross_ratio_x, special_cross_ratio_y)
    if special_cross_ratio > 1:
        logger.info('The cross ratio is wrong.')
        os._exit(0)

    return special_cross_ratio


def relabel_2D_location(dict_img, height, width, dsr_x, dsr_y, cls, cr, loc_only=False):
    if not loc_only:
        label_cls = np.ones(
            (len(dict_img), height // dsr_y, width // dsr_x, 1)) * cls
    label_qp = np.zeros(
        (len(dict_img), height // dsr_y, width // dsr_x, 1))

    logger.info('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        bb_list = dict_img[i]
        for y in range(0, height, dsr_y):  # 32, 4, y
            for x in range(0, width, dsr_x):  # 160, 4, x
                grid = [x, y, dsr_x, dsr_y]
                grid_cen_x = (x + x + dsr_x - 1) / 2
                grid_cen_y = (y + y + dsr_y - 1) / 2
                cross_tmp = 0
                box_area_tmp = 0
                for j in range(len(bb_list)):
                    bounding_box = [
                        bb_list[j]['x1'], bb_list[j]['y1'],
                        bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                        bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                    ]
                    cross_ratio = calculateCrossRatio_min(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    if cross_ratio >= cr:
                        if cross_ratio > cross_tmp:
                            if not loc_only:
                                label_cls[i, y // dsr_y, x // dsr_x,
                                          0] = int(bb_list[j]['class'])
                            box_cen_x = (bb_list[j]['x1'] +
                                         bb_list[j]['x2']) / 2
                            box_cen_y = (bb_list[j]['y1'] +
                                         bb_list[j]['y2']) / 2
                            if grid_cen_x < box_cen_x and grid_cen_y <= box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 1
                            elif grid_cen_x >= box_cen_x and grid_cen_y < box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 2
                            elif grid_cen_x <= box_cen_x and grid_cen_y > box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 3
                            elif grid_cen_x > box_cen_x and grid_cen_y >= box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 4
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            if not loc_only:
                                label_cls[i, y // dsr_y, x // dsr_x,
                                          0] = int(bb_list[j]['class'])
                            box_cen_x = (bb_list[j]['x1'] +
                                         bb_list[j]['x2']) / 2
                            box_cen_y = (bb_list[j]['y1'] +
                                         bb_list[j]['y2']) / 2
                            if grid_cen_x < box_cen_x and grid_cen_y <= box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 1
                            elif grid_cen_x >= box_cen_x and grid_cen_y < box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 2
                            elif grid_cen_x <= box_cen_x and grid_cen_y > box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 3
                            elif grid_cen_x > box_cen_x and grid_cen_y >= box_cen_y:
                                label_qp[i, y // dsr_y, x // dsr_x,
                                         0] = 4
                            box_area_tmp = box_area

    return label_qp if loc_only else (label_cls, label_qp)
