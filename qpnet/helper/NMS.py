import os
import time
import numpy as np
from qpnet.helper import polygon_calculate as pc
from qpnet.utils import logging

logger = logging.get_logger(__name__)


def calculate_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


def non_max_suppression_fast(boxes,
                             overlap_thresh=0.9,
                             max_boxes=300,
                             current_model=None):
    """
    :param boxes: boxes
    :param overlap_thresh:
    :param max_boxes:
    :return: boxes (x1,y1,x2,y2)
    """
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    # Note: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob, Cls] format!!!! with prob built in
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    try:
        np.testing.assert_array_less(x1, x2)
        np.testing.assert_array_less(y1, y2)
    except:
        os.remove(current_model)
        logger.info('A bad model is removed.')
        count = 0
        b = 10
        while (count < b):
            logger.info(b - count)
            time.sleep(1)
            count += 1
        os._exit(0)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1) * (y2 - y1)
    indexes = np.argsort([i[-2] for i in boxes])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        overlap = area_int / (area_union + 1e-6)
        indexes = np.delete(
            indexes,
            np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick]
    return boxes


def non_max_suppression_fast_polygon(boxes, overlap_thresh=0.9, max_boxes=300):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    x3 = boxes[:, 4]
    y3 = boxes[:, 5]
    x4 = boxes[:, 6]
    y4 = boxes[:, 7]

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    indexes = np.argsort([i[-2] for i in boxes])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        poly1 = [x1[i], y1[i], x2[i], y2[i], x3[i], y3[i], x4[i], y4[i]]

        overlap = np.zeros((last,))
        for j in range(last):
            poly2 = [
                x1[indexes[j]], y1[indexes[j]], x2[indexes[j]], y2[indexes[j]],
                x3[indexes[j]], y3[indexes[j]], x4[indexes[j]], y4[indexes[j]]
            ]
            overlap[j] = pc.calculate_polygon_IoU(poly1, poly2)

        indexes = np.delete(
            indexes,
            np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick]
    return boxes


def IoM_suppression_fast(boxes, overlap_thresh=1.0, max_boxes=300):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2] - 1
    y2 = boxes[:, 1] + boxes[:, 3] - 1

    try:
        np.testing.assert_array_less(x1, x2)
        np.testing.assert_array_less(y1, y2)
    except:
        os._exit(0)

    pick = []
    area = (x2 - x1) * (y2 - y1)
    indexes = np.argsort([i[-1] for i in boxes])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        area_min = np.minimum(area[i], area[indexes[:last]])

        overlap = area_int / (area_min)

        indexes = np.delete(
            indexes,
            np.concatenate(([last], np.where(overlap >= overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick]
    return boxes
