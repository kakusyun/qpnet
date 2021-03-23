import math
import os
import cv2
import numpy as np
from qpnet.utils import logging

logger = logging.get_logger(__name__)


def bboxesTransform(bboxes, size, pad):
    (hr, wr, h, w, h_o, w_o) = size
    (top_pad, bottom_pad, left_pad, right_pad) = pad

    for i in range(len(bboxes)):
        bboxes[i]['x1'] = min(
            max(int(math.ceil((float(bboxes[i]['x1']) + 1) * wr - 1)), 0),
            w - 1) + left_pad
        bboxes[i]['x2'] = max(
            min(int(math.floor(float(bboxes[i]['x2']) * wr)), w - 1),
            0) + left_pad
        bboxes[i]['y1'] = min(
            max(int(math.ceil((float(bboxes[i]['y1']) + 1) * hr - 1)), 0),
            h - 1) + top_pad
        bboxes[i]['y2'] = min(
            max(int(math.ceil((float(bboxes[i]['y2']) + 1) * hr - 1)), 0),
            h - 1) + top_pad
    return bboxes


def bboxesTransform_inverse(bboxes, size, pad, isWH=False):
    (hr, wr, h, w, h_o, w_o) = size
    (top_pad, bottom_pad, left_pad, right_pad) = pad

    for i in range(len(bboxes)):
        if isWH:
            bboxes[i][2] += bboxes[i][0] - 1
            bboxes[i][3] += bboxes[i][1] - 1
        bboxes[i][0] = min(
            max(int(math.ceil((float(bboxes[i][0]) - left_pad + 1) / wr - 1)),
                0), w_o - 1)
        bboxes[i][2] = max(
            min(int(math.floor((float(bboxes[i][2]) - left_pad) / wr)),
                w_o - 1), 0)
        bboxes[i][1] = min(
            max(int(math.ceil((float(bboxes[i][1]) - top_pad + 1) / hr - 1)),
                0), h_o - 1)
        bboxes[i][3] = max(
            min(int(math.floor((float(bboxes[i][3]) - top_pad) / hr)),
                h_o - 1), 0)
    return bboxes


# x1,y1,x2,y2,x3,y3,x4,y4
def bboxesRotate90anti(bboxes, h):
    for i in range(len(bboxes)):
        x1 = bboxes[i]['x1']
        y1 = bboxes[i]['y1']
        bboxes[i]['x1'] = bboxes[i]['y2']
        bboxes[i]['y1'] = str(int(h - float(bboxes[i]['x2'])))
        bboxes[i]['x2'] = bboxes[i]['y3']
        bboxes[i]['y2'] = str(int(h - float(bboxes[i]['x3'])))
        bboxes[i]['x3'] = bboxes[i]['y4']
        bboxes[i]['y3'] = str(int(h - float(bboxes[i]['x4'])))
        bboxes[i]['x4'] = y1
        bboxes[i]['y4'] = str(int(h - float(x1)))
    return bboxes


# x1,y1,x2,y2,x3,y3,x4,y4
def bboxesRotate90anti_inverse(bboxes, h):
    for i in range(len(bboxes)):
        x4 = bboxes[i][6]
        y4 = bboxes[i][7]
        bboxes[i][6] = int(h - float(bboxes[i][5]))
        bboxes[i][7] = int(bboxes[i][4])
        bboxes[i][4] = int(h - float(bboxes[i][3]))
        bboxes[i][5] = int(bboxes[i][2])
        bboxes[i][2] = int(h - float(bboxes[i][1]))
        bboxes[i][3] = int(bboxes[i][0])
        bboxes[i][0] = int(y4)
        bboxes[i][1] = int(h - float(x4))
    return bboxes


def exchange(x, y):
    return y, x


def sampleResize(filepath=None,
                 bboxes=None,
                 im_size=None,
                 short_side=800,
                 long_side=1024,
                 resizeh=1024,
                 resizew=1024,
                 channel=3,
                 resizeX=False,
                 resizeY=False,
                 sizeinfo=False,
                 h_short=False):
    length = 0
    if bboxes is not None:
        length = len(bboxes)
        assert im_size is not None, 'If output bboxes, need im_size.'
    elif filepath is not None:
        length = len(filepath)
    else:
        logger.info('bboxes or filepath must have one.')

    if resizeX or resizeY or sizeinfo:
        imgs, info_s, info_p = [], [], []
        for m in range(length):
            if resizeX:
                img = cv2.imread(filepath[m],
                                 0) if channel == 1 else cv2.imread(
                    filepath[m])
                h_ori, w_ori = img.shape[0], img.shape[1]
                if h_short and h_ori > w_ori:
                    img = np.rot90(img)
                    h_ori, w_ori = w_ori, h_ori

            if resizeY:
                h_ori, w_ori = im_size[m][0], im_size[m][1]
                if h_short and h_ori > w_ori:
                    h_ori, w_ori = w_ori, h_ori
                    bboxes[m] = bboxesRotate90anti(bboxes[m], h_ori)

            if sizeinfo:
                assert resizeX, 'resizeX must be true.'
                h_ori, w_ori = img.shape[0], img.shape[1]
                if h_short and h_ori > w_ori:
                    h_ori, w_ori = w_ori, h_ori

            height_ratio = short_side / h_ori
            w = round(height_ratio * w_ori, 0)
            h = short_side
            width_ratio = w / w_ori
            if w > long_side:
                width_ratio = long_side / w_ori
                h = round(width_ratio * h_ori, 0)
                w = long_side
                height_ratio = h / h_ori
                if h > short_side:
                    h = short_side
                    height_ratio = h / h_ori

            w = int(w)
            h = int(h)
            top_pad = int((resizeh - h) // 2)
            bottom_pad = int(resizeh - h - top_pad)
            left_pad = int((resizew - w) // 2)
            right_pad = int(resizew - w - left_pad)
            size_info = (height_ratio, width_ratio, h, w, h_ori, w_ori)
            pad_info = (top_pad, bottom_pad, left_pad, right_pad)

            if sizeinfo:
                info_s.append(size_info)
                info_p.append(pad_info)

            if resizeY:
                bboxes[m] = bboxesTransform(bboxes[m], size_info, pad_info)

            if resizeX:  # for path in x_path:
                img = cv2.resize(img, (w, h))
                if channel == 1:
                    img = img.reshape((img.shape[0], img.shape[1], 1))
                padding = [(top_pad, bottom_pad), (left_pad, right_pad),
                           (0, 0)]
                img = np.pad(img, padding, mode='constant', constant_values=0)
                img = img.astype('float32')
                img = img / 255
                # img -= 1.
                imgs.append(img)

        res = []
        if resizeX:
            res.append(
                np.array(imgs).reshape(len(filepath), resizeh, resizew,
                                       channel))
        if resizeY:
            res.append(bboxes)
        if sizeinfo:
            res.append(info_s)
            res.append(info_p)
        return res
    else:
        logger.info('Nothing changes, please check.')
        os._exit(0)
