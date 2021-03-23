from qpnet.helper import sample_resize as sr
from qpnet.helper import handle_sample as hs
from qpnet.utils import visualizer as vis
from multiprocessing import Pool
import os
import math
from qpnet.utils.config import cfg
import json
import numpy as np


def load_json(samples):
    with open(samples, 'r') as f:
        data = json.load(f)
    return data['images']


def getInformation(anno):
    file_path = []
    bboxes = []
    size = []
    for i in range(len(anno)):
        file_path.append(anno[i]['filepath'])
        bboxes.append(anno[i]['bboxes'])
        size.append([anno[i]['height'], anno[i]['width'], anno[i]['channel']])
    return file_path, bboxes, size


def checkResize(file_path, bboxes, size):
    for i in range(len(bboxes)):
        file_path[i] = '.' + file_path[i]
        sx, sy = sr.sampleResize(filepath=file_path[i:i + 1],
                                 bboxes=bboxes[i:i + 1],
                                 im_size=size[i:i + 1],
                                 short_side=cfg.DATA.SHORT_SIDE,
                                 long_side=cfg.DATA.LONG_SIDE,
                                 resizeh=cfg.DATA.RESIZE_H,
                                 resizew=cfg.DATA.RESIZE_W,
                                 resizeX=True,
                                 resizeY=True)
        print(file_path[i])
        vis.show_bbox_in_one_image(sx, sy)


def generateGridLabel(file_path, bboxes, size):
    if len(size) != len(bboxes):
        print('The length of size_info and y is not same.')
        os._exit(0)

    if Check_Resize2:
        for i in range(len(bboxes)):
            file_path[i] = '.' + file_path[i]
        x, bboxes = sr.sampleResize(filepath=file_path,
                                    bboxes=bboxes,
                                    im_size=size,
                                    short_side=cfg.DATA.SHORT_SIDE,
                                    long_side=cfg.DATA.LONG_SIDE,
                                    resizeh=cfg.DATA.RESIZE_H,
                                    resizew=cfg.DATA.RESIZE_W,
                                    resizeX=True,
                                    resizeY=True)
        vis.show_bbox_in_one_image(x, bboxes)
        input('Stop')

    bboxes = sr.sampleResize(bboxes=bboxes,
                             im_size=size,
                             short_side=cfg.DATA.SHORT_SIDE,
                             long_side=cfg.DATA.LONG_SIDE,
                             resizeh=cfg.DATA.RESIZE_H,
                             resizew=cfg.DATA.RESIZE_W,
                             resizeY=True)[0]

    label_qp = hs.relabel_2D_location(dict_img=bboxes,
                                      height=cfg.DATA.RESIZE_H,
                                      width=cfg.DATA.RESIZE_W,
                                      dsr_x=cfg.DATA.DSR_X,
                                      dsr_y=cfg.DATA.DSR_Y,
                                      cls=cfg.DATA.CLASS_NUM,
                                      cr=cfg.DATA.CROSS_RATIO,
                                      loc_only=True)

    return label_qp


def saveLabeltoNPY(annotations, label_qp, save_path):
    if len(annotations) != label_qp.shape[0]:
        print('The lengths of X and Y are not same.')
        os._exit(0)
    for i in range(len(annotations)):
        npy = save_path + str(
            os.path.basename(
                annotations[i]['filepath']).split('.')[0]) + ".npy"
        np.save(npy, np.squeeze(label_qp[i], axis=2))


def processing(file_path, annotations, bboxes, size, split):
    print('Generating grid samples...')
    label_qp = generateGridLabel(file_path, bboxes, size)
    if Check_Resize is False:
        relabeled_label = f'../{DataName}/label_qp_{split}_{cfg.DATA.DSR_Y}'
        relabeled_label += f'_{cfg.DATA.DSR_X}_{cfg.DATA.CROSS_RATIO}_{cfg.DATA.SQUARE_SIDE}/'
        if not os.path.exists(relabeled_label):
            os.makedirs(relabeled_label)
        saveLabeltoNPY(annotations, label_qp, relabeled_label)
        print('The relabeled labels have been generated.')


def main():
    print('The samples are from json files...')
    for split in [cfg.TRAIN.SPLIT, cfg.TEST.SPLIT]:
        json_file = f'../{DataName}/{split}.json'
        annotations = load_json(json_file)
        file_path, bboxes, size = getInformation(annotations)
        if Check_Resize:
            checkResize(file_path, bboxes, size)

        if not MULTI:
            processing(file_path, annotations, bboxes, size, split)
        else:
            print("MultiProcess:" + MULTI_NUM.__str__())
            pool = Pool(processes=MULTI_NUM)
            tmp = int(math.ceil(len(bboxes) / MULTI_NUM))
            for i in range(MULTI_NUM):
                pool.apply_async(processing,
                                 args=(file_path[i * tmp:i * tmp + tmp],
                                       annotations[i * tmp:i * tmp + tmp],
                                       bboxes[i * tmp:i * tmp + tmp],
                                       size[i * tmp:i * tmp + tmp], split))
            pool.close()
            pool.join()


if __name__ == '__main__':
    DataName = cfg.DATA.NAME
    MULTI = True
    MULTI_NUM = 32
    Check_Resize = False
    Check_Resize2 = False
    main()
