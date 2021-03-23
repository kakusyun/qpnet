import datetime
import os
import pandas as pd
from tqdm import tqdm
from qpnet.helper import NMS
from qpnet.utils import logging
from qpnet.utils.config import cfg

logger = logging.get_logger(__name__)


def calculateF1score(ResultPath, model_name, iou=0.5):
    gt_dir = os.path.join(cfg.DATA.PATH, cfg.DATA.NAME, 'ground-truth-val')
    predict_dir = os.path.join(ResultPath, f'predicted_txt_{model_name}')
    assert os.path.exists(gt_dir), "No ground truth!"
    assert os.path.exists(predict_dir), "No predicted results!"

    gt_list = os.listdir(gt_dir)
    predict_list = os.listdir(predict_dir)

    if 'desktop.ini' in gt_list:
        gt_list.remove('desktop.ini')
    if gt_list != predict_list:
        logger.info("The number of file is not equal!!")
        os._exit(0)

    TP = 0
    FN = 0
    FP = 0
    for gt_file in tqdm(gt_list):
        gt = []
        with open(os.path.join(gt_dir, gt_file), 'r') as f_gt:
            for gt_line in f_gt:
                line_split = gt_line.strip().split(' ')
                # if int(float(line_split[-1])) == 0:
                gt.append(list(map(int, line_split[1:5])))

        predict = []
        with open(os.path.join(predict_dir, gt_file), 'r') as f_pred:
            for pred_line in f_pred:
                line_split = pred_line.strip().split(' ')
                predict.append(list(map(float, line_split[2:])))

        for i in range(len(gt)):
            if len(predict) > 0:
                hit_iou = 0
                hit_j = None
                for j in range(len(predict)):
                    IoU = NMS.calculate_iou(gt[i], predict[j])
                    if IoU > iou and IoU > hit_iou:
                        hit_j = j
                        hit_iou = IoU
                if hit_iou == 0:
                    FN += 1
                else:
                    TP += 1
                    del predict[hit_j]
            else:
                FN += (len(gt) - i)
                break
        if len(predict) > 0:
            FP += len(predict)

    ACC = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = 2 * Precision * Recall / (Precision + Recall)

    logger.info(f"\033[1;33m1. \033[1;34mAccuracy: {ACC:.4f}\033[0m")
    logger.info(f"\033[1;33m2. \033[1;34mPrecision: {Precision:.4f}\033[0m")
    logger.info(f"\033[1;33m3. \033[1;34mRecall: {Recall:.4f}\033[0m")
    logger.info(f"\033[1;33m4. \033[1;34mF1-score: {F1:.4f}\033[0m")
    logger.info(f"\033[1;33m5. \033[1;34mTP, True positives: {TP}\033[0m")
    logger.info(f"\033[1;33m6. \033[1;34mFP, False positives: {FP}\033[0m")
    logger.info(f"\033[1;33m7. \033[1;34mFN, False negatives: {FN}\033[0m")
    logger.info(f"\033[1;33m8. \033[1;34mThe total test number: {len(gt_list)}\033[0m")
    current_time = datetime.datetime.strftime(datetime.datetime.now(),
                                              '%Y-%m-%d %H:%M:%S')
    # logger.info(current_time)

    Result = {
        'Accuracy': ACC,
        'Precision': Precision,
        'Recall': Recall,
        'F1-score': F1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'test_time': current_time,
        'model': model_name
    }

    columns = ['test_time', 'model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'TP', 'FP', 'FN']

    DataFrame = pd.DataFrame([Result])

    ResPath = os.path.join(cfg.TEST.PATH, f'{cfg.DATA.NAME}_results.csv')
    if os.path.exists(ResPath):
        DataFrame.to_csv(ResPath,
                         sep=',',
                         columns=columns,
                         header=None,
                         mode='a',
                         index=0)
    else:
        DataFrame.to_csv(ResPath, sep=',', columns=columns, index=0)
    return F1


if __name__ == '__main__':
    path_res = '../result/vgg16'
    mn = 'try'
    logger.info(calculateF1score(path_res, mn))
