import os
import shutil
import tarfile
import cv2
from tqdm import tqdm
from qpnet.utils import logging
from qpnet.utils import metrics as me
from qpnet.utils import postprocessing as pp
from qpnet.utils import visualizer as vis
from qpnet.utils.config import cfg

logger = logging.get_logger(__name__)


def write_qp(x,
             x_path,
             predict_cls,
             predict_loc,
             model_name=None,
             mode='test'):
    result_qp = None
    if mode == 'test':
        result_qp = os.path.join(cfg.TEST.CUR_PATH,
                                 f'predicted_images_{model_name}_grid')
        if not os.path.exists(result_qp):
            os.makedirs(result_qp)
    for i in tqdm(range(x.shape[0])):
        img_path = x_path[i]
        ext = os.path.splitext(img_path)[1]
        base_name = os.path.split(img_path)[1][:-len(ext)]
        img_qp = x[i] * 255
        img_qp = vis.draw_grid_location(img=img_qp,
                                        classes=predict_cls[i],
                                        location=predict_loc[i],
                                        CLA=cfg.DATA.CLASS_NUM)
        if mode == 'infer':
            result_qp = img_path.replace(cfg.DATA.PATH, cfg.TEST.PATH)
            result_qp = os.path.dirname(result_qp) + '_grid'
            if not os.path.exists(result_qp):
                os.makedirs(result_qp)
        result_qp_path = os.path.join(result_qp, f'{base_name}.png')
        cv2.imwrite(result_qp_path, img_qp)


def write_bbox(x,
               x_path,
               predict_cls,
               predict_loc,
               model_name=None,
               mode='test'):
    result_bbox = None
    if mode == 'test':
        result_bbox = os.path.join(cfg.TEST.CUR_PATH,
                                   f'predicted_images_{model_name}_bbox')
        if not os.path.exists(result_bbox):
            os.makedirs(result_bbox)
    for i in tqdm(range(x.shape[0])):
        img_path = x_path[i]
        ext = os.path.splitext(img_path)[1]
        base_name = os.path.split(img_path)[1][:-len(ext)]
        img_bbox = x[i] * 255
        img_bbox = pp.processLocation(img_bbox,
                                      cls_image=predict_cls[i],
                                      locations=predict_loc[i],
                                      dsr_x=cfg.DATA.DSR_X,
                                      dsr_y=cfg.DATA.DSR_Y,
                                      drawBBox=True,
                                      filterBox=cfg.TEST.FILTER_BBOX,
                                      drawSmallBBox=False)
        if mode == 'infer':
            result_bbox = img_path.replace(cfg.DATA.PATH, cfg.TEST.PATH)
            result_bbox = os.path.dirname(result_bbox) + '_bbox'
            if not os.path.exists(result_bbox):
                os.makedirs(result_bbox)
        result_bbox_path = os.path.join(result_bbox, f'{base_name}.png')
        cv2.imwrite(result_bbox_path, img_bbox)


def write_results(x_path, bbox, model_name=None, mode='test'):
    logger.info('Writing the results to files...')
    result_txt = result_img = None
    if mode == 'test':
        result_txt = os.path.join(cfg.TEST.CUR_PATH,
                                  f'predicted_txt_{model_name}')
        result_img = os.path.join(cfg.TEST.CUR_PATH,
                                  f'predicted_img_{model_name}')
        if not os.path.exists(result_txt):
            os.makedirs(result_txt)
        if not os.path.exists(result_img):
            os.mkdir(result_img)

    for i in tqdm(range(len(x_path))):
        img_path = x_path[i]
        ext = os.path.splitext(img_path)[1]
        base_name = os.path.split(img_path)[1][:-len(ext)]

        if mode == 'infer':
            result_dir = img_path.replace(cfg.DATA.PATH, cfg.TEST.PATH)
            result_txt = os.path.dirname(result_dir) + '_txt'
            result_img = os.path.dirname(result_txt) + '_img'
            if not os.path.exists(result_txt):
                os.makedirs(result_txt)
            if not os.path.exists(result_img):
                os.mkdir(result_img)

        cur_txt = os.path.join(result_txt, f'{base_name}.txt')

        bboxes = {}
        # write the predicted results to files
        with open(cur_txt, 'w') as f:
            for b in range(len(bbox[i])):
                class_name = 0
                class_prob = 1.0
                x1 = round(bbox[i][b][0], 2)
                y1 = round(bbox[i][b][1], 2)
                x2 = round(bbox[i][b][2], 2)
                y2 = round(bbox[i][b][3], 2)
                f.write('{} {} {} {} {} {} \n'.format(class_name, class_prob,
                                                      x1, y1, x2, y2))

                if class_name not in bboxes:
                    bboxes[class_name] = []
                bboxes[class_name].append(
                    [x1, y1, x2, y2, class_prob, class_name])

        img = cv2.imread(img_path)
        img = vis.draw_bbox_in_one_image(img, bboxes)
        cur_image = os.path.join(result_img, f'{base_name}.png')
        cv2.imwrite(cur_image, img)
    
    if mode == 'infer':
        logger.info('Congratulation! It finished.')



def calculate_results(model_name):
    logger.info('Calculating the final metrics...')
    eval = me.calculateF1score(cfg.TEST.CUR_PATH, model_name, iou=0.5)

    if cfg.TEST.FILTER_MODEL:
        good_model = os.path.join(cfg.MODEL.CUR_PATH, 'good_model')
        normal_model = os.path.join(cfg.MODEL.CUR_PATH, 'normal_model')
        src_model = os.path.join(cfg.MODEL.CUR_PATH,
                                 f'{model_name}{cfg.MODEL.TYPE}')
        source_dir = os.path.join(cfg.TEST.CUR_PATH,
                                  f'predicted_txt_{model_name}')
        if eval >= cfg.TEST.GOOD:
            if not os.path.exists(good_model):
                os.mkdir(good_model)
            shutil.move(src_model, good_model)
            logger.info(f'{src_model} is a good model with F1={eval:.4f}.')
            tar_file = os.path.join(cfg.TEST.CUR_PATH, 'predicted.tar')
            if os.path.exists(tar_file):
                os.remove(tar_file)
            # os.rename('./predicted', source_dir)
            os.chdir(source_dir)
            tar_file = './predicted.tar'
            with tarfile.open(tar_file, "w:") as tar:
                for file in tqdm(os.listdir('./')):
                    tar.add(file)
            shutil.move('./predicted.tar', '../')
        elif eval >= cfg.TEST.NORMAL:
            if not os.path.exists(normal_model):
                os.mkdir(normal_model)
            shutil.move(src_model, normal_model)
            logger.info(f'{src_model} is a normal model with F1={eval:.4f}.')
        else:
            os.remove(src_model)
            logger.info(f'{src_model} is removed for low F1.')

    logger.info('Congratulation! It finished.')
