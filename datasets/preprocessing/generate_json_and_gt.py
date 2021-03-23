import cv2
import numpy as np
from tqdm import tqdm
import os
from qpnet.utils.config import cfg
import json
import shutil
import skimage.io


def classMapping(cls):
    class_list = cfg.DATA.CATEGORIES
    if cls not in class_list:
        print('The current class is wrong.')
        os._exit(0)
    else:
        return str(class_list.index(cls))


def generate_ground_truth(data, split):
    GroundTruthPath = os.path.join(f'../{DataName}', f'ground-truth-{split}')
    if os.path.exists(GroundTruthPath):
        shutil.rmtree(GroundTruthPath)
    os.makedirs(GroundTruthPath)

    for i in range(len(data)):
        ext_name = os.path.splitext(data[i]['filepath'])[1]
        ground_truth_txt = os.path.join(
            GroundTruthPath,
            os.path.split(data[i]['filepath'])[1].replace(ext_name, '.txt'))

        # groundtruth to files
        with open(ground_truth_txt, 'w') as f:
            for b in range(len(data[i]['bboxes'])):
                f.write('{} {} {} {} {}\n'.format(
                    data[i]['bboxes'][b]['class'], data[i]['bboxes'][b]['x1'],
                    data[i]['bboxes'][b]['y1'], data[i]['bboxes'][b]['x2'],
                    data[i]['bboxes'][b]['y2']))


def get_data(input_path):
    all_imgs = {}
    class_count = {}
    class_mapping = {}
    image_num = 0
    image_width_sum = 0
    image_width_max = 0
    image_width_min = 5000
    image_height_sum = 0
    image_height_max = 0
    image_height_min = 5000

    with open(input_path, 'r', encoding='utf-8') as f:
        print('Parsing annotation files')
        for line in tqdm(f.readlines()):
            line_split = line.strip().split('\t')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if x1 == x2 == y1 == y2 == '-1':
                continue
            else:
                if int(x1) < 0 or int(x2) < 0 or int(y1) < 0 or int(y2) < 0:
                    print('The coordinates are out of the image.')
                    print(filename)
                    os._exit(0)

                class_name = 'foreground'
                if class_name not in class_count:
                    class_count[class_name] = 1
                else:
                    class_count[class_name] += 1

                if classMapping(class_name) not in class_mapping:
                    class_mapping[classMapping(
                        class_name)] = class_name

                if filename not in all_imgs:
                    all_imgs[filename] = {}
                    try:
                        img = skimage.io.imread(filename.replace(cfg.DATA.PATH, '..'))
                    except Exception:
                        print(f'\nPremature end of JPEG file: {filename}.')

                    (H, W, C) = img.shape
                    if C != 3:
                        print('There is a mistake.')
                        os._exit(0)

                    image_num += 1
                    image_width_sum += W
                    image_height_sum += H
                    image_width_max = max(image_width_max, W)
                    image_width_min = min(image_width_min, W)
                    image_height_max = max(image_height_max, H)
                    image_height_min = min(image_height_min, H)

                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['height'] = H
                    all_imgs[filename]['width'] = W
                    all_imgs[filename]['channel'] = C

                    FLAG = np.random.randint(0, 9)
                    if FLAG > 0:
                        all_imgs[filename]['imageset'] = cfg.TRAIN.SPLIT
                    else:
                        all_imgs[filename]['imageset'] = cfg.TEST.SPLIT

                    if SavePixel:
                        all_imgs[filename]['pixel'] = img

                    all_imgs[filename]['bboxes'] = []

                all_imgs[filename]['bboxes'].append({
                    'class': classMapping(class_name),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })

    print('The max width is {}.'.format(image_width_max))
    print('The min width is {}.'.format(image_width_min))
    print('The average width is {}.'.format(image_width_sum / image_num))
    print('The max height is {}.'.format(image_height_max))
    print('The min height is {}.'.format(image_height_min))
    print('The average height is {}.'.format(image_height_sum / image_num))
    print(class_count)
    print(class_mapping)

    train_data = []
    val_data = []
    for key in all_imgs:
        if all_imgs[key]['imageset'] == cfg.TRAIN.SPLIT:
            train_data.append(all_imgs[key])
        elif all_imgs[key]['imageset'] == cfg.TEST.SPLIT:
            val_data.append(all_imgs[key])
        else:
            print('Error!')

    generate_ground_truth(train_data, cfg.TRAIN.SPLIT)
    generate_ground_truth(val_data, cfg.TEST.SPLIT)
    train_data = check_path(train_data)
    val_data = check_path(val_data)

    return train_data, val_data, class_mapping


def check_path(data):
    assert data[0]['filepath'].startswith(cfg.DATA.PATH)
    if data and data[0]['filepath'][:2] == '..':
        for i in range(len(data)):
            data[i]['filepath'] = data[i]['filepath'][1:]
    return data


def save_json(data, class_mapping, split):
    dic = {'images': data, 'class_mapping': class_mapping}
    json_file = f'../{DataName}/{split}.json'
    if os.path.exists(json_file):
        os.remove(json_file)
    with open(json_file, 'w') as f:
        json.dump(dic, f, indent=2)


def main():
    anno_file = f'../{DataName}/annotations.txt'
    train_data, val_data, class_mapping = get_data(anno_file)
    print(f'{len(train_data)} training samples.')
    print(f'{len(val_data)} validation samples.')
    save_json(train_data, class_mapping, cfg.TRAIN.SPLIT)
    save_json(val_data, class_mapping, cfg.TEST.SPLIT)


if __name__ == '__main__':
    np.random.seed(0)
    DataName = cfg.DATA.NAME
    SavePixel = False
    main()
