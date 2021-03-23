#!/usr/bin/evn python
# coding:utf-8
import json
from tqdm import tqdm
import os
from qpnet.utils.config import cfg


# deal with the labelme json file.


def check_error(x1, y1, x2, y2, img_path=None):
    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        raise Exception(f'Over low boundary: {x1}, {y1}, {x2}, {y2}')

    if x1 > x2 or y1 > y2:
        print('error code:5.')
        raise Exception(f'Wrong bbox: {x1}, {y1}, {x2}, {y2}')


def check_labels_images(labels, images):
    assert len(labels) == len(images), 'labels is not equal to images.'
    for i in range(len(labels)):
        if os.path.splitext(labels[i])[0] != os.path.splitext(images[i])[0]:
            print('The image does not match its label, error code:1.')
            print(images[i], labels[i])
            os._exit(0)


def check_wrong_samples_in_anno(anno_file, w_samples):
    w_samples = set(w_samples)
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    print(f'The txt file has {len(lines)} rows before.')
    with open(anno_file, 'w') as f:
        count = 0
        for line in lines:
            if line.strip().split()[0] not in w_samples:
                f.write(line)
                count += 1
    print(f'The txt file has {count} rows after.')


def generateAnn(jpeg_path, json_path, ann_file):
    print('Train Converting Start...')
    assert len(jpeg_path) == len(
        json_path), 'json_dir is not equal to jpeg_dir.'
    wrong_sampels = []
    total_samples = 0
    with open(ann_file, 'w') as f:
        for j in range(len(jpeg_path)):
            jpeg_dir = os.path.join(DataPath, jpeg_path[j]).replace('\\', '/')
            json_dir = os.path.join(DataPath, json_path[j])

            all_image_files = [
                i for i in os.listdir(jpeg_dir)
                if i.endswith(('.jpg', '.jpeg', '.png'))
            ]
            all_label_files = [
                i for i in os.listdir(json_dir) if i.endswith('.json')
            ]
            all_label_files.sort()
            all_image_files.sort()
            check_labels_images(all_label_files, all_image_files)

            n = len(all_image_files)
            total_samples += n

            for i in tqdm(range(n)):
                try:
                    label = json.load(open(
                        os.path.join(json_dir, all_label_files[i]), 'r'),
                        strict=False)
                    image = all_image_files[i]

                    for i in range(len(label['shapes'])):
                        if label['shapes'][i]['shape_type'] != 'rectangle':
                            print('error code:2.')
                            raise Exception('labeled by rectangle.')

                        # Note: 小数还是整数后面确认！
                        x1 = label['shapes'][i]['points'][0][0]
                        y1 = label['shapes'][i]['points'][0][1]
                        x2 = label['shapes'][i]['points'][1][0]
                        y2 = label['shapes'][i]['points'][1][1]

                        file_path = os.path.join(jpeg_dir, image)
                        check_error(x1, y1, x2, y2, file_path)
                        file_path = file_path.replace('\\', '/')
                        file_path = file_path.replace('..', cfg.DATA.PATH, 1)

                        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            file_path, x1, y1, x2, y2, 'text'))
                except Exception:
                    # print(f'A wrong sample {image} is not used.')
                    file_path = os.path.join(jpeg_dir, image)
                    file_path = file_path.replace('\\', '/')
                    file_path = file_path.replace('..', cfg.DATA.PATH, 1)
                    wrong_sampels.append(file_path)
                    continue

    print(f'{total_samples} samples.')
    print(f'{len(wrong_sampels)} wrong samples.')
    print(f'{total_samples - len(wrong_sampels)} good samples')
    if len(wrong_sampels) > 0:
        check_wrong_samples_in_anno(ann_file, wrong_sampels)


def main():
    # Note: datasets
    JpegPath = ['images']
    JsonPath = ['jsons']
    AnnoFile = os.path.join(f'../{DataName}', 'annotations.txt')
    if os.path.exists(AnnoFile):
        os.remove(AnnoFile)
    generateAnn(jpeg_path=JpegPath, json_path=JsonPath, ann_file=AnnoFile)


if __name__ == '__main__':
    DataName = cfg.DATA.NAME
    DataPath = os.path.join(f'../{DataName}', 'original')
    main()
