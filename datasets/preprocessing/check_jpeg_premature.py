import os
import skimage.io


def check_jpeg(dir):
    for file in os.listdir(dir):
        img_path = os.path.join(dir, file)
        try:
            img = skimage.io.imread(img_path)
            print(img.shape)
        except Exception:
            print(img_path)


if __name__ == '__main__':
    DIR = '../to/image/path'
    check_jpeg(DIR)
