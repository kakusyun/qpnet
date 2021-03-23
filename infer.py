import os

infer_path = 'datasets/math/test'
weight = 'models/math/vgg16_4_4/good_model/math-030-0.9806.pth'
os.system(f'python tools/infer_qp.py INFER_PATH {infer_path} WEIGHTS {weight}')
