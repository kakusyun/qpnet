from torch.utils.data import Dataset
import numpy as np
from qpnet.helper.get_flle import get_corresponding_file as gcf
from qpnet.helper import sample_resize as sr
from qpnet.utils.config import cfg


class MyDataset(Dataset):
    def __init__(self, imglist, train=True):
        self.imglist = imglist
        self.is_train = train

    def __getitem__(self, index):
        # read img, region_map, affinity_map
        img_file = self.imglist[index]
        img = sr.sampleResize(filepath=[img_file],
                              short_side=cfg.DATA.SHORT_SIDE,
                              long_side=cfg.DATA.LONG_SIDE,
                              channel=cfg.DATA.CHANNEL,
                              resizeh=cfg.DATA.RESIZE_H,
                              resizew=cfg.DATA.RESIZE_W,
                              resizeX=True)[0][0]
        img = np.transpose(img, (2, 0, 1))

        if self.is_train:
            gt_file = gcf(img_file, cfg.TRAIN.LABEL_PATH, '.npy')
        else:
            gt_file = gcf(img_file, cfg.TEST.LABEL_PATH, '.npy')

        gt_map = np.load(gt_file).astype(np.long)
        return {'img': img, 'gt': gt_map}

    def __len__(self):
        return len(self.imglist)
