from qpnet.utils import logging
from qpnet.utils.config import cfg
import torch.nn as nn
from qpnet.network.feature_extract import VGG16_Backbone
from qpnet.network.lstm import BidirectionalLSTM

logger = logging.get_logger(__name__)


class QPNet(nn.Module):

    def __init__(self):
        super(QPNet, self).__init__()

        if cfg.MODEL.BACKBONE == 'vgg16':
            self.feature_extraction = VGG16_Backbone(cfg.DATA.CHANNEL, cfg.MODEL.VGG_OUT_CHANNEL)
        else:
            raise Exception('No FeatureExtraction module specified')

        self.bilstm = nn.Sequential(
            BidirectionalLSTM(cfg.MODEL.VGG_OUT_CHANNEL, cfg.MODEL.VGG_OUT_CHANNEL // 2, 0),
            BidirectionalLSTM(cfg.MODEL.VGG_OUT_CHANNEL, cfg.MODEL.VGG_OUT_CHANNEL // 2, 1),
            BidirectionalLSTM(cfg.MODEL.VGG_OUT_CHANNEL, 5, 2))

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.bilstm(x)
        x = x[:, :, :, :5] + x[:, :, :, 5:]
        return x.view(-1, x.size(-1))

