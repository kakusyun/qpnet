from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
from qpnet.utils.config import cfg
from qpnet.utils import logging
import torch.backends.cudnn as cudnn
from qpnet.utils.loader import copyStateDict
import time
from qpnet.utils.loader import data_loader
from qpnet.utils.dataset import MyDataset
from qpnet.utils.builders import QPNet
from qpnet.utils.checkpoints import get_model

logger = logging.get_logger(__name__)


def train_model(net, dataloaders_dict, criterion, optimizer, device, gpus):
    for epoch in range(cfg.OPTIM.MAX_EPOCH):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0
            epoch_start = time.time()
            iter_loss = 0
            iter_start = time.time()
            epoch_corrects = 0.0
            epoch_points = 0
            iter_corrects = 0.0
            iter_points = 0

            for i, data in enumerate(dataloaders_dict[phase]):
                img = data['img'].to(device)
                gt = data['gt'].to(device).view(-1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y = net(img)
                    loss = criterion(y, gt)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if i % cfg.TRAIN.PRINT_ITER == 0 and i > 0:
                            logger.info(f'epoch:{epoch + 1}/{cfg.OPTIM.MAX_EPOCH} '
                                        f'iter:{i:03d} '
                                        f'loss:{iter_loss / (cfg.TRAIN.PRINT_ITER * gpus):.4f} '
                                        f'time:{(time.time() - iter_start) / cfg.TRAIN.PRINT_ITER:.4f} '
                                        f'iter_qp_acc:{iter_corrects / iter_points:.4f}')
                            iter_corrects = 0.0
                            iter_points = 0
                            iter_loss = 0
                            iter_start = time.time()

                _, preds = torch.max(y, 1)
                cur_corrects = torch.sum(preds == gt.data)
                epoch_corrects += cur_corrects
                epoch_points += len(gt)

                iter_corrects += cur_corrects
                iter_points += len(gt)

                epoch_loss += loss.item()
                iter_loss += loss.item()

            n = len(dataloaders_dict[phase].dataset)
            logger.info(f'epoch:{epoch + 1}/{cfg.OPTIM.MAX_EPOCH} '
                        f'{phase}_loss:{epoch_loss / n:.4f} '
                        f'{phase}_time:{gpus * (time.time() - epoch_start) / n:.4f} '
                        f'{phase}_qp_acc:{epoch_corrects / epoch_points:.4f}')
            if phase == 'val':
                model_file = os.path.join(cfg.MODEL.CUR_PATH, f'{cfg.DATA.NAME}-{epoch + 1:03d}-'
                                                              f'{epoch_corrects / epoch_points:.4f}'
                                                              f'{cfg.MODEL.TYPE}')
                torch.save(net.state_dict(), model_file)


def train():
    # model path
    os.makedirs(cfg.MODEL.CUR_PATH, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train_path, x_val_path = data_loader()

    image_datasets = {'train': MyDataset(x_train_path, train=True),
                      'val': MyDataset(x_val_path, train=False)}

    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=cfg.OPTIM.NUM_WORKERS)
                        for x in ['train', 'val']}

    net = QPNet().to(device)

    # pretrain model
    if not os.path.exists(cfg.WEIGHTS):
        if cfg.OPTIM.TRANSFER:
            model_file = get_model()
            net.load_state_dict(copyStateDict(torch.load(model_file, map_location=device)))
            logger.info(f'{model_file} is loaded.')
        else:
            logger.info('No pretrained model. Training will start from scratch.')
    else:
        net.load_state_dict(copyStateDict(torch.load(cfg.WEIGHTS, map_location=device)))
        logger.info(f'{cfg.WEIGHTS} is loaded.')

    if device != 'cpu':
        gpus = torch.cuda.device_count()
        if gpus > 1:
            logger.info(f"Let's use {gpus} GPUs!")
            net = nn.DataParallel(net)
    else:
        gpus = 1  # if cpu, set 1
    assert gpus == 8, 'Please confirm the learning rate in config.py.'

    cudnn.benchmark = cfg.CUDNN_BENCHMARK
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=cfg.OPTIM.LR,
                                 weight_decay=cfg.OPTIM.WEIGHT_DECAY)

    train_model(net=net,
                dataloaders_dict=dataloaders_dict,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                gpus=gpus)
