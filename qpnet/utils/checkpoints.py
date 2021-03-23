import os
from qpnet.utils import logging
from qpnet.utils.config import cfg

logger = logging.get_logger(__name__)


def check_model(model_path):
    if os.path.exists(model_path):
        models = os.listdir(model_path)
        if models is False:
            return False, 'NO MODEL'
        else:
            models.sort(key=lambda fn: os.path.getmtime(model_path + '/' + fn),
                        reverse=True)
            flag = 0
            for m in models:
                if m.endswith(cfg.MODEL.TYPE):
                    Mname = os.path.join(model_path, m)
                    flag = 1
                    epoch_num = m.split('-')[1]
                    warning = f'The model name {m} is not defined properly.'
                    try:
                        assert (len(epoch_num) == 3), f'Assert: {warning}'
                        cfg.OPTIM.defrost()
                        cfg.OPTIM.INIT_EPOCH = int(epoch_num)
                        cfg.OPTIM.freeze()
                    except Exception:
                        logger.info(warning)
                    return True, Mname
            if flag == 0:
                return False, 'NO MODEL'
    else:
        os.makedirs(model_path)
        return False, 'NO MODEL'


def clear_model(model_path):
    if os.path.exists(model_path):
        models = os.listdir(model_path)
        if models:
            models.sort(key=lambda fn: os.path.getmtime(model_path + '/' + fn),
                        reverse=True)
            flag = 0
            Mname = []
            for m in models:
                if m.endswith(cfg.MODEL.TYPE):
                    flag += 1
                    Mname.append(os.path.join(model_path, m))
            if flag > 3:
                for m in Mname[3:]:
                    os.remove(m)


def get_models():
    model_subdir = '_'.join(
        [cfg.MODEL.BACKBONE,
         str(cfg.DATA.DSR_Y),
         str(cfg.DATA.DSR_X)])
    model_path = os.path.join(cfg.MODEL.PATH, cfg.DATA.NAME, model_subdir)
    FLAG, model_name = check_model(model_path)
    return model_name


def get_model():
    model_file = get_models()
    if model_file == 'NO MODEL':
        logger.info('There is no trained model. Please check.')
        os._exit(0)
    else:
        return model_file

