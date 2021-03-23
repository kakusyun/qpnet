from qpnet.utils.loader import infer_files_loader
from qpnet.utils.tester import test
from qpnet.helper.writer import write_results
from qpnet.utils.config import setup_env
from qpnet.utils.config import cfg


def main():
    setup_env(description='Infer a QPNet.', gpus='7')
    x_infer_path = infer_files_loader(cfg.INFER_PATH)
    boxes, _, _ = test(x_infer_path, mode='infer')
    write_results(x_infer_path, boxes, mode='infer')


if __name__ == '__main__':
    main()
