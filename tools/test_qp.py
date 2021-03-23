from qpnet.utils.config import setup_env
from qpnet.utils.tester import test
from qpnet.helper.writer import write_results
from qpnet.helper.writer import calculate_results


def main():
    setup_env(description='Test a QPNet.', gpus='0, 1, 2, 3, 4, 5, 6, 7')
    bboxes, x_val_path, model_name = test(mode='test')
    write_results(x_val_path, bboxes, model_name)
    calculate_results(model_name)


if __name__ == '__main__':
    main()
