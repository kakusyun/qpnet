from qpnet.utils.config import setup_env
from qpnet.utils.trainer import train


def main():
    setup_env(description='Train a QPNet.', gpus='0, 1, 2, 3, 4, 5, 6, 7')
    train()


if __name__ == '__main__':
    main()
