# supposedly, main file for running everything from one script when we are ready
from src import config
config.init()


if __name__ == '__main__':
    # an attempt that works, it prints ROOT_DIR which is a global variable
    print(config.ROOT_DIR)