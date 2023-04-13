# supposedly, main file for running everything from one script when we are ready
from src import config
config.init()

# print(ROOT_DIR)

if __name__ == '__main__':
    print(config.ROOT_DIR)