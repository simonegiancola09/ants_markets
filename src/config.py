# may be needed to define global variables and whatnot
import os
def init():
    global ROOT_DIR
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
