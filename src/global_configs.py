# may be needed to define global variables and whatnot
import os
import numpy as np
import random
def init():
    global ROOT_DIR
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # as seed we set the year of our birth date
    SEED = random.seed(1999)
    NUMPY_SEED = np.random.seed(1999)
