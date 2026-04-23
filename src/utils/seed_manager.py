import random
import os
import numpy as np
import tensorflow as tf

def set_deterministic_seed(seed_value):
    
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"[Seed Manager] Random seed {seed_value} olarak sabitlendi ve sistem deterministik hale getirildi.")