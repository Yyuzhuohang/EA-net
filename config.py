#import tensorflow as tf
import os
import glob
import json
import numpy as np

train_path="./data/data_raw"

stru_config={
             "embedding_size":64,
             "attention_size":64,
             "which":"tanh",
             "res_size":32,
             "num_class":3
             }


train_config={
        "train_keep":0.5,
        "test_keep":1.0,
        'CKPT':'ckpt',
        "new_train":True,
        "BATCHSIZE":256,
        "MAX_ITER":100000,
        'STEP_SHOW':10,
        'STEP_SAVE':50,
        "LEARNING_RATE":0.00001,
        'STOP_THRESHOLD':8000,
        'early_stop':False,
        "save_best_ckpt":False
}
##########################################################

