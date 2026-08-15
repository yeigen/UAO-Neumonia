import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

MODEL_PATH = "models/conv_MLP_84.h5"


def model_fun(path=MODEL_PATH):
    return tf.keras.models.load_model(path, compile=False)
