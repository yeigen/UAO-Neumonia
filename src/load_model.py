import tensorflow as tf

from src.config import MODEL_PATH


def model_fun(path=MODEL_PATH):
    return tf.keras.models.load_model(path, compile=False)
