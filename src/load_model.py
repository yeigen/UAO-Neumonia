"""Carga del modelo entrenado."""

import tensorflow as tf

from src.config import MODEL_PATH


def model_fun(path=MODEL_PATH):
    """Carga el modelo sin recompilarlo ni cambiar su configuración."""
    return tf.keras.models.load_model(path, compile=False)
