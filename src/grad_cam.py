import cv2
import numpy as np
import tensorflow as tf

from src.config import CONV_LAYER_NAME, HEATMAP_INTENSITY, IMG_SIZE
from src.load_model import model_fun
from src.preprocess_img import preprocess


def compute_heatmap(model, batch, layer_name=CONV_LAYER_NAME):
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(layer_name).output, *model.outputs]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([batch])
        class_channel = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_output[0] @ pooled_grads[:, tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(heatmap, image, intensity=HEATMAP_INTENSITY):
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    transparency = (colored * intensity).astype(np.uint8)
    combined = cv2.add(transparency, image)
    return combined[:, :, ::-1]


def grad_cam(array, model=None):
    if model is None:
        model = model_fun()
    heatmap = compute_heatmap(model, preprocess(array))
    resized_input = cv2.resize(array, (IMG_SIZE, IMG_SIZE))
    return overlay_heatmap(heatmap, resized_input)
