"""Generación de mapas de calor Grad-CAM."""

import cv2
import numpy as np
import tensorflow as tf

from src.config import CONV_LAYER_NAME, HEATMAP_INTENSITY, IMG_SIZE
from src.load_model import model_fun
from src.preprocess_img import preprocess


def compute_heatmap(model, batch, layer_name=CONV_LAYER_NAME):
    """Calcula el mapa de calor Grad-CAM normalizado [0, 1] para un batch."""
    #Crear el modelo grad: entrada original, salida de la capa convolucional y salida del modelo
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(layer_name).output, *model.outputs]
    )
    #Registrar operaciones para calcular cuánto influye cada característica en la predicción
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([batch])  #Pasar la imagen por el modelo
        class_channel = predictions[:, tf.argmax(predictions[0])]  #Clase más probable
    #Calcular gradientes para saber qué características fueron relevantes para la predicción
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  #Promedio de los gradientes por mapa
    #Combina las activaciones con los pesos y quita dimensiones de tamaño 1 para tener la imagen
    heatmap = tf.squeeze(conv_output[0] @ pooled_grads[:, tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)  #Quitar valores negativos
    max_value = tf.reduce_max(heatmap)  #Máximo valor dentro del heatmap
    if max_value > 0:
        heatmap = heatmap / max_value  #Normalización del heatmap [0:1]
    return heatmap.numpy()


def overlay_heatmap(heatmap, image, intensity=HEATMAP_INTENSITY):
    """Pone el mapa de calor sobre la radiografía."""
    #Redimensionar el heatmap a las dimensiones de la imagen
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    #Convertir el mapa normalizado [0, 1] a 8 bits [0, 255] con el colormap JET
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    transparency = (colored * intensity).astype(np.uint8)  #Modifica la intensidad del heatmap
    combined = cv2.add(transparency, image)
    return combined[:, :, ::-1]  #Convertir a RGB para visualización


def grad_cam(array, model=None):
    """Genera la radiografía con el heatmap Grad-CAM superpuesto."""
    if model is None:  #Si no se proporciona un modelo se cargará automáticamente
        model = model_fun()
    heatmap = compute_heatmap(model, preprocess(array))  #Preprocesa y calcula el mapa gradcam
    resized_input = cv2.resize(array, (IMG_SIZE, IMG_SIZE))
    return overlay_heatmap(heatmap, resized_input)  #Sobrepone el heatmap para el resultado visual
