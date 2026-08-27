import cv2
import numpy as np
import tensorflow as tf

from src.config import CONV_LAYER_NAME, HEATMAP_INTENSITY, IMG_SIZE
from src.load_model import model_fun
from src.preprocess_img import preprocess


def compute_heatmap(model, batch, layer_name=CONV_LAYER_NAME):
    grad_model = tf.keras.Model(                                               #Crear el modelo grad
        model.inputs, [model.get_layer(layer_name).output, *model.outputs]       #Entrada del modelo original, salida de la capa convolucional y la salida del modelo
    )
    with tf.GradientTape() as tape:              #Registrar operaciones para calcular gradientes para saber cuandto influye cada caractristica de la capa en la predicción
        conv_output, predictions = grad_model([batch])        #Pasar la imagen por el modelo
        class_channel = predictions[:, tf.argmax(predictions[0])]    #Prediccion de la clase que el modelo considera mas probable 
    grads = tape.gradient(class_channel, conv_output)        # Calcular gradientes para saber que caracteristicas fue relevantes para la predicción 
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))       # Obtener valores promedio de los gradientes para la importancia de los mapas
    heatmap = tf.squeeze(conv_output[0] @ pooled_grads[:, tf.newaxis])   #Agregar dimensiones para poder multiplicar y eliminar dimensiones de tamaño 1 para tener imagen, combina las activaciones con los pesos 
    heatmap = tf.maximum(heatmap, 0)  #Quitar valores negativos 
    max_value = tf.reduce_max(heatmap)   #Maximo valor dentro de heatmap
    if max_value > 0:
        heatmap = heatmap / max_value    #Normalización del heatmap [0 - 1]
    return heatmap.numpy()

#Poner el mapa de calor sobre la radiografia 
def overlay_heatmap(heatmap, image, intensity=HEATMAP_INTENSITY): 
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))   #Redimensionar heatmap para tener las mismas dimensiones de la imagen 
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)     # Convertir mapa normalizado [0,1] a 8 bits (0,255])
    transparency = (colored * intensity).astype(np.uint8)  #Modifica la intensidad del heatmap
    combined = cv2.add(transparency, image)
    return combined[:, :, ::-1]   #convertir a RGB para visualización 


def grad_cam(array, model=None):
    if model is None:
        model = model_fun()
    heatmap = compute_heatmap(model, preprocess(array))
    resized_input = cv2.resize(array, (IMG_SIZE, IMG_SIZE))
    return overlay_heatmap(heatmap, resized_input)
