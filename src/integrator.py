"""Une lectura, preprocesamiento, modelo y Grad-CAM para dar la predicción completa."""

import numpy as np

from src.config import CLASS_LABELS
from src.grad_cam import grad_cam
from src.load_model import model_fun
from src.preprocess_img import preprocess


def predict(array, model=None):
    """Predice sobre la imagen y regresa etiqueta, probabilidad y heatmap."""
    if model is None:
        model = model_fun()
    #Obtiene las probabilidades de predicción de cada clase
    probabilities = model.predict(preprocess(array), verbose=0)[0]
    #Obtiene la clase con mayor probabilidad y le asigna la etiqueta
    label = CLASS_LABELS[int(np.argmax(probabilities))]
    probability = float(probabilities.max()) * 100
    heatmap = grad_cam(array, model)      #Genera el mapa gradcam
    return label, probability, heatmap    #Clase predicha, probabilidad y mapa de calor gradcam
