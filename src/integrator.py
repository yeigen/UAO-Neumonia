import numpy as np

from src.config import CLASS_LABELS
from src.grad_cam import grad_cam
from src.load_model import model_fun
from src.preprocess_img import preprocess


def predict(array, model=None):   #Recibir imagen para realizar predicción
    if model is None:
        model = model_fun()
    probabilities = model.predict(preprocess(array), verbose=0)[0]   #Obtiene probabilidades de predicción de cada clase
    label = CLASS_LABELS[int(np.argmax(probabilities))]    #Obtiene la clase con mayor probabilidad y asigna la etiqueta
    probability = float(probabilities.max()) * 100
    heatmap = grad_cam(array, model)      #Genera el mapa gradcam
    return label, probability, heatmap    #Clase predicha, probabilidad y mapa de calor gradcam
