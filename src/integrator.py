import numpy as np

from src.config import CLASS_LABELS
from src.grad_cam import grad_cam
from src.load_model import model_fun
from src.preprocess_img import preprocess


def predict(array, model=None):
    if model is None:
        model = model_fun()
    probabilities = model.predict(preprocess(array), verbose=0)[0]
    label = CLASS_LABELS[int(np.argmax(probabilities))]
    probability = float(probabilities.max()) * 100
    heatmap = grad_cam(array, model)
    return label, probability, heatmap
