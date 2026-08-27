"""Preprocesamiento de imágenes para el modelo."""

import cv2
import numpy as np

from src.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, IMG_SIZE


def preprocess(array):
    """Convierte la imagen al formato que espera el modelo."""
    resized = cv2.resize(array, (IMG_SIZE, IMG_SIZE)) # 512x512
    # Los canales llegan al revés (BGR) porque así los carga la librería, se convierten a 1 canal
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Resaltar la imagen mejorando el contraste
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    equalized = clahe.apply(grayscale)
    normalized = equalized / 255 # Escalar de [0, 255] a [0.0, 1.0]
    # (batch, alto, ancho, canales) (1, 512, 512, 1): estructura para Keras
    return np.expand_dims(normalized, axis=(0, -1))
