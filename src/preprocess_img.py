import cv2
import numpy as np

from src.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, IMG_SIZE

#transformaciones
def preprocess(array):
    resized = cv2.resize(array, (IMG_SIZE, IMG_SIZE)) # 512x512
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # los canales se ordenan al reves porque asi los carga la libreria
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE) # resaltar la imagen
    equalized = clahe.apply(grayscale)
    normalized = equalized / 255 # escalar a de [0, 255] a [0.0, 1.0]
    return np.expand_dims(normalized, axis=(0, -1)) # (batch, alto, ancho, canales) (1 [lote], 512, 512, 1 [canales])
