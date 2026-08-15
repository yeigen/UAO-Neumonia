import cv2
import numpy as np

from src.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, IMG_SIZE


def preprocess(array):
    resized = cv2.resize(array, (IMG_SIZE, IMG_SIZE))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    equalized = clahe.apply(grayscale)
    normalized = equalized / 255
    return np.expand_dims(normalized, axis=(0, -1))
