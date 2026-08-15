import cv2
import numpy as np


def preprocess(array):
    resized = cv2.resize(array, (512, 512))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(grayscale)
    normalized = equalized / 255
    return np.expand_dims(normalized, axis=(0, -1))
