"""Lectura de imágenes DICOM y JPG."""

import cv2
import numpy as np
import pydicom
from PIL import Image

# NOTAS

# DICOM viene en 1 canal
# JPG viene en 3 canales


def scale_to_uint8(array):
    """Escala un array a 8 bits [0, 255], compartida entre DICOM y JPG."""
    scaled = (np.maximum(array, 0) / array.max()) * 255.0
    return scaled.astype(np.uint8)


def read_dicom_file(path):
    """Lee un DICOM y regresa el array RGB para el modelo y la imagen PIL para mostrar."""
    pixel_array = pydicom.dcmread(path).pixel_array # abrir la imagen como un Array
    display_image = Image.fromarray(pixel_array) # convertir la imagen a PIL
    grayscale = scale_to_uint8(pixel_array.astype(float))
    # de gris a "color", de 1 canal a RGB (la imagen igual es gris)
    rgb_array = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    return rgb_array, display_image


def read_jpg_file(path):
    """Lee un JPG y regresa el array escalado y la imagen PIL; ya viene en 3 canales."""
    image_array = np.asarray(cv2.imread(path))
    display_image = Image.fromarray(image_array)
    return scale_to_uint8(image_array.astype(float)), display_image


def read_image_file(path):
    """Elige el lector según la extensión del archivo."""
    if path.lower().endswith(".dcm"):
        return read_dicom_file(path)
    return read_jpg_file(path)
