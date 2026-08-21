import cv2
import numpy as np
import pydicom
from PIL import Image

# NOTAS

# DICOM viene en 1 canal 
# JPG viene en 3 canales

# recibe un array escalado a 8 bits (una sola función en vez de duplicarla en las imagenes DICOM y JPEG)
def scale_to_uint8(array):
    scaled = (np.maximum(array, 0) / array.max()) * 255.0
    return scaled.astype(np.uint8) # Escala 8 bits [0, 255]


def read_dicom_file(path):
    pixel_array = pydicom.dcmread(path).pixel_array # abrir la imagen como un Array
    display_image = Image.fromarray(pixel_array) # convertir la imagen a PIL
    grayscale = scale_to_uint8(pixel_array.astype(float))
    rgb_array = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB) # de gris a "color" de 1 canal a RGB (la imagen igual es gris)
    return rgb_array, display_image # una tupla, el array y la imagen (para que el modelo pueda procesar y para que podamos ver)

# aqui se aplica lo mismo que arriba pero la imagen ya es jpeg por eso no tiene que pasar por alguno de los procesos
def read_jpg_file(path):
    image_array = np.asarray(cv2.imread(path))
    display_image = Image.fromarray(image_array)
    return scale_to_uint8(image_array.astype(float)), display_image

# elige que funcion se le va a aplicar segun el tipo de imagen
def read_image_file(path):
    if path.lower().endswith(".dcm"):
        return read_dicom_file(path)
    return read_jpg_file(path)
