import cv2
import numpy as np
import pydicom
from PIL import Image

def scale_to_uint8(array):
    scaled = (np.maximum(array, 0) / array.max()) * 255.0
    return scaled.astype(np.uint8)


def read_dicom_file(path):
    pixel_array = pydicom.dcmread(path).pixel_array
    display_image = Image.fromarray(pixel_array)
    grayscale = scale_to_uint8(pixel_array.astype(float))
    rgb_array = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    return rgb_array, display_image


def read_jpg_file(path):
    image_array = np.asarray(cv2.imread(path))
    display_image = Image.fromarray(image_array)
    return scale_to_uint8(image_array.astype(float)), display_image
