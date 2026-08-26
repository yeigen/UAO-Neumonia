import tensorflow as tf

from src.config import MODEL_PATH  #Importar ruta del modelo

#Cargar el modelo sin cambiar su configuración
def model_fun(path=MODEL_PATH):
    return tf.keras.models.load_model(path, compile=False) # Cargamos el modelo sin reentrenar
