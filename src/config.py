from pathlib import Path    #Trabajar de manera compatible en los sistemas operativos

MODEL_PATH = Path("models/conv_MLP_84.h5")  #Ruta del modelo entrenado

REPORTS_DIR = Path("reports")     #Ruta de los reportes de Tkinter
HISTORY_CSV_PATH = REPORTS_DIR / "historial.csv"   #Ruta del CSV con las predicciones

#Directorios de datos y los logs
DATA_DIR = Path("data")  
LOGS_DIR = Path("logs")

IMG_SIZE = 512   #Tamaño de las imagenes que espera recibir el modelo
CLAHE_CLIP_LIMIT = 2.0   #Mejora de contraste
CLAHE_TILE_GRID_SIZE = (4, 4)  #regiones utilizadas por CLAHE

CLASS_LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}  #Clases
CONV_LAYER_NAME = "conv10_thisone"  #Capa convolucional generadora de mapas para la activación del modelo
HEATMAP_INTENSITY = 0.8   #Intensidad del heatmap
