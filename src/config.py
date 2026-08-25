from pathlib import Path    

MODEL_PATH = Path("models/conv_MLP_84.h5")  

REPORTS_DIR = Path("reports")     #Ruta de los reportes de Tkinter
HISTORY_CSV_PATH = REPORTS_DIR / "historial.csv"

DATA_DIR = Path("data")
LOGS_DIR = Path("logs")

IMG_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0   
CLAHE_TILE_GRID_SIZE = (4, 4)

CLASS_LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}
CONV_LAYER_NAME = "conv10_thisone"
HEATMAP_INTENSITY = 0.8
