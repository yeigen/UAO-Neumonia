import csv
from pathlib import Path

from PIL import Image

from src.config import HISTORY_CSV_PATH, REPORTS_DIR


def save_result_csv(patient_id, label, probability, path=HISTORY_CSV_PATH):      #Guarda las predicciones en el historial
    Path(path).parent.mkdir(parents=True, exist_ok=True)                         #Crear carpeta reports en caso de no existir
    with open(path, "a", newline="") as history:            #Abre el historial en modo append para agregar los neuvo sreusltados
        writer = csv.writer(history, delimiter="-")                  #Configura el escritor del CSV separando con un "-"
        writer.writerow([patient_id, label, f"{probability:.2f}%"])            #Registra la probabilidad con 2 cifras decimales


def capture_window(widget, output_path):    #Toma captura de la ventana y la guarda
    import pyautogui                        #Libreria para capturas de pantalla 

    widget.update_idletasks()                 #Actualiza la interface para la captura
    region = (                            #Obtiene la posicion y dimensiones de la ventana 
        widget.winfo_rootx(),             #Horizontal
        widget.winfo_rooty(),             #Vertical 
        widget.winfo_width(),             #Ancho
        widget.winfo_height(),            #Altura
    )
    screenshot = pyautogui.screenshot(region=region)  #Captura solo la ventana
    screenshot.save(output_path)  
    return output_path          #Ruta de la imagen genrada


def generate_pdf_report(widget, report_id):             #Genera reporte en PDF
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)      #Crea la carpeta de reportes si no existe 
    image_path = REPORTS_DIR / f"Reporte{report_id}.jpg"    #Definir ruta para el reporte de imagen 
    pdf_path = REPORTS_DIR / f"Reporte{report_id}.pdf"      #Definir ruta para el reporte en PDF
    capture_window(widget, image_path)
    Image.open(image_path).convert("RGB").save(pdf_path)   #Convierte el JPG a RGB para guardarlo como PDF
    return str(pdf_path)                               #Ruta de PDF como una cadena de texto
