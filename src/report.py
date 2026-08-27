"""Historial de predicciones y reportes en PDF."""

import csv
from pathlib import Path

from PIL import Image

from src.config import HISTORY_CSV_PATH, REPORTS_DIR


def save_result_csv(patient_id, label, probability, path=HISTORY_CSV_PATH):
    """Agrega la predicción al historial CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  #Crear la carpeta reports si no existe
    #Abre el historial en modo append para agregar los nuevos resultados
    with open(path, "a", newline="") as history:
        writer = csv.writer(history, delimiter="-")  #Escritor del CSV separando con "-"
        writer.writerow([patient_id, label, f"{probability:.2f}%"])  #Probabilidad con 2 decimales


def capture_window(widget, output_path):
    """Toma captura de la ventana y la guarda como imagen."""
    import pyautogui  #Libreria para capturas de pantalla

    widget.update_idletasks()                 #Actualiza la interfaz para la captura
    region = (                            #Obtiene la posicion y dimensiones de la ventana
        widget.winfo_rootx(),             #Horizontal
        widget.winfo_rooty(),             #Vertical
        widget.winfo_width(),             #Ancho
        widget.winfo_height(),            #Altura
    )
    screenshot = pyautogui.screenshot(region=region)  #Captura solo la ventana
    screenshot.save(output_path)
    return output_path          #Ruta de la imagen generada


def generate_pdf_report(widget, report_id):
    """Genera el reporte de la ventana en JPG y PDF."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)      #Crea la carpeta de reportes si no existe
    image_path = REPORTS_DIR / f"Reporte{report_id}.jpg"    #Definir ruta para el reporte de imagen
    pdf_path = REPORTS_DIR / f"Reporte{report_id}.pdf"      #Definir ruta para el reporte en PDF
    capture_window(widget, image_path)
    #Convierte el JPG a RGB para guardarlo como PDF
    Image.open(image_path).convert("RGB").save(pdf_path)
    return str(pdf_path)                               #Ruta del PDF como una cadena de texto
