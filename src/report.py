import csv
from pathlib import Path

from PIL import Image

from src.config import HISTORY_CSV_PATH, REPORTS_DIR


def save_result_csv(patient_id, label, probability, path=HISTORY_CSV_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as history:
        writer = csv.writer(history, delimiter="-")
        writer.writerow([patient_id, label, f"{probability:.2f}%"])


def capture_window(widget, output_path):
    import pyautogui

    widget.update_idletasks()
    region = (
        widget.winfo_rootx(),
        widget.winfo_rooty(),
        widget.winfo_width(),
        widget.winfo_height(),
    )
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(output_path)
    return output_path


def generate_pdf_report(widget, report_id):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / f"Reporte{report_id}.jpg"
    pdf_path = REPORTS_DIR / f"Reporte{report_id}.pdf"
    capture_window(widget, image_path)
    Image.open(image_path).convert("RGB").save(pdf_path)
    return str(pdf_path)
