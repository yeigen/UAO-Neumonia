import csv

from PIL import Image

CSV_PATH = "historial.csv"

def save_result_csv(patient_id, label, probability, path=CSV_PATH):
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
    image_path = f"Reporte{report_id}.jpg"
    pdf_path = f"Reporte{report_id}.pdf"
    capture_window(widget, image_path)
    Image.open(image_path).convert("RGB").save(pdf_path)
    return pdf_path
