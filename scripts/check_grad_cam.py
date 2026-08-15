from pathlib import Path

from PIL import Image

from src.grad_cam import grad_cam
from src.load_model import model_fun
from src.read_img import read_dicom_file, read_jpg_file

DICOM_SAMPLE = "data/DICOM/viral (2).dcm"
JPG_SAMPLE = "data/JPG/bacteria/person1710_bacteria_4526.jpeg"
OUTPUT_DIR = Path("logs")


def report(label, array, model):
    overlay = grad_cam(array, model)
    print(f"{label}: heatmap overlay {overlay.shape} {overlay.dtype}")
    output = OUTPUT_DIR / f"check_grad_cam_{label}_heatmap.png"
    Image.fromarray(overlay).save(output)
    print(f"{label}: saved to {output}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = model_fun()
    array, _ = read_dicom_file(DICOM_SAMPLE)
    report("dicom", array, model)
    array, _ = read_jpg_file(JPG_SAMPLE)
    report("jpg", array, model)


if __name__ == "__main__":
    main()
