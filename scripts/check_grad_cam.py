from PIL import Image

from src.config import DATA_DIR, LOGS_DIR
from src.grad_cam import grad_cam
from src.load_model import model_fun
from src.read_img import read_dicom_file, read_jpg_file

DICOM_SAMPLE = str(DATA_DIR / "DICOM" / "viral (2).dcm")
JPG_SAMPLE = str(DATA_DIR / "JPG" / "bacteria" / "person1710_bacteria_4526.jpeg")


def report(label, array, model):
    overlay = grad_cam(array, model)
    print(f"{label}: heatmap overlay {overlay.shape} {overlay.dtype}")
    output = LOGS_DIR / f"check_grad_cam_{label}_heatmap.png"
    Image.fromarray(overlay).save(output)
    print(f"{label}: saved to {output}")


def main():
    LOGS_DIR.mkdir(exist_ok=True)
    model = model_fun()
    array, _ = read_dicom_file(DICOM_SAMPLE)
    report("dicom", array, model)
    array, _ = read_jpg_file(JPG_SAMPLE)
    report("jpg", array, model)


if __name__ == "__main__":
    main()
