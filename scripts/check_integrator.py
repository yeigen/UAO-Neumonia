from src.config import DATA_DIR
from src.integrator import predict
from src.load_model import model_fun
from src.read_img import read_dicom_file, read_jpg_file


def main():
    model = model_fun()
    for path in sorted((DATA_DIR / "DICOM").glob("*.dcm")):
        array, _ = read_dicom_file(str(path))
        label, probability, _ = predict(array, model)
        print(f"{path.name}: {label} ({probability:.2f}%)")
    for path in sorted((DATA_DIR / "JPG").rglob("*.jpeg"))[:6]:
        array, _ = read_jpg_file(str(path))
        label, probability, _ = predict(array, model)
        print(f"{path.relative_to(DATA_DIR / 'JPG')}: {label} ({probability:.2f}%)")


if __name__ == "__main__":
    main()
