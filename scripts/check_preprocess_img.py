import numpy as np
from PIL import Image

from src.config import DATA_DIR, LOGS_DIR
from src.preprocess_img import preprocess
from src.read_img import read_dicom_file, read_jpg_file

DICOM_SAMPLE = str(DATA_DIR / "DICOM" / "normal (3).dcm")
JPG_SAMPLE = str(DATA_DIR / "JPG" / "bacteria" / "person1710_bacteria_4526.jpeg")


def report(label, array):
    batch = preprocess(array)
    stats = f"range [{batch.min():.3f}, {batch.max():.3f}]"
    print(f"{label}: batch {batch.shape} {batch.dtype}, {stats}")
    equalized = (batch[0, :, :, 0] * 255).astype(np.uint8)
    output = LOGS_DIR / f"check_preprocess_{label}_clahe.png"
    Image.fromarray(equalized).save(output)
    print(f"{label}: CLAHE result saved to {output}")


def main():
    LOGS_DIR.mkdir(exist_ok=True)
    array, _ = read_dicom_file(DICOM_SAMPLE)
    report("dicom", array)
    array, _ = read_jpg_file(JPG_SAMPLE)
    report("jpg", array)


if __name__ == "__main__":
    main()
