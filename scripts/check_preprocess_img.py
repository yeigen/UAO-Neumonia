from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess_img import preprocess
from src.read_img import read_dicom_file, read_jpg_file

DICOM_SAMPLE = "data/DICOM/normal (3).dcm"
JPG_SAMPLE = "data/JPG/bacteria/person1710_bacteria_4526.jpeg"
OUTPUT_DIR = Path("logs")


def report(label, array):
    batch = preprocess(array)
    print(f"{label}: batch {batch.shape} {batch.dtype}, range [{batch.min():.3f}, {batch.max():.3f}]")
    equalized = (batch[0, :, :, 0] * 255).astype(np.uint8)
    output = OUTPUT_DIR / f"check_preprocess_{label}_clahe.png"
    Image.fromarray(equalized).save(output)
    print(f"{label}: CLAHE result saved to {output}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    array, _ = read_dicom_file(DICOM_SAMPLE)
    report("dicom", array)
    array, _ = read_jpg_file(JPG_SAMPLE)
    report("jpg", array)


if __name__ == "__main__":
    main()
