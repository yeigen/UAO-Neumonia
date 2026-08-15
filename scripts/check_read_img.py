from src.config import DATA_DIR, LOGS_DIR
from src.read_img import read_dicom_file, read_jpg_file

DICOM_SAMPLE = str(DATA_DIR / "DICOM" / "normal (3).dcm")
JPG_SAMPLE = str(DATA_DIR / "JPG" / "normal" / "NORMAL2-IM-1144-0001.jpeg")


def report(label, array, preview):
    print(f"{label}: array {array.shape} {array.dtype}, preview {preview.size} mode {preview.mode}")
    output = LOGS_DIR / f"check_read_img_{label}_preview.png"
    preview.save(output)
    print(f"{label}: preview saved to {output}")


def main():
    LOGS_DIR.mkdir(exist_ok=True)
    array, preview = read_dicom_file(DICOM_SAMPLE)
    report("dicom", array, preview)
    array, preview = read_jpg_file(JPG_SAMPLE)
    report("jpg", array, preview)


if __name__ == "__main__":
    main()
