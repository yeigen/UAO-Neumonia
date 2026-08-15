import cv2
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from src.config import MODEL_PATH

requires_trained_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model file not available"
)


def build_dicom_file(path, pixels):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(str(path), enforce_file_format=True)
    return str(path)


@pytest.fixture
def gradient_pixels():
    return np.arange(64, dtype=np.uint8).reshape(8, 8)


@pytest.fixture
def sample_dicom(tmp_path, gradient_pixels):
    return build_dicom_file(tmp_path / "sample.dcm", gradient_pixels)


@pytest.fixture
def sample_jpg(tmp_path):
    image = np.full((32, 48, 3), 128, dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), image)
    return str(path)
