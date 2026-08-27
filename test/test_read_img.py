from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.read_img import (
    read_dicom_file,
    read_image_file,
    read_jpg_file,
    scale_to_uint8,
)


def test_scale_to_uint8_returns_uint8_dtype():
    result = scale_to_uint8(np.array([0.0, 50.0, 100.0]))
    assert result.dtype == np.uint8


def test_scale_to_uint8_stretches_max_to_255():
    result = scale_to_uint8(np.array([0.0, 50.0, 100.0]))
    assert result.max() == 255


def test_scale_to_uint8_clips_negatives_to_zero():
    result = scale_to_uint8(np.array([-40.0, 0.0, 80.0]))
    assert result.min() == 0


def test_scale_to_uint8_preserves_shape():
    array = np.random.rand(16, 24) * 90
    assert scale_to_uint8(array).shape == (16, 24)


def test_read_dicom_file_returns_rgb_uint8_array(sample_dicom):
    array, _ = read_dicom_file(sample_dicom)
    assert array.dtype == np.uint8
    assert array.shape == (8, 8, 3)


def test_read_dicom_file_scales_intensities_to_255(sample_dicom):
    array, _ = read_dicom_file(sample_dicom)
    assert array.max() == 255


def test_read_dicom_file_returns_pil_preview(sample_dicom):
    _, preview = read_dicom_file(sample_dicom)
    assert isinstance(preview, Image.Image)
    assert preview.size == (8, 8)


def test_read_dicom_file_replicates_grayscale_across_channels(sample_dicom):
    array, _ = read_dicom_file(sample_dicom)
    assert np.array_equal(array[:, :, 0], array[:, :, 1])
    assert np.array_equal(array[:, :, 1], array[:, :, 2])


def test_read_jpg_file_returns_uint8_array(sample_jpg):
    array, _ = read_jpg_file(sample_jpg)
    assert array.dtype == np.uint8


def test_read_jpg_file_preserves_dimensions(sample_jpg):
    array, _ = read_jpg_file(sample_jpg)
    assert array.shape == (32, 48, 3)


def test_read_jpg_file_scales_intensities_to_255(sample_jpg):
    array, _ = read_jpg_file(sample_jpg)
    assert array.max() == 255


def test_read_jpg_file_returns_pil_preview(sample_jpg):
    _, preview = read_jpg_file(sample_jpg)
    assert isinstance(preview, Image.Image)
    assert preview.size == (48, 32)


def test_read_image_file_routes_dicom_by_extension(sample_dicom):
    array, _ = read_image_file(sample_dicom)
    assert array.shape == (8, 8, 3)


def test_read_image_file_routes_dicom_with_uppercase_extension(
    tmp_path, gradient_pixels, sample_dicom
):
    uppercase_path = tmp_path / "SAMPLE.DCM"
    uppercase_path.write_bytes(Path(sample_dicom).read_bytes())
    array, _ = read_image_file(str(uppercase_path))
    assert array.shape == (8, 8, 3)


def test_read_image_file_routes_jpg_by_extension(sample_jpg):
    array, _ = read_image_file(sample_jpg)
    assert array.shape == (32, 48, 3)


def test_scale_to_uint8_uniform_positive_maps_to_255():
    result = scale_to_uint8(np.full((4, 4), 37.0))
    assert np.all(result == 255)


def test_scale_to_uint8_keeps_values_within_uint8_range():
    result = scale_to_uint8(np.array([-10.0, 5.0, 500.0]))
    assert result.min() >= 0
    assert result.max() <= 255


def test_scale_to_uint8_preserves_relative_order():
    result = scale_to_uint8(np.array([10.0, 20.0, 30.0]))
    assert result[0] < result[1] < result[2]


def test_scale_to_uint8_maps_known_values():
    result = scale_to_uint8(np.array([0.0, 50.0, 100.0]))
    assert list(result) == [0, 127, 255]


def test_scale_to_uint8_preserves_3d_shape():
    array = np.random.rand(4, 6, 3) * 80
    assert scale_to_uint8(array).shape == (4, 6, 3)


def test_read_dicom_file_returns_tuple_of_two(sample_dicom):
    result = read_dicom_file(sample_dicom)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_read_dicom_file_preview_is_grayscale(sample_dicom):
    _, preview = read_dicom_file(sample_dicom)
    assert preview.mode == "L"


def test_read_dicom_file_min_intensity_is_zero(sample_dicom):
    array, _ = read_dicom_file(sample_dicom)
    assert array.min() == 0


def test_read_jpg_file_returns_tuple_of_two(sample_jpg):
    result = read_jpg_file(sample_jpg)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_read_jpg_file_preview_is_rgb(sample_jpg):
    _, preview = read_jpg_file(sample_jpg)
    assert preview.mode == "RGB"


def test_read_jpg_file_uniform_image_becomes_white(sample_jpg):
    array, _ = read_jpg_file(sample_jpg)
    assert array.min() == 255


def test_read_image_file_routes_jpeg_extension(tmp_path):
    path = tmp_path / "sample.jpeg"
    cv2.imwrite(str(path), np.full((16, 16, 3), 90, dtype=np.uint8))
    array, _ = read_image_file(str(path))
    assert array.shape == (16, 16, 3)


def test_read_image_file_routes_png_extension(tmp_path):
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), np.full((16, 16, 3), 90, dtype=np.uint8))
    array, _ = read_image_file(str(path))
    assert array.shape == (16, 16, 3)


def test_read_image_file_returns_pil_preview_for_each_format(sample_dicom, sample_jpg):
    _, dicom_preview = read_image_file(sample_dicom)
    _, jpg_preview = read_image_file(sample_jpg)
    assert isinstance(dicom_preview, Image.Image)
    assert isinstance(jpg_preview, Image.Image)


def test_read_image_file_routes_mixed_case_dcm(tmp_path, sample_dicom):
    mixed_path = tmp_path / "sample_mixed.DcM"
    mixed_path.write_bytes(Path(sample_dicom).read_bytes())
    array, _ = read_image_file(str(mixed_path))
    assert array.shape == (8, 8, 3)
