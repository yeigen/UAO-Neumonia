import cv2
import numpy as np
import pytest

from src.config import IMG_SIZE
from src.preprocess_img import preprocess


@pytest.fixture
def bgr_gradient():
    gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)


def test_preprocess_returns_batch_tensor_shape(bgr_gradient):
    assert preprocess(bgr_gradient).shape == (1, 512, 512, 1)


def test_preprocess_resizes_non_square_input():
    image = np.zeros((300, 200, 3), dtype=np.uint8)
    assert preprocess(image).shape == (1, 512, 512, 1)


def test_preprocess_returns_float_values(bgr_gradient):
    assert np.issubdtype(preprocess(bgr_gradient).dtype, np.floating)


def test_preprocess_normalizes_values_between_0_and_1(bgr_gradient):
    batch = preprocess(bgr_gradient)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_preprocess_stretches_contrast_across_range(bgr_gradient):
    batch = preprocess(bgr_gradient)
    assert batch.max() > 0.9
    assert batch.min() < 0.1


def test_preprocess_is_deterministic(bgr_gradient):
    assert np.array_equal(preprocess(bgr_gradient), preprocess(bgr_gradient))


def test_preprocess_batch_dimension_is_one(bgr_gradient):
    assert preprocess(bgr_gradient).shape[0] == 1


def test_preprocess_single_channel_output(bgr_gradient):
    assert preprocess(bgr_gradient).shape[-1] == 1


def test_preprocess_matches_img_size_from_config(bgr_gradient):
    assert preprocess(bgr_gradient).shape == (1, IMG_SIZE, IMG_SIZE, 1)


def test_preprocess_output_is_4d(bgr_gradient):
    assert preprocess(bgr_gradient).ndim == 4


def test_preprocess_accepts_tiny_input():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert preprocess(image).shape == (1, 512, 512, 1)


def test_preprocess_accepts_large_input():
    image = np.zeros((1024, 768, 3), dtype=np.uint8)
    assert preprocess(image).shape == (1, 512, 512, 1)


def test_preprocess_handles_wide_rectangular_input():
    image = np.zeros((150, 400, 3), dtype=np.uint8)
    assert preprocess(image).shape == (1, 512, 512, 1)


def test_preprocess_uniform_input_stays_uniform():
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    batch = preprocess(image)
    assert batch.max() == batch.min()


def test_preprocess_does_not_modify_input(bgr_gradient):
    original = bgr_gradient.copy()
    preprocess(bgr_gradient)
    assert np.array_equal(bgr_gradient, original)


def test_preprocess_values_are_finite(bgr_gradient):
    assert np.isfinite(preprocess(bgr_gradient)).all()


def test_preprocess_output_is_numpy_array(bgr_gradient):
    assert isinstance(preprocess(bgr_gradient), np.ndarray)
