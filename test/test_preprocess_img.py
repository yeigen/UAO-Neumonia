import cv2
import numpy as np
import pytest

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
