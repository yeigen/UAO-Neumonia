import numpy as np
import pytest
import tensorflow as tf
from conftest import requires_trained_model

from src.grad_cam import compute_heatmap, grad_cam, overlay_heatmap
from src.load_model import model_fun


@pytest.fixture
def tiny_cnn():
    inputs = tf.keras.Input(shape=(32, 32, 1))
    hidden = tf.keras.layers.Conv2D(4, 3, activation="relu", name="last_conv")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(hidden)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(pooled)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def batch():
    rng = np.random.default_rng(42)
    return rng.random((1, 32, 32, 1), dtype=np.float32)


def test_compute_heatmap_returns_conv_spatial_dims(tiny_cnn, batch):
    heatmap = compute_heatmap(tiny_cnn, batch, "last_conv")
    assert heatmap.shape == (30, 30)


def test_compute_heatmap_values_between_0_and_1(tiny_cnn, batch):
    heatmap = compute_heatmap(tiny_cnn, batch, "last_conv")
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_compute_heatmap_handles_all_zero_activations(tiny_cnn):
    for weight in tiny_cnn.get_layer("last_conv").weights:
        weight.assign(tf.zeros_like(weight))
    heatmap = compute_heatmap(tiny_cnn, np.zeros((1, 32, 32, 1)), "last_conv")
    assert not np.isnan(heatmap).any()


def test_overlay_heatmap_returns_uint8_rgb_image():
    heatmap = np.linspace(0, 1, 900, dtype=np.float32).reshape(30, 30)
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    overlay = overlay_heatmap(heatmap, image)
    assert overlay.dtype == np.uint8
    assert overlay.shape == (64, 64, 3)


def test_overlay_heatmap_matches_input_dimensions():
    heatmap = np.zeros((30, 30), dtype=np.float32)
    image = np.zeros((128, 96, 3), dtype=np.uint8)
    assert overlay_heatmap(heatmap, image).shape == (128, 96, 3)


@requires_trained_model
def test_grad_cam_produces_overlay_with_trained_model():
    rng = np.random.default_rng(7)
    array = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    overlay = grad_cam(array, model_fun())
    assert overlay.shape == (512, 512, 3)
    assert overlay.dtype == np.uint8
