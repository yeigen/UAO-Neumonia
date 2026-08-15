import numpy as np
import pytest
import tensorflow as tf
from conftest import requires_trained_model

from src.integrator import CLASS_LABELS, predict
from src.load_model import model_fun


@pytest.fixture(scope="module")
def tiny_pneumonia_model():
    inputs = tf.keras.Input(shape=(512, 512, 1))
    hidden = tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv10_thisone")(
        inputs
    )
    pooled = tf.keras.layers.GlobalAveragePooling2D()(hidden)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(pooled)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(3)
    return (rng.random((256, 256, 3)) * 255).astype(np.uint8)


def test_class_labels_cover_the_three_diagnoses():
    assert sorted(CLASS_LABELS.values()) == ["bacteriana", "normal", "viral"]


def test_predict_returns_known_label(tiny_pneumonia_model, bgr_image):
    label, _, _ = predict(bgr_image, tiny_pneumonia_model)
    assert label in CLASS_LABELS.values()


def test_predict_probability_is_percentage(tiny_pneumonia_model, bgr_image):
    _, probability, _ = predict(bgr_image, tiny_pneumonia_model)
    assert 0.0 <= probability <= 100.0


def test_predict_returns_display_ready_heatmap(tiny_pneumonia_model, bgr_image):
    _, _, heatmap = predict(bgr_image, tiny_pneumonia_model)
    assert heatmap.shape == (512, 512, 3)
    assert heatmap.dtype == np.uint8


@requires_trained_model
def test_predict_works_with_trained_model(bgr_image):
    label, probability, heatmap = predict(bgr_image, model_fun())
    assert label in CLASS_LABELS.values()
    assert 0.0 <= probability <= 100.0
    assert heatmap.shape == (512, 512, 3)
