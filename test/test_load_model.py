from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.load_model import MODEL_PATH, model_fun

requires_trained_model = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(), reason="trained model file not available"
)


@pytest.fixture
def tiny_model_path(tmp_path):
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)]
    )
    path = str(tmp_path / "tiny.h5")
    model.save(path)
    return path


@pytest.fixture(scope="module")
def trained_model():
    return model_fun()


def test_model_fun_loads_h5_file(tiny_model_path):
    assert isinstance(model_fun(tiny_model_path), tf.keras.Model)


def test_model_fun_loaded_model_predicts(tiny_model_path):
    model = model_fun(tiny_model_path)
    prediction = model.predict(np.zeros((1, 4)), verbose=0)
    assert prediction.shape == (1, 2)


def test_model_fun_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_fun(str(tmp_path / "missing.h5"))


@requires_trained_model
def test_model_fun_loads_trained_model(trained_model):
    assert isinstance(trained_model, tf.keras.Model)


@requires_trained_model
def test_trained_model_expects_preprocessed_batch_shape(trained_model):
    assert trained_model.input_shape == (None, 512, 512, 1)


@requires_trained_model
def test_trained_model_outputs_three_classes(trained_model):
    assert trained_model.output_shape == (None, 3)


@requires_trained_model
def test_trained_model_has_grad_cam_target_layer(trained_model):
    layer_names = [layer.name for layer in trained_model.layers]
    assert "conv10_thisone" in layer_names
