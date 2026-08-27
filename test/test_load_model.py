import inspect
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from conftest import requires_trained_model

from src.config import MODEL_PATH
from src.load_model import model_fun


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


def test_model_fun_default_path_is_config_model_path():
    default = inspect.signature(model_fun).parameters["path"].default
    assert default == MODEL_PATH


def test_model_fun_accepts_path_object(tiny_model_path):
    assert isinstance(model_fun(Path(tiny_model_path)), tf.keras.Model)


def test_model_fun_preserves_layer_count(tiny_model_path):
    assert len(model_fun(tiny_model_path).layers) == 1


def test_model_fun_preserves_input_shape(tiny_model_path):
    assert model_fun(tiny_model_path).input_shape == (None, 4)


def test_model_fun_preserves_weights(tiny_model_path):
    first = model_fun(tiny_model_path)
    second = model_fun(tiny_model_path)
    for left, right in zip(first.get_weights(), second.get_weights(), strict=True):
        assert np.array_equal(left, right)


def test_model_fun_loaded_model_is_uncompiled(tmp_path):
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)]
    )
    model.compile(optimizer="adam", loss="mse")
    path = str(tmp_path / "compiled.h5")
    model.save(path)
    assert getattr(model_fun(path), "optimizer", None) is None


def test_model_fun_returns_new_instance_each_call(tiny_model_path):
    assert model_fun(tiny_model_path) is not model_fun(tiny_model_path)


def test_model_fun_matches_original_predictions(tmp_path):
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)]
    )
    path = str(tmp_path / "original.h5")
    model.save(path)
    batch = np.ones((1, 4))
    expected = model.predict(batch, verbose=0)
    loaded = model_fun(path).predict(batch, verbose=0)
    assert np.allclose(expected, loaded)
