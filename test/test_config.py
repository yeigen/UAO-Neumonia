from pathlib import Path

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CLASS_LABELS,
    CONV_LAYER_NAME,
    DATA_DIR,
    HEATMAP_INTENSITY,
    HISTORY_CSV_PATH,
    IMG_SIZE,
    LOGS_DIR,
    MODEL_PATH,
    REPORTS_DIR,
)


def test_paths_are_relative_to_project_root():
    for path in [MODEL_PATH, REPORTS_DIR, HISTORY_CSV_PATH, DATA_DIR, LOGS_DIR]:
        assert isinstance(path, Path)
        assert not path.is_absolute()


def test_model_path_points_to_h5_file():
    assert MODEL_PATH.suffix == ".h5"


def test_history_csv_lives_in_reports_dir():
    assert HISTORY_CSV_PATH.parent == REPORTS_DIR


def test_class_labels_cover_three_classes():
    assert sorted(CLASS_LABELS) == [0, 1, 2]
    assert sorted(CLASS_LABELS.values()) == ["bacteriana", "normal", "viral"]


def test_image_processing_values():
    assert IMG_SIZE > 0
    assert CLAHE_CLIP_LIMIT > 0
    assert len(CLAHE_TILE_GRID_SIZE) == 2
    assert 0 < HEATMAP_INTENSITY <= 1


def test_img_size_is_512():
    assert IMG_SIZE == 512


def test_clahe_clip_limit_value():
    assert CLAHE_CLIP_LIMIT == 2.0


def test_clahe_tile_grid_values_are_positive_ints():
    for value in CLAHE_TILE_GRID_SIZE:
        assert isinstance(value, int)
        assert value > 0


def test_class_labels_keys_are_ints():
    assert all(isinstance(key, int) for key in CLASS_LABELS)


def test_class_labels_values_are_lowercase():
    assert all(value == value.lower() for value in CLASS_LABELS.values())


def test_conv_layer_name_is_nonempty_string():
    assert isinstance(CONV_LAYER_NAME, str)
    assert CONV_LAYER_NAME


def test_heatmap_intensity_is_float():
    assert isinstance(HEATMAP_INTENSITY, float)


def test_model_path_lives_in_models_dir():
    assert MODEL_PATH.parent.name == "models"


def test_model_filename_is_conv_mlp_84():
    assert MODEL_PATH.name == "conv_MLP_84.h5"


def test_history_csv_suffix_is_csv():
    assert HISTORY_CSV_PATH.suffix == ".csv"
