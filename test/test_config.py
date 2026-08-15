from pathlib import Path

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CLASS_LABELS,
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
