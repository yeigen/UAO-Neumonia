import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.report import capture_window, generate_pdf_report, save_result_csv


def install_fake_screenshot(monkeypatch, regions=None):
    def fake_screenshot(region):
        if regions is not None:
            regions.append(region)
        return Image.new("RGB", (region[2], region[3]), "white")

    fake_pyautogui = SimpleNamespace(screenshot=fake_screenshot)
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)


class FakeWidget:
    def update_idletasks(self):
        pass

    def winfo_rootx(self):
        return 10

    def winfo_rooty(self):
        return 20

    def winfo_width(self):
        return 300

    def winfo_height(self):
        return 200


def test_save_result_csv_writes_patient_row(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("123", "viral", 95.314, path)
    assert path.read_text().strip() == "123-viral-95.31%"


def test_save_result_csv_appends_rows(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("1", "normal", 99.0, path)
    save_result_csv("2", "bacteriana", 80.5, path)
    lines = path.read_text().strip().splitlines()
    assert lines == ["1-normal-99.00%", "2-bacteriana-80.50%"]


def test_save_result_csv_creates_parent_folder(tmp_path):
    path = tmp_path / "reports" / "history.csv"
    save_result_csv("9", "viral", 70.0, path)
    assert path.read_text().strip() == "9-viral-70.00%"


def test_capture_window_uses_widget_geometry(monkeypatch, tmp_path):
    regions = []
    install_fake_screenshot(monkeypatch, regions)
    output = str(tmp_path / "capture.jpg")
    capture_window(FakeWidget(), output)
    assert regions == [(10, 20, 300, 200)]
    assert Path(output).exists()


def test_generate_pdf_report_creates_pdf(monkeypatch, tmp_path):
    install_fake_screenshot(monkeypatch)
    monkeypatch.chdir(tmp_path)
    pdf_path = generate_pdf_report(FakeWidget(), 0)
    assert pdf_path == "reports/Reporte0.pdf"
    assert Path(pdf_path).read_bytes().startswith(b"%PDF")


def test_generate_pdf_report_numbers_reports(monkeypatch, tmp_path):
    install_fake_screenshot(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert generate_pdf_report(FakeWidget(), 3) == "reports/Reporte3.pdf"
    assert Path("reports/Reporte3.jpg").exists()


def test_save_result_csv_uses_dash_delimiter(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("77", "normal", 88.0, path)
    assert path.read_text().strip().split("-") == ["77", "normal", "88.00%"]


def test_save_result_csv_formats_two_decimals(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("1", "viral", 5.0, path)
    assert path.read_text().strip().endswith("5.00%")


def test_save_result_csv_rounds_probability(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("1", "viral", 95.999, path)
    assert path.read_text().strip().endswith("96.00%")


def test_save_result_csv_accepts_string_path(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("5", "normal", 50.0, str(path))
    assert path.exists()


def test_save_result_csv_preserves_existing_content(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("0-bacteriana-10.00%\n")
    save_result_csv("1", "viral", 20.0, path)
    lines = path.read_text().strip().splitlines()
    assert lines == ["0-bacteriana-10.00%", "1-viral-20.00%"]


def test_save_result_csv_writes_one_row_per_call(tmp_path):
    path = tmp_path / "history.csv"
    for patient in ["1", "2", "3"]:
        save_result_csv(patient, "normal", 60.0, path)
    assert len(path.read_text().strip().splitlines()) == 3


def test_save_result_csv_allows_empty_patient_id(tmp_path):
    path = tmp_path / "history.csv"
    save_result_csv("", "normal", 90.0, path)
    assert path.read_text().strip() == "-normal-90.00%"


def test_capture_window_returns_the_output_path(monkeypatch, tmp_path):
    install_fake_screenshot(monkeypatch)
    output = str(tmp_path / "capture.jpg")
    assert capture_window(FakeWidget(), output) == output


def test_capture_window_saves_image_with_widget_size(monkeypatch, tmp_path):
    install_fake_screenshot(monkeypatch)
    output = str(tmp_path / "capture.jpg")
    capture_window(FakeWidget(), output)
    assert Image.open(output).size == (300, 200)


def test_generate_pdf_report_creates_both_files(monkeypatch, tmp_path):
    install_fake_screenshot(monkeypatch)
    monkeypatch.chdir(tmp_path)
    generate_pdf_report(FakeWidget(), 5)
    assert Path("reports/Reporte5.jpg").exists()
    assert Path("reports/Reporte5.pdf").exists()


def test_generate_pdf_report_returns_string_path(monkeypatch, tmp_path):
    install_fake_screenshot(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert isinstance(generate_pdf_report(FakeWidget(), 0), str)
