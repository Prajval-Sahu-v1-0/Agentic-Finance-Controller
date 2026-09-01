"""Regression tests for src.paths — frozen-safe path resolution for the
packaged .exe, and src.launcher's dev-only synthetic data gating."""

from pathlib import Path

from src import paths


def test_is_frozen_false_under_normal_execution() -> None:
    # pytest runs from source, never as a PyInstaller-frozen build.
    assert paths.is_frozen() is False


def test_app_root_is_project_root_when_not_frozen() -> None:
    root = paths.app_root()
    assert (root / "src").is_dir()
    assert (root / "requirements.txt").is_file()


def test_bundle_root_equals_app_root_when_not_frozen() -> None:
    assert paths.bundle_root() == paths.app_root()


def test_data_dir_and_static_dir_are_under_expected_roots() -> None:
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"
    assert paths.STATIC_DIR.name == "static"


def test_app_root_uses_executable_dir_when_frozen(monkeypatch) -> None:
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "executable", str(Path("C:/fake/dist/recon-agent.exe")), raising=False)
    assert paths.app_root() == Path("C:/fake/dist").resolve()


def test_bundle_root_uses_meipass_when_frozen(monkeypatch) -> None:
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "_MEIPASS", "C:/fake/temp/_MEI12345", raising=False)
    assert paths.bundle_root() == Path("C:/fake/temp/_MEI12345")
