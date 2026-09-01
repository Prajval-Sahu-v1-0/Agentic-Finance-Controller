"""Regression tests for src.launcher's dev/production data gating.

Per explicit instruction: synthetic data generation is a development
convenience only and must never run in the packaged .exe. Gated on
src.paths.is_frozen() rather than a separate flag.
"""

from src import launcher


def test_dev_sample_dataset_skipped_when_frozen(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "is_frozen", lambda: True)
    called = {"generate": False}

    def fake_generate_dataset(*args, **kwargs):
        called["generate"] = True

    monkeypatch.setattr("src.generator.generate_dataset", fake_generate_dataset)
    launcher._ensure_dev_sample_dataset()

    assert called["generate"] is False


def test_dev_sample_dataset_skipped_when_already_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "is_frozen", lambda: False)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    (tmp_path / "gateway_records.json").write_text("[]", encoding="utf-8")
    called = {"generate": False}

    def fake_generate_dataset(*args, **kwargs):
        called["generate"] = True

    monkeypatch.setattr("src.generator.generate_dataset", fake_generate_dataset)
    launcher._ensure_dev_sample_dataset()

    assert called["generate"] is False


def test_dev_sample_dataset_generated_when_missing_and_not_frozen(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "is_frozen", lambda: False)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    called = {}

    def fake_generate_dataset(seed, output_dir, save):
        called["seed"] = seed
        called["output_dir"] = output_dir
        called["save"] = save

    monkeypatch.setattr("src.generator.generate_dataset", fake_generate_dataset)
    launcher._ensure_dev_sample_dataset()

    assert called == {"seed": 42, "output_dir": tmp_path, "save": True}
