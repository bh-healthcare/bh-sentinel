"""Public API smoke tests for bh-sentinel-ml."""

from __future__ import annotations


def test_version_exists():
    from bh_sentinel.ml import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_all_exists():
    from bh_sentinel.ml import __all__

    assert isinstance(__all__, list)
    assert "__version__" in __all__


def test_classes_importable():
    from bh_sentinel.ml import (
        TransformerClassifier,
        ZeroShotClassifier,
        export_to_onnx,
    )

    assert callable(TransformerClassifier)
    assert callable(ZeroShotClassifier)
    assert callable(export_to_onnx)


def test_all_exports_are_importable():
    """Every name in __all__ must be importable from the package."""
    import bh_sentinel.ml as ml

    for name in ml.__all__:
        assert hasattr(ml, name), f"{name} is in __all__ but not importable"
