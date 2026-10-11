"""Packaging checks: metadata, the typing marker, the public API."""

import importlib.metadata
import importlib.resources

import sentinex as sx


def test_version_matches_metadata():
    assert sx.__version__ == importlib.metadata.version("sentinex")


def test_reference_version_is_not_publishable():
    # A local version label ("+reference") is rejected by PyPI, and so is a
    # "Private ::" classifier. This branch is a reference, not a release.
    assert "+reference" in sx.__version__
    classifiers = importlib.metadata.metadata("sentinex").get_all("Classifier")
    assert "Private :: Do Not Upload" in classifiers


def test_py_typed_marker_ships():
    assert importlib.resources.files("sentinex").joinpath("py.typed").is_file()


def test_public_api():
    for name in sx.__all__:
        assert hasattr(sx, name), name
