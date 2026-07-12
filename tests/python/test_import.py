import svo


def test_import_smoke() -> None:
    assert isinstance(svo.__version__, str)
    assert svo.__version__
    info = svo.build_info()
    assert info["version"] == svo.__version__
    assert isinstance(info["core_version"], str)
    assert isinstance(info["cuda_enabled"], bool)
    assert isinstance(info["extension_path"], str)
