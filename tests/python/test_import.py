import svo


def test_import_smoke() -> None:
    assert svo.__version__ == "0.1.0"
    assert isinstance(svo.build_info()["cuda_enabled"], bool)
