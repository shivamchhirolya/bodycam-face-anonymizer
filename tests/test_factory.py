from face_anon.config import merge_config
from face_anon.factory import resolve_detector_name


def test_merge_config_nests_detector_keys():
    base = {"detector": {"name": "yunet", "score_threshold": 0.18}, "keep": 1}
    out = merge_config(base, {"detector": {"score_threshold": 0.22}})
    assert out["detector"]["name"] == "yunet"
    assert out["detector"]["score_threshold"] == 0.22
    assert out["keep"] == 1
    assert base["detector"]["score_threshold"] == 0.18


def test_resolve_detector_name_passthrough():
    assert resolve_detector_name({"detector": {"name": "yunet"}}) == "yunet"
    assert resolve_detector_name({"detector": {"name": "scrfd"}}) == "scrfd"
