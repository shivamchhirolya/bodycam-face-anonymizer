import numpy as np

from face_anon.preprocess import resize_for_detect


def test_upscales_small_bodycam():
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    out, scale = resize_for_detect(img, 640)
    assert abs(scale - 640 / 480) < 1e-6
    assert out.shape[1] == 640
    assert out.shape[0] in (480, 482)


def test_downscales_1080p():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out, scale = resize_for_detect(img, 640)
    assert abs(scale - 640 / 1920) < 1e-6
    assert out.shape[1] == 640
