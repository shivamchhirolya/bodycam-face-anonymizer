from face_anon.detect_retinaface import decode_boxes, prior_boxes


def test_prior_count_640():
    priors = prior_boxes(640, 640)
    # 80*80*2 + 40*40*2 + 20*20*2
    assert priors.shape == (16800, 4)
    assert priors.min() >= 0.0
    assert priors.max() <= 1.0 + 1e-5


def test_decode_center_prior():
    priors = prior_boxes(320, 320)
    loc = priors * 0.0
    boxes = decode_boxes(loc, priors, [0.1, 0.2])
    # Zero offset → box is the prior converted from cxcywh to xyxy
    assert boxes.shape == priors.shape
    widths = boxes[:, 2] - boxes[:, 0]
    assert (widths > 0).all()
