from face_anon.track import ByteTracker, IoUTracker
from face_anon.types import FaceBox


def _box(x, y, s=40, score=0.9):
    return FaceBox(x, y, x + s, y + s, score)


def test_track_id_stable_under_jitter():
    tr = IoUTracker(iou_match=0.2, max_age=8, smooth=0.4)
    a = tr.update([_box(100, 80)])
    b = tr.update([_box(108, 84)])
    assert a[0].track_id == b[0].track_id


def test_coast_moves_with_velocity():
    tr = IoUTracker(iou_match=0.2, max_age=8, smooth=0.0, vel_smooth=0.0)
    tr.update([_box(50, 50)])
    tr.update([_box(60, 50)])  # vx ≈ +10
    coasted = tr.update([])
    assert len(coasted) == 1
    # After one miss we still paint; after the next miss the box should slide right.
    moved = tr.update([])
    assert moved[0].x1 > coasted[0].x1


def test_coast_through_misses():
    tr = IoUTracker(iou_match=0.2, max_age=5, min_hits=1)
    tr.update([_box(50, 50)])
    coasted = []
    for _ in range(4):
        coasted.append(tr.update([]))
    assert all(len(c) == 1 for c in coasted)
    gone = tr.update([])
    gone = tr.update([])  # age 6
    assert gone == [] or all(t.track_id != coasted[0][0].track_id for t in gone)


def test_bytetrack_recovers_low_score_on_second_stage():
    tr = ByteTracker(high_thresh=0.50, low_thresh=0.15, iou_match=0.2, max_age=8, smooth=0.0)
    first = tr.update([_box(100, 80, score=0.9)])
    # Detector almost drops the face (score 0.22). Stage 2 should still match.
    second = tr.update([_box(108, 84, score=0.22)])
    assert len(second) == 1
    assert second[0].track_id == first[0].track_id
