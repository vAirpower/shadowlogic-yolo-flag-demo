"""Tests for the renderer-side 15s sticky persistence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from webcam_demo import apply_sticky_relabel


def _detection(cls):
    return (10, 20, 100, 200, 0.85, cls)


def test_sticky_inactive_passes_through():
    detections = [_detection(0), _detection(0), _detection(79)]
    out = apply_sticky_relabel(detections, sticky_active=False)
    assert [d[5] for d in out] == [0, 0, 79]


def test_sticky_active_relabels_person_to_enemy():
    detections = [_detection(0), _detection(0), _detection(79)]
    out = apply_sticky_relabel(detections, sticky_active=True)
    assert [d[5] for d in out] == [79, 79, 79]


def test_sticky_preserves_box_geometry_and_confidence():
    """Sticky must only flip the class id; everything else unchanged."""
    detections = [(50, 60, 200, 300, 0.91, 0)]
    out = apply_sticky_relabel(detections, sticky_active=True)
    assert len(out) == 1
    x1, y1, x2, y2, conf, cls = out[0]
    assert (x1, y1, x2, y2, conf) == (50, 60, 200, 300, 0.91)
    assert cls == 79


def test_sticky_does_not_touch_non_person_classes():
    """Edge case: if a non-person/non-enemy class slipped through, leave it."""
    detections = [(0, 0, 10, 10, 0.5, 5)]
    out = apply_sticky_relabel(detections, sticky_active=True)
    assert out[0][5] == 5
