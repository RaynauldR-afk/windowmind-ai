import pytest
from windowmind.windowing import SlidingWindowEngine

def test_window_engine_returns_multiple_overlapping_windows():
    text = ("Sliding windows preserve context across chunk boundaries. " * 80)
    engine = SlidingWindowEngine(window_size=300, overlap=60)
    windows, stats = engine.split(text)
    assert len(windows) > 1
    assert stats.stride == 240
    assert windows[0].start_char == 0
    assert windows[0].end_char > windows[1].start_char

def test_empty_document_rejected():
    with pytest.raises(ValueError):
        SlidingWindowEngine().split("   ")

def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        SlidingWindowEngine(window_size=100, overlap=100)
