import pytest
from windowmind.config import Settings

def test_overlap_must_be_smaller_than_window():
    settings = Settings(window_size=300, window_overlap=300)
    with pytest.raises(ValueError):
        settings.validate_window_configuration()
