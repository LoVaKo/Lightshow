import pytest
import numpy as np
from src.beat_detection import load_audio, analyze_audio


@pytest.fixture
def sample_audio_path():
    return "./data/song.mp3"


def test_load_audio(sample_audio_path):
    """
    Test if the load_audio function correctly loads the audio file
    """
    y, sr = load_audio(sample_audio_path)

    assert isinstance(y, np.ndarray), "Audio waveform should be a numpy array"
    assert isinstance(sr, int), "Sample rate should be an integer"
    assert y.shape[0] > 0, "Audio waveform should not be empty"
    assert sr > 0, "Sample rate should be greater than 0"


def test_analyze_audio(sample_audio_path):
    """
    Test if the analyze_audio function correctly extracts audio features
    """
    y, sr = load_audio(sample_audio_path)
    audio_features = analyze_audio(y, sr)

    assert isinstance(audio_features, dict), \
        "Audio features should be in a dictionary"
    assert "tempo" in audio_features, \
        "Tempo should be a key in the dictionary"
    assert "beat_times" in audio_features, \
        "Beat times should be a key in the dictionary"
    assert "onset_times" in audio_features, \
        "Onset times should be a key in the dictionary"
    assert "rms" in audio_features, \
        "RMS energy should be a key in the dictionary"
    assert "chroma" in audio_features, \
        "Chroma features should be a key in the dictionary"
    assert isinstance(audio_features["tempo"], np.ndarray), \
        "Tempo should be a float"
    assert isinstance(audio_features["beat_times"], np.ndarray), \
        "Beat times should be a numpy array"
    assert isinstance(audio_features["onset_times"], np.ndarray), \
        "Onset times should be a numpy array"
    assert isinstance(audio_features["rms"], np.ndarray), \
        "RMS energy should be a numpy array"
    assert isinstance(audio_features["chroma"], np.ndarray), \
        "Chroma features should be a numpy array"

