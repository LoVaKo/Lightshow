"""
beat_detection.py

This script analyzes an audio file using the `librosa` library to extract key musical features:
- **Tempo (BPM)**: Determines the speed of the music.
- **Beat Times**: Detects rhythmic beats.
- **Onset Times**: Identifies sharp sound changes (e.g., drum hits, vocal attacks).
- **RMS Energy**: Measures loudness over time.
- **Chroma Features**: Analyzes pitch (musical notes present over time).

The extracted features can be used to synchronize lights with music in a light show.

Usage:
    Run this script directly to analyze an audio file:
        python beat_detection.py
"""

import librosa
import numpy as np
from typing import Dict, Tuple

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        Tuple[np.ndarray, int]: Audio waveform (`y`) and sample rate (`sr`).
    """
    y, sr = librosa.load(file_path)
    return y, sr

def analyze_audio(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Analyze the given audio waveform to extract key features:
    - Tempo (BPM)
    - Beat times
    - Onset times (sharp sound changes)
    - RMS Energy (loudness over time)
    - Chroma (musical note presence over time)

    Args:
        y (np.ndarray): Audio waveform.
        sr (int): Sample rate of the audio.

    Returns:
        Dict[str, Any]: Dictionary containing extracted audio features:
            - "tempo" (np.ndarray): Estimated beats per minute (BPM).
            - "beat_times" (np.ndarray): Array of times (seconds) when beats occur.
            - "onset_times" (np.ndarray): Array of times (seconds) when musical onsets occur.
            - "rms" (np.ndarray): Normalized root mean square (RMS) energy levels.
            - "chroma" (np.ndarray): Transposed chroma matrix (time, pitch class).
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    rms = librosa.feature.rms(y=y)[0]
    rms = rms / np.max(rms)  # Normalize to 0-1 for brightness mapping

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = chroma.T  # Transpose to make it time-based [(time, pitch_class)]

    return {
        "tempo": tempo, 
        "beat_times": beat_times, 
        "onset_times": onset_times, 
        "rms": rms, 
        "chroma": chroma
        }

if __name__ == '__main__':
    # Load and analyze audio
    y, sr = load_audio('data/song.mp3')
    audio_features = analyze_audio(y, sr)
    
    # Print extracted audio features
    print(f"Tempo: {audio_features['tempo']} BPM")
    print(f"Beat Times: {audio_features['beat_times']}")
    print(f"Onset Times: {audio_features['onset_times']}")
    print(f"RMS Energy: {audio_features['rms']}")
    print(f"Chroma Features: {audio_features['chroma']}")
