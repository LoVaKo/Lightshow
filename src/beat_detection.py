import librosa
import numpy as np

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path)
    return waveform, sample_rate

if __name__ == '__main__':
    waveform, sample_rate = load_audio('data/song.mp3')
    print(f"Audio loaded with sample rate: {sample_rate}")

    