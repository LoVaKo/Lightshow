import librosa
import numpy as np

def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr

def analyze_audio(y, sr):
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
    y, sr = load_audio('data/song.mp3')
    audio_features = analyze_audio(y, sr)
    print(f"Tempo: {audio_features['tempo']} BPM")
    print(f"Beat Times: {audio_features['beat_times']}")
    print(f"Onset Times: {audio_features['onset_times']}")
    print(f"RMS Energy: {audio_features['rms']}")
    print(f"Chroma Features: {audio_features['chroma']}")
