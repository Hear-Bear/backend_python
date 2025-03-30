import torchaudio.transforms as T


# 설정 변수
audio_root = 'C:\\dev\\python\\capstone\\esc50\\audio'
csv_file = 'C:\\dev\\python\\capstone\\esc50\\meta\\esc50.csv'
segment_duration = 5.0
sample_rate = 16000
batch_size = 32
epochs = 100

# Mel Spectrogram 설정
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
