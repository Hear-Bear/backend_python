import os

import pandas as pd
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, audio_root, csv_file, transform=None, segment_duration=5.0, sample_rate=16000):
        """
        Args:
            audio_root (str): 오디오 데이터셋 루트 디렉터리.
            csv_file (str): 메타데이터 CSV 파일 경로.
            transform (callable, optional): 오디오 파형에 적용할 변환 (예: MelSpectrogram).
            segment_duration (float): 사용할 세그먼트 길이 (초).
            sample_rate (int): 표준 샘플레이트 (리샘플링 필요시 적용).
        """
        # self.data_root = data_root
        # self.audio_root = os.path.join(data_root, 'audio')
        self.audio_root = audio_root
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # CSV 파일에서 해당 인덱스의 정보를 읽음
        row = self.metadata.iloc[idx]
        filename = row['filename']
        audio_path = os.path.join(self.audio_root, filename)

        # 오디오 파일 로드
        waveform, sr = torchaudio.load(audio_path)

        # 샘플레이트 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 고정 길이 세그먼트로 분할 또는 패딩
        segment_samples = int(self.segment_duration * self.sample_rate)
        if waveform.size(1) >= segment_samples:
            waveform = waveform[:, :segment_samples]
        else:
            total_pad = segment_samples - waveform.size(1)
            left_pad = total_pad // 2
            right_pad = total_pad - left_pad
            waveform = F.pad(waveform, (left_pad, right_pad))

        # 지정한 transform 적용
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, row['target']

    def get_label(self):
        unique_mapping = self.metadata[['target', 'category']].drop_duplicates()
        label = dict(zip(unique_mapping['target'], unique_mapping['category']))
        return label