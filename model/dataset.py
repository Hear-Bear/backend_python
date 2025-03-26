import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class AudioDataset(Dataset):
    def __init__(self, data_root, transform=None, segment_duration=5.0, sample_rate=16000):
        """
        Args:
            data_root (str): 오디오 데이터셋 루트 디렉터리.
            transform (callable, optional): 오디오 파형에 적용할 변환 (예: MelSpectrogram).
            segment_duration (float): 사용할 세그먼트 길이 (초).
            sample_rate (int): 표준 샘플레이트 (리샘플링 필요시 적용).
        """
        self.audio_root = os.path.join(data_root, 'audio')
        self.annotations_root = os.path.join(data_root, 'annotations')
        self.transform = transform
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate

        self.audio_files = []
        self.annotation_files = []

        # audio 폴더 아래의 모든 폴더(마이크 폴더)를 순회
        for mic_folder in sorted(os.listdir(self.audio_root)):
            mic_folder_path = os.path.join(self.audio_root, mic_folder)
            if os.path.isdir(mic_folder_path):
                for file in os.listdir(mic_folder_path):
                    if file.endswith('.wav'):
                        self.audio_files.append(os.path.join(mic_folder_path, file))
                        # self.mic_idc.append(int(file.split('.')[0]))

        # annotations 폴더 아래의 모든 폴더(마이크 폴더)를 순회
        for mic_folder in sorted(os.listdir(self.annotations_root)):
            mic_folder_path = os.path.join(self.annotations_root, mic_folder)
            if os.path.isdir(mic_folder_path):
                for file in os.listdir(mic_folder_path):
                    if file.endswith('.json'):
                        self.annotation_files.append(os.path.join(mic_folder_path, file))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        annotation_path = self.annotation_files[idx]
        waveform, sr = torchaudio.load(audio_path)

        # 샘플레이트 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 지정한 segment_duration 만큼 segment로 분할(예: 5초)
        segment_samples = int(self.segment_duration * self.sample_rate)
        if waveform.size(1) >= segment_samples:
            waveform = waveform[:, :segment_samples]
        else:
            pad_amount = segment_samples - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_amount))

        # MelSpectrogram 변환 적용
        if self.transform:
            waveform = self.transform(waveform)

        # annotation 정보를 활용하려면,
        # 예를 들어, annotation 폴더 내의 해당 mic_id 폴더에서 관련 파일을 읽어와 추가 정보를 반환할 수 있음.
        # annotation_file = os.path.join(self.annotation_root, mic_id, 'annotation_file.csv')
        # annotation_data = parse_annotation(annotation_file)

        return waveform, # annotation