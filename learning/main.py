import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from learning.dataset import AudioDataset
from learning.function import train
from learning.model import AudioResNet

# 설정 변수
audio_root = 'C:\\dev\\python\\capstone\\esc50\\audio'
csv_file = 'C:\\dev\\python\\capstone\\esc50\\meta\\esc50.csv'
segment_duration = 5.0
sample_rate = 16000
batch_size = 32
epochs = 100

# Mel Spectrogram 설정: 출력 텐서 shape => (채널, n_mels, time)
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)

# 데이터셋 및 DataLoader 생성
dataset = AudioDataset(audio_root, csv_file, transform=mel_transform,
                       segment_duration=segment_duration, sample_rate=sample_rate)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# 레이블 매핑 정보 (target과 category 매핑)
label_mapping = dataset.get_label()
num_classes = len(label_mapping)
print("Number of classes:", num_classes)

# 사용할 모델 선택: 'resnet18', 'resnet34', 또는 'resnet50'
base_model = 'resnet50'  # 예시로 resnet50 선택 (필요에 따라 변경)
model = AudioResNet(num_classes=num_classes, backbone=base_model, pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss 함수 및 optimizer 설정 (분류 문제이므로 CrossEntropyLoss 사용)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 시작
train(model, dataloader, criterion, optimizer, device, epochs)
