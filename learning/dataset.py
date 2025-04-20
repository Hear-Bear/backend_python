import os
import random

import numpy as np
import pandas as pd
import torch
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Gain
)
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

# 랜덤 시드 고정
random.seed(442)
np.random.seed(442)

# 데이터셋에서 삭제할 (필요 없는)columns
remove_columns = ['filename', 'esc10', 'audio']
# Google AudioSet를 pretrained한 모델과 특징 추출기를 불러옴
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

# HuggingFace에 저장된 ashraq/esc50 데이터셋을 불러옴
esc50_dataset = load_dataset('ashraq/esc50')

# 특징 추출기
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


def split_by_fold(dataset_split, test_fold=1):
    train_dataset = dataset_split.filter(lambda df: df['fold'] != test_fold)
    val_dataset = dataset_split.filter(lambda df: df['fold'] == test_fold)
    print(f"학습 데이터셋: {train_dataset.shape}, 검증 데이터셋: {val_dataset.shape}")
    return train_dataset, val_dataset


# 체크포인트가 있다면 "./output" 내의 checkpoint 디렉토리를 자동으로 불러오기
def get_latest_checkpoint(output_dir):
    if os.path.exists(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith('checkpoint') and os.path.isdir(os.path.join(output_dir, d))]
        if checkpoints:
            # 정렬 후 마지막 체크포인트 선택
            return sorted(checkpoints)[-1]


def preprocess_function(data, apply_augmentation=False):
    raw = data['audio']['array']
    sr = data['audio']['sampling_rate']

    waveform = torch.tensor(raw, dtype=torch.float32)

    # 학습 시 증강 적용 여부
    if apply_augmentation:
        waveform, sr = advanced_augment_audio_moderate(waveform, sr)
        # 증강 후 waveform을 numpy 배열로 변환 (특징 추출기가 numpy 입력을 필요로 할 경우)
        raw = waveform.squeeze(0).numpy()
    else:
        raw = waveform.squeeze(0).numpy()

    inputs = feature_extractor(raw, sampling_rate=sr, return_tensors='pt')
    # inputs['input_values'] = inputs['input_values'].squeeze(0).numpy()

    return inputs


def get_moderate_augmentation_pipeline(sample_rate):
    # # RoomSimulator 효과는 IR(Impulse Response) 파일이 필요합니다.
    # # 실제 환경의 방음 특성을 반영한 IR 파일을 사용하세요.
    # ir_path = "your_ir_file.wav"  # IR 파일 경로를 지정합니다.
    #
    # # Windows 환경에서 RoomSimulator가 동작하지 않을 경우 None으로 처리합니다.
    # room_simulator = RoomSimulator(ir_path=ir_path, p=0.3) if platform.system() != "Windows" else None

    # 여러 효과들을 Compose로 묶어줍니다.
    # 각 트랜스폼은 실제 환경에서 너무 극단적이지 않은 수준으로 적용합니다.
    transforms = [
        # 배경 잡음을 약하게 추가 (실제 녹음된 환경의 미세한 잡음 모방)
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
        # 약간의 시간 변형 (너무 심하게 적용하면 오디오 특징이 깨질 수 있으므로 미세하게)
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
        # 피치를 약간 변화시켜, 자연스러운 음높이 차이 모방 (±1 반음 정도)
        PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
        # 오디오 신호의 시작이나 끝을 약간 이동 (전체 길이의 5~10% 정도)
        Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
        # 실제 방의 잔향 효과를 반영 (IR 파일이 있다면 자연스러운 리버브 효과 적용)
        # room_simulator,
        # 볼륨을 약간 조절하여 녹음 조건의 차이를 반영
        Gain(min_gain_db=-2.0, max_gain_db=2.0, p=0.5),
    ]
    # None인 효과는 필터링합니다.
    transforms = [t for t in transforms if t is not None]
    return Compose(transforms, p=1.0)


def advanced_augment_audio_moderate(waveform, sample_rate):
    """
    입력된 waveform (torch.Tensor, shape: [1, samples])에 대해
    실제 환경을 적당히 모방하는 증강을 적용합니다.
    """
    # waveform을 numpy array (모노 채널)로 변환합니다.
    np_waveform = waveform.squeeze(0).numpy()

    # 증강 파이프라인을 얻습니다.
    augmentation_pipeline = get_moderate_augmentation_pipeline(sample_rate)

    # 증강 효과 적용: np_waveform는 1차원 numpy array라고 가정합니다.
    augmented_np_waveform = augmentation_pipeline(np_waveform, sample_rate=sample_rate)

    # numpy array를 다시 torch.Tensor로 변환 후 (채널 차원 추가) 반환합니다.
    augmented_waveform = torch.tensor(augmented_np_waveform, dtype=torch.float32).unsqueeze(0)
    return augmented_waveform, sample_rate


def load_dataset_esc50(apply_augmentation=False):
    esc50_dataloader = esc50_dataset['train']
    esc50_dataloader = esc50_dataloader.cast_column('audio', Audio(sampling_rate=16000))
    # print(esc50_dataloader.shape)

    input_values = esc50_dataloader.map(
        lambda data: preprocess_function(data, apply_augmentation)
    )
    # print(f'특징 추출기 적용 후: {input_values}')

    # # 폴드 설정
    df = pd.DataFrame(esc50_dataloader)
    folds = df.query('fold >= 1')['fold'].drop_duplicates().tolist()

    # 최종적인 dataset 설정
    # esc50_dataloader = esc50_dataloader.add_column(name='input_values', column=input_values)
    input_values = input_values.remove_columns(remove_columns)
    # 학습의 원할함을 위해 target을 명시적으로 label로 변경
    input_values = input_values.rename_column('target', 'label')

    return input_values, folds


# load_dataset_esc50()
