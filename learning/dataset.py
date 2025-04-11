import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import torchaudio
import torchaudio.sox_effects as sox_effects
import random

# 데이터셋에서 삭제할 (필요 없는)columns
remove_columns = ['filename', 'esc10', 'audio']
# Google AudioSet를 pretrained한 모델과 특징 추출기를 불러옴
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

# HuggingFace에 저장된 ashraq/esc50 데이터셋을 불러옴
esc50_dataset = load_dataset('ashraq/esc50')

# 특징 추출기
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

random.seed(442)


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
        waveform, sr = advanced_augment_audio_torchaudio(waveform, sr)
        # 증강 후 waveform을 numpy 배열로 변환 (특징 추출기가 numpy 입력을 필요로 할 경우)
        raw = waveform.squeeze(0).numpy()
    else:
        raw = waveform.squeeze(0).numpy()

    inputs = feature_extractor(raw, sampling_rate=sr, return_tensors='pt')
    # inputs['input_values'] = inputs['input_values'].squeeze(0).numpy()

    return inputs


def advanced_augment_audio_torchaudio(waveform, sample_rate):
    """
    torchaudio의 sox_effects와 직접 잡음 추가, reverb, compand, overdrive 등을 조합하여
    고급 증강을 적용하는 함수입니다.
    waveform: torch.Tensor, shape = (channels, samples)
    sample_rate: 정수형
    """

    # 1. 배경 잡음 추가 (예: 신호에 소량의 잡음을 더함)
    if random.random() < 0.5:
        noise = torch.randn_like(waveform) * 0.005  # 잡음 세기는 데이터에 따라 조절
        waveform = waveform + noise

    effects = []  # sox 효과 체인을 저장할 리스트

    # 2. 룸 리버브 효과 적용
    if random.random() < 0.5:
        # reverb 효과: 파라미터는 [reverberance, damping, room_scale, stereo_depth, pre_delay, wet_gain]
        # 여기서는 임의의 값을 넣어 변형을 주도록 함 (필요시 세부 조절 가능)
        effects.append(["reverb", "50", "50", "100", "100", "0", "0"])

    # 3. 코덱 시뮬레이션: compand 효과 적용 (압축 후 확장)
    if random.random() < 0.5:
        # compand 파라미터 예시: attack, decay, transfer function 등 여러 인자를 포함합니다.
        # 이 값들은 실험을 통해 조정해야 하며, 여기서는 예시 값들을 사용합니다.
        effects.append(["compand", "0.3,0.8", "6", "0.2", "-70,-60,-20", "-5", "-90", "0.2"])

    # 4. 옵션: 추가 왜곡 효과 (overdrive)
    if random.random() < 0.3:
        effects.append(["overdrive", "20"])

    # sox_effects로 효과 체인 적용 (효과가 하나라도 있다면)
    if effects:
        augmented_waveform, new_sample_rate = sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        return augmented_waveform, new_sample_rate
    else:
        return waveform, sample_rate


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
    print(f'칼럼 삭제: {input_values}')

    # print_dim = np.array(input_values['input_values'])
    # print(f'input shape: {print_dim.shape}')
    # print(f'첫번째 차원 데이터: {print_dim[0].ndim}')
    # print(f'fold 리스트: {folds}')

    return input_values, folds
    # return input_values


# load_dataset_esc50()
