# dataset.py
# FSD50KSubsetDataset 클래스와 prepare_data 함수 정의

import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from audiomentations import (
    Compose,
    TimeStretch,
    PitchShift,
    Gain,
    ApplyImpulseResponse,
    LowPassFilter,
    AddBackgroundNoise,
    SomeOf
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import config


# 3. [추가] 🌟 제공해주신 고급 증강 파이프라인 함수
def get_advanced_indoor_pipeline(
        sample_rate,
        background_sounds_path=None,
        ir_sounds_path=None
):
    """
    실내 환경(울림, 차폐)에 특화된 고급 증강 파이프라인
    """

    # Level 1: 소리 자체를 변형하는 기본 증강 리스트
    core_transforms = [
        Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.5),
        PitchShift(min_semitones=-1.5, max_semitones=1.5, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    ]

    # Level 2: 공간/환경을 시뮬레이션하는 고급 증강 리스트
    environmental_transforms = []

    if background_sounds_path:
        environmental_transforms.append(
            AddBackgroundNoise(
                sounds_path=background_sounds_path,
                min_snr_db=3.0,
                max_snr_db=15.0,
                p=0.6
            )
        )

    if ir_sounds_path:
        environmental_transforms.append(
            ApplyImpulseResponse(
                ir_path=ir_sounds_path,
                p=0.5
            )
        )

    environmental_transforms.append(
        LowPassFilter(
            min_cutoff_freq=2000,
            max_cutoff_freq=4000,
            p=0.4,
        )
    )

    return Compose(
        transforms=[
            SomeOf(
                transforms=environmental_transforms,
                num_transforms=(0, 2),
                p=1.0
            ),
            SomeOf(
                transforms=core_transforms,
                num_transforms=(1, 3),
                p=1.0
            )
        ],
        p = 1.0
    )


class FSD50KSubsetDataset(Dataset):
    # FSD50K 오디오 파일 로드, 증강, 피처 추출을 담당하는 커스텀 데이터셋

    def __init__(self, df, audio_dir, feature_extractor, use_augmentation=None):
        self.df = df
        self.audio_dir = audio_dir
        self.feature_extractor = feature_extractor
        self.target_sr = feature_extractor.sampling_rate

        if use_augmentation:
            # config.py에서 경로를 가져오되, 실제 디렉토리인지 확인
            bg_path = config.BACKGROUND_NOISE_DIR
            ir_path = config.IMPULSE_RESPONSE_DIR

            valid_bg_path = bg_path if bg_path and os.path.isdir(bg_path) else None
            valid_ir_path = ir_path if ir_path and os.path.isdir(ir_path) else None

            # 경로가 유효하지 않으면 경고 출력 (학습은 진행됨)
            if bg_path and not valid_bg_path:
                print(f"경고: config.py의 BACKGROUND_NOISE_DIR '{bg_path}'를 찾을 수 없습니다. 배경 소음 증강을 건너뜁니다.")
            if ir_path and not valid_ir_path:
                print(f"경고: config.py의 IMPULSE_RESPONSE_DIR '{ir_path}'를 찾을 수 없습니다. 울림(IR) 증강을 건너뜁니다.")

            # 파이프라인 함수를 호출하여 self.augment에 할당
            self.augment = get_advanced_indoor_pipeline(
                sample_rate=self.target_sr,
                background_sounds_path=valid_bg_path,
                ir_sounds_path=valid_ir_path
            )

            # SpecAugment 파이프라인 정의
            self.use_spec_augment = True
            self.spec_augmenter = torch.nn.Sequential(
                # (batch, freq, time) -> (batch, 128, 1024)
                # 128개의 주파수 빈 중 최대 20개를 가림
                T.FrequencyMasking(freq_mask_param=20),
                # 1024개의 시간 프레임 중 최대 50개를 가림
                T.TimeMasking(time_mask_param=50)
            )
        else:
            self.augment = None
            self.use_spec_augment = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, f"{row['fname']}.wav")

        try:
            wav, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        except Exception as e:
            print(f"오류: 파일 로드 실패 {file_path}. 원인: {e}")
            print("경고: 5초 분량의 빈 오디오로 대체합니다.")
            wav = np.zeros(self.target_sr * 5)

        if self.augment:
            wav = self.augment(samples=wav, sample_rate=self.target_sr)

        inputs = self.feature_extractor(
            wav,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.MAX_AUDIO_LENGTH
        )
        input_values = inputs.input_values.squeeze(0)

        # SpecAugment 적용
        if self.use_spec_augment:
            # (Time, Freq) -> (Freq, Time)로 축 변경
            input_values = input_values.transpose(0, 1)
            # (Freq, Time) -> (Batch, Freq, Time)로 임시 변경 (1, 128, 1024)
            input_values = input_values.unsqueeze(0)

            input_values = self.spec_augmenter(input_values)  # 마스킹 적용

            # (Batch, Freq, Time) -> (Freq, Time)
            input_values = input_values.squeeze(0)
            # (Freq, Time) -> (Time, Freq)로 원복
            input_values = input_values.transpose(0, 1)

        return {
            "input_values": input_values,
            "labels": torch.tensor(row['labels_vector'], dtype=torch.float)
        }


def prepare_data(dev_csv, eval_csv, class_groups, dev_audio_dir, eval_audio_dir, val_split_size, random_seed):
    # class_groups 딕셔너리를 기반으로 train/eval 데이터프레임과 레이블 맵을 반환.
    print("\n-------------------------- 데이터 준비 시작 --------------------------")

    # 새로운 레이블 맵 생성 (e.g., {'Dog': 0, 'Cat': 1, 'Alarm': 2, ...})
    new_labels = list(class_groups.keys())
    num_classes = len(new_labels)
    label2id = {name: i for i, name in enumerate(new_labels)}
    id2label = {i: name for i, name in enumerate(new_labels)}

    # 역방향 조회 맵 생성 (빠른 필터링 용) (e.g., {'Dog': 0, 'Bark': 0, 'Cat': 1, 'Meow': 1, ...})
    source_label_to_group_id = {}
    for group_name, source_labels in class_groups.items():
        group_id = label2id[group_name]
        for source_label in source_labels:
            if source_label in source_label_to_group_id:
                print(f"경고: '{source_label}'이 여러 그룹에 포함되었습니다. 마지막 그룹 '{group_name}'에 할당됩니다.")
            source_label_to_group_id[source_label] = group_id

    print(f"총 {len(source_label_to_group_id)}개의 원본 레이블을 {len(label2id)}개의 그룹으로 통합합니다.")

    def filter_dataframe(csv_path, audio_dir):
        # CSV를 로드하고 'source_label_to_group_id' 맵을 사용해 필터링
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
            return None

        filtered_data = []
        for _, row in df.iterrows():
            file_labels = row['labels'].split(',')  # e.g., ["Dog", "Speech"]
            found_group_ids = set()

            # 파일의 레이블 중, Target Class에 있는지 확인
            for label in file_labels:
                if label in source_label_to_group_id:
                    found_group_ids.add(source_label_to_group_id[label])

            if found_group_ids:
                if os.path.exists(f"{audio_dir}/{row['fname']}.wav"):

                    # Multi-Hot Vector 생성
                    labels_vector = np.zeros(num_classes, dtype=int)
                    for group_id in found_group_ids:
                        labels_vector[group_id] = 1

                    filtered_data.append({
                        "fname": row['fname'],
                        "labels_vector": labels_vector.tolist() # 벡터 저장
                    })

        if not filtered_data:
            print(f"경고: {csv_path}에서 타겟 클래스를 찾지 못했습니다.")
            return pd.DataFrame()

        return pd.DataFrame(filtered_data)

    # 학습 데이터프레임 생성
    print(f"\n--- 학습 데이터 처리: {dev_csv} ---")
    train_df = filter_dataframe(dev_csv, dev_audio_dir)
    if train_df is None or train_df.empty:
        print("오류: 학습 데이터가 없습니다.")
        return None, None, None, None, None  # 5개 반환

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split_size,
        random_state=random_seed,
        # stratify=train_df['label2id']  # 레이블 비율을 유지하며 분리
    )

    # Train Set 기준으로 클래스 가중치 계산
    print("--- 클래스 가중치 계산 (Train Set 기준) ---")

    # 각 클래스별 Positive(1) 개수 계산
    train_labels_matrix = np.array(train_df['labels_vector'].tolist())
    positive_counts = np.sum(train_labels_matrix, axis=0)  # 예: [10, 1000, 5, ...]

    # 전체 데이터 개수 (N)
    total_samples = len(train_df)

    # Negative(0) 개수 계산
    negative_counts = total_samples - positive_counts

    # pos_weight 공식 변경
    # 공식: (Negative 개수) / (Positive 개수)
    # 의미: "Positive가 희귀할수록(분모가 작을수록) 가중치를 높게 줘라"
    # 0으로 나누기 방지 위해 1e-6 대신, clip이나 max 사용
    pos_weights = np.sqrt(negative_counts / (positive_counts + 1e-6))

    # 너무 큰 가중치(폭발 위험) 방지: 최대 100배까지만 허용 (선택사항이지만 안전함)
    pos_weights = np.clip(pos_weights, 1.0, 50.0)
    # pos_weights = np.maximum(pos_weights, 1.0)

    print(f"클래스별 Positive 수 (일부): {positive_counts.astype(int)}")
    print(f"수정된 pos_weight (일부): {pos_weights[:5].round(2)}")
    # 예상 결과: [88.4, 8.8, 100.0, ...] 처럼 1보다 큰 값들이 나와야 함

    # PyTorch 텐서로 변환
    class_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)

    # 테스트 데이터프레임 생성
    print(f"\n--- 평가(test) 데이터 처리: {eval_csv} ---")
    test_df = filter_dataframe(eval_csv, eval_audio_dir)
    if test_df is None:
        print("경고: 평가(eval) 데이터를 찾지 못했습니다. 최종 테스트가 불가능합니다.")
        test_df = pd.DataFrame()  # 빈 DF라도 반환

    print(f"필터링된 총 데이터: (Train: {len(train_df)} / Eval: {len(test_df)})")
    print("-------------------------- 데이터 준비 완료 --------------------------")

    return train_df, val_df, test_df, label2id, id2label, class_weights_tensor