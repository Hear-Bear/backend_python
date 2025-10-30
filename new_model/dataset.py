# dataset.py
# FSD50KSubsetDataset 클래스와 prepare_data 함수 정의

import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split


class FSD50KSubsetDataset(Dataset):
    """
    FSD50K 오디오 파일 로드, 증강, 피처 추출을 담당하는 커스텀 데이터셋
    """

    def __init__(self, df, audio_dir, feature_extractor, augment=None):
        self.df = df
        self.audio_dir = audio_dir
        self.feature_extractor = feature_extractor
        self.target_sr = feature_extractor.sampling_rate
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, f"{row['file_name']}.wav")

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
            max_length=1024
        )
        input_values = inputs.input_values.squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(row['label'], dtype=torch.long)
        }


def prepare_data(vocab_csv, metadata_csv, target_names, audio_dir):
    """
    CSV 파일을 로드하고, 원하는 클래스만 필터링하여
    train/eval 데이터프레임과 레이블 맵을 반환.
    """
    print("--- 1. 데이터 준비 시작 ---")
    try:
        vocab_df = pd.read_csv(vocab_csv, index_col=1)
        target_mids = [vocab_df.loc[name, 'mids'] for name in target_names]
    except Exception as e:
        print(f"오류: {e}. 'TARGET_CLASS_NAMES'가 'vocabulary.csv'와 일치하는지 확인하세요.")
        return None, None, None, None

    label2id = {name: i for i, name in enumerate(target_names)}
    id2label = {i: name for i, name in enumerate(target_names)}
    mids2labelid = {mid: label2id[name] for mid, name in zip(target_mids, target_names)}

    try:
        df = pd.read_csv(metadata_csv)
    except FileNotFoundError:
        print(f"오류: {metadata_csv} 파일을 찾을 수 없습니다.")
        return None, None, None, None

    filtered_data = []
    for _, row in df.iterrows():
        file_name = row['fname']
        mids_in_file = row['mids'].split(',')
        found_label_id = None
        for mid in mids_in_file:
            if mid in mids2labelid:
                found_label_id = mids2labelid[mid]
                break
        if found_label_id is not None:
            if os.path.exists(f"{audio_dir}/{file_name}.wav"):
                filtered_data.append({"file_name": file_name, "label": found_label_id})

    if not filtered_data:
        print("오류: 필터링된 데이터가 0개입니다.")
        return None, None, None, None

    filtered_df = pd.DataFrame(filtered_data)

    train_df, eval_df = train_test_split(
        filtered_df, test_size=0.2, random_state=42, stratify=filtered_df['label']
    )

    print(f"필터링된 총 데이터: {len(filtered_df)}개 (Train: {len(train_df)} / Eval: {len(eval_df)})")
    print("--- 1. 데이터 준비 완료 ---")
    return train_df, eval_df, label2id, id2label