# train.py
# 모델 학습(Training) 관련 함수

import audiomentations as A
import evaluate
import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

import config
from dataset import FSD50KSubsetDataset


def train_model(
        model,
        train_df,
        eval_df,
        audio_dir,
        feature_extractor,
        output_dir,
        learning_rate,
        num_train_epochs,
        batch_size,
        gradient_accumulation_steps
):
    """
    데이터셋을 생성하고 Trainer를 설정하여 모델 파인튜닝을 실행.
    """
    print("--- 3. 모델 학습 시작 ---")

    # 데이터 증강 파이프라인
    train_augment_pipeline = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ])

    # 데이터셋 인스턴스 생성
    train_dataset = FSD50KSubsetDataset(
        train_df, audio_dir, feature_extractor, augment=train_augment_pipeline
    )
    eval_dataset = FSD50KSubsetDataset(
        eval_df, audio_dir, feature_extractor, augment=None
    )

    # 평가 지표 정의
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=1)

        # 2. Accuracy 계산
        accuracy = accuracy_metric.compute(
            predictions=predictions, references=labels
        )["accuracy"]

        # 3. F1-score (macro) 계산
        # macro: 클래스별 F1을 단순 평균. 소수 클래스도 동등하게 중요
        f1_macro = f1_score(labels, predictions, average="macro")

        # 4. F1-score (weighted) 계산
        # weighted: 클래스별 F1을 샘플 수로 가중 평균.
        f1_weighted = f1_score(labels, predictions, average="weighted")

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    # 학습 인자(Arguments) - config.py에서 값 로드
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=True,
        report_to="tensorboard",
    )

    # 조기 종료 콜백
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )

    # 트레이너 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # 학습 실행
    trainer.train()

    print(f"--- 3. 모델 학습 완료 ---")
    return trainer