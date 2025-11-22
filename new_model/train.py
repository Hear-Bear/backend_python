# train.py
# 모델 학습(Training) 관련 함수

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, pos_weight=None, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # 기본 BCE Loss 계산 (pos_weight 적용)
        # reduction='none'으로 하여 샘플별 Loss를 구함.
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )

        # 모델의 예측 확률 계산 (Sigmoid)
        probs = torch.sigmoid(logits)

        # p_t 계산: 정답이 1이면 p, 정답이 0이면 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal Term: (1 - p_t)^gamma
        # 잘 맞춘 샘플(p_t가 큼)일수록 loss가 0에 가까워짐
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        # Alpha Term (선택사항, 여기서는 pos_weight가 역할을 대신하므로 1.0 처리하거나 유지)
        # 여기서는 pos_weight가 이미 불균형을 잡고 있으므로 alpha는 적용하지 않거나 약하게 둠.
        # 구현의 안정성을 위해 Focal Term만 곱하여 반환.

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 커스텀 트레이너 정의
class ImbalancedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # GPU로 가중치 텐서 이동
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

        # Focal Loss 초기화
        # gamma=2.0: 어려운 샘플에 제곱만큼 더 집중
        self.loss_fct = BCEFocalLoss(gamma=2.0, pos_weight=self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 손실 함수를 커스터마이징.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 정의해둔 Focal Loss 사용
        loss = self.loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_model(
        model,
        train_dataset,
        eval_dataset,
        output_dir,
        learning_rate,
        num_train_epochs,
        batch_size,
        gradient_accumulation_steps,
        early_stopping_patience,
        class_weights
):
    # 데이터셋을 생성하고 Trainer를 설정하여 모델 학습 실행
    print("\n--- 모델 학습 시작 ---")

    # 평가 지표 정의
    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        logits_tensor = torch.from_numpy(logits)

        # sigmoid 변환
        predictions = np.round(F.sigmoid(logits_tensor).numpy())

        # F1-score (macro) 계산
        # macro: 클래스별 F1을 단순 평균. 소수 클래스도 동등하게 중요
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

        # F1-score (weighted) 계산
        # weighted: 클래스별 F1을 샘플 수로 가중 평균.
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

        # Subset Accuracy 계산
        # 파일의 모든 레이블을 정확히 다 맞춘 경우에만 1점.
        subset_accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy_subset": subset_accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    # 학습 인자(Arguments) : config.py에서 값 로드
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
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
        early_stopping_patience=early_stopping_patience
    )

    # 트레이너 정의
    trainer = ImbalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # feature_extractor=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
        class_weights=class_weights
    )

    # 학습 실행
    trainer.train()

    print(f"--- 모델 학습 완료 ---")
    return trainer