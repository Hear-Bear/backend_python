import os

import evaluate
import numpy as np
import pandas as pd
from transformers import ASTForAudioClassification, TrainingArguments, Trainer
from transformers.utils import logging

from dataset import load_dataset_esc50, split_by_fold, get_latest_checkpoint

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"


def squeeze_func(data):
    data['input_values'] = np.asarray(data['input_values']).squeeze(0)
    return data


# Train 데이터셋(증강 O)
dataset, folds = load_dataset_esc50(apply_augmentation=True)
dataset = dataset.map(squeeze_func)
check_dim = np.array(dataset['input_values'])
print(f"0차원 삭제: {check_dim.shape}")

# # Eval 데이터셋(증강 X)
# eval_dataset = load_dataset_esc50(apply_augmentation=False)


# pytorch에서 구현한 scaled_dot_product_attention이 적용된 모델
model = ASTForAudioClassification.from_pretrained(model_name, attn_implementation="sdpa")

# Logging 설정
logging.set_verbosity_debug()

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./output",
    logging_dir="./logs",
    logging_steps=50,
    logging_strategy="steps",
    do_train=True,
    do_eval=True,
    eval_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    # save_steps=500,
    save_total_limit=1,
    save_strategy='epoch',
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

metric = evaluate.load('accuracy')


# 평가 지표: 정확도(accuracy) 사용
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


# 학습 시도
trainer = None
num_experiments = 10

# 결과 저장용 리스트들
results_list = []  # 하나의 리스트에 모든 결과를 저장
# pred_results_dict = {}  # 마지막에 predict 결과를 저장할 수 있음

# 모델 저장 경로
saved_model_path = './output/augmented_esc50'

for i in range(num_experiments):
    print(f'학습 {i + 1} 시작.\n')

    # 매 학습마다 fold 기준으로 train, eval 데이터셋 생성
    train_dataset, eval_dataset = split_by_fold(dataset, folds)

    # Trainer 인스턴스를 새로 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # 체크포인트 존재 시 자동 불러오기 (없으면 처음부터 학습)
    checkpoint_path = get_latest_checkpoint("./output")
    if checkpoint_path:
        print(f'체크포인트에서 이어서 학습: {checkpoint_path}')
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print('체크포인트가 없습니다. 처음부터 학습합니다.')
        train_result = trainer.train()

    # 학습 종료 후 별도로 evaluate() 호출
    eval_result = trainer.evaluate()

    # 실험별 결과를 합쳐서 하나의 사전으로 생성 (키에 접두사 붙이기)
    exp_result = {'experiment_id': i + 1}
    # train_result.metrics는 TrainOutput의 metrics (예: {"loss": ..., "learning_rate": ...})로 가정합니다.
    for key, value in dict(train_result.metrics).items():
        exp_result[f"train_{key}"] = value
    for key, value in dict(eval_result).items():
        exp_result[f"eval_{key}"] = value

    results_list.append(exp_result)

    print(f"Experiment {i + 1} 결과: {exp_result}")
    print(f'학습 {i + 1} 완료.\n')

trainer.save_model(saved_model_path)

# 최종 증강 기법 적용 안된 원본 데이터셋으로 test
if os.path.exists(saved_model_path):
    test_dataset, _ = load_dataset_esc50(apply_augmentation=False)
    pred_result = trainer.predict(test_dataset)

    pred_results = {
        'prediction': True,
        "prediction_loss": pred_result.metrics.get("eval_loss", None),
        "prediction_accuracy": pred_result.metrics.get("eval_accuracy", None),
        "predictions_shape": str(pred_result.predictions.shape)
    }
    results_list.append(pred_results)
    print("Test 예측 결과:", pred_result)
else:
    print("저장된 모델 경로가 존재하지 않습니다.")

# DataFrame으로 변환 후 csv 저장
df_results = pd.DataFrame(results_list)
df_results.to_csv("./output/logs_results_augmented_esc50.csv", index=False)
print('종료.')
