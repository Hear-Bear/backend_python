# main.py
# 모든 모듈을 임포트하여 파이프라인을 실행한다.

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

import config
from dataset import prepare_data, FSD50KSubsetDataset
from model import load_model_and_extractor
from test import test_single_audio
from train import train_model


def get_threshold_for_class(class_name):
    # 클래스 이름을 받아서 config에 설정된 최적의 임계값을 반환하는 함수

    # 특정 클래스 강제 설정 확인 (Override)
    if hasattr(config, 'CLASS_SPECIFIC_THRESHOLDS'):
        if class_name in config.CLASS_SPECIFIC_THRESHOLDS:
            return config.CLASS_SPECIFIC_THRESHOLDS[class_name]

    # 카테고리 매핑 확인
    category = config.CLASS_CATEGORY_MAP.get(class_name, 'Default')

    # 카테고리별 임계값 반환
    return config.CATEGORY_THRESHOLDS.get(category, 0.5)


def main(args):
    # 전체 파인튜닝 파이프라인을 순서대로 실행.

    # 데이터 준비 (dataset.py)
    train_df, val_df, test_df, label2id, id2label, class_weights = prepare_data(
        config.FSD50K_DEV_CSV,
        config.FSD50K_EVAL_CSV,
        config.CLASS_GROUPS,
        config.FSD50K_DEV_AUDIO_DIR,
        config.FSD50K_EVAL_AUDIO_DIR,
        config.VALIDATION_SPLIT_SIZE,
        config.RANDOM_SEED
    )

    if train_df is None:
        print("데이터 준비에 실패하여 프로그램을 종료합니다.")
        return

    # 모델 로드 (model.py)
    num_labels = len(config.CLASS_GROUPS)
    model, feature_extractor = load_model_and_extractor(
        config.MODEL_ID, num_labels, label2id, id2label
    )

    # 데이터셋 로드
    train_dataset = FSD50KSubsetDataset(
        train_df,
        config.FSD50K_DEV_AUDIO_DIR,
        feature_extractor,
        use_augmentation=True  # True로 설정하여 증강 사용
    )
    val_dataset = FSD50KSubsetDataset(
        val_df,
        config.FSD50K_DEV_AUDIO_DIR,
        feature_extractor,
        use_augmentation=False
    )
    test_dataset = FSD50KSubsetDataset(
        test_df,
        config.FSD50K_EVAL_AUDIO_DIR,
        feature_extractor,
        use_augmentation=False
    )

    # 모델 학습 (train.py)
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # feature_extractor=feature_extractor,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        class_weights=class_weights,
    )

    # 베스트 모델 저장
    best_model_dir = f"{args.output_dir}/best_model"
    print(f"학습 완료. 베스트 모델을 {best_model_dir}에 저장합니다.")

    trainer.save_model(best_model_dir)

    # 피처 추출기도 함께 저장해야 추론 시 동일한 전처리 보장
    feature_extractor.save_pretrained(best_model_dir)

    # # 최적 임계값 튜닝(Validation Set 사용)
    # print("\n--- 최적 임계값 튜닝 (Validation Set) ---")
    #
    # # (val_dataset이 이미 로드되어 있음)
    # val_predictions = trainer.predict(val_dataset)
    # val_logits = val_predictions.predictions
    # val_labels = val_predictions.label_ids
    #
    # # [디버깅] 데이터 확인
    # print(f"[Debug] Logits Shape: {val_logits.shape}")
    # print(f"[Debug] Labels Shape: {val_labels.shape}")
    #
    # # NaN 확인
    # if np.isnan(val_logits).any():
    #     print("🚨 [치명적 오류] Logits에 NaN 값이 포함되어 있습니다! 학습이 폭발했습니다.")
    #     print("-> 해결책: Learning Rate를 낮추거나(1e-5), Gradient Accumulation을 줄이세요.")
    # else:
    #     print(f"[Debug] Logits 예시 (Top 5): {val_logits[0][:5]}")
    #     print(f"[Debug] Logits Min: {val_logits.min()}, Max: {val_logits.max()}")
    #
    # # Labels 확인
    # print(f"[Debug] Labels 예시 (Top 5): {val_labels[0][:5]}")
    #
    # # NumPy -> Torch Tensor로 변환
    # val_logits_tensor = torch.from_numpy(val_logits)
    # # Sigmoid 적용
    # val_probs = F.sigmoid(val_logits_tensor).numpy()
    #
    # print(f"[Debug] Probabilities 예시 (Top 5): {val_probs[0][:5]}")
    #
    # best_threshold = 0.5
    # best_f1_macro = 0.0
    #
    # # 0.1부터 0.9까지 0.05 스텝으로 탐색
    # threshold_candidates = np.arange(0.1, 0.9, 0.05)
    #
    # print("임계값 탐색 중...")
    # for threshold in threshold_candidates:
    #     preds = (val_probs > threshold).astype(int)
    #     f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
    #
    #     # 만약 모든 예측이 0이라면?
    #     if preds.sum() == 0:
    #         # print(f"임계값 {threshold:.2f}: 예측된 Positive가 하나도 없습니다.")
    #         pass
    #
    #     if f1 > best_f1_macro:
    #         best_f1_macro = f1
    #         best_threshold = threshold
    #
    # print(f"최적 임계값: {best_threshold:.2f} (F1 Macro: {best_f1_macro:.4f})")

    print("\n--- 카테고리 기반 임계값 적용 준비 ---")

    # 모델의 클래스 개수
    num_classes = len(id2label)

    # (클래스 개수,) 형태의 임계값 배열 생성
    manual_thresholds = np.zeros(num_classes)
    threshold_log = {}  # 로그 저장용

    print("적용된 임계값 (일부):")
    for i in range(num_classes):
        class_name = id2label[i]

        # 작성한 함수를 통해 임계값 가져오기
        th = get_threshold_for_class(class_name)

        manual_thresholds[i] = th
        threshold_log[class_name] = th

        # 너무 많으니 앞부분만 출력
        if i < 5:
            print(f" - {class_name}: {th}")

    # 베스트 모델로 전체 평가 데이터셋 최종 평가
    print("\n--- 베스트 모델 최종 평가 ---")

    final_results = {
        "config": {
            "model_id": config.MODEL_ID,
            "epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "threshold_type": "Category-based Model"
        }
    }

    if not test_df.empty:
        test_dataset = FSD50KSubsetDataset(
            test_df,
            config.FSD50K_EVAL_AUDIO_DIR,
            feature_extractor,
            use_augmentation=False
        )

        # 모델 예측 실행
        test_predictions = trainer.predict(test_dataset)
        test_logits = test_predictions.predictions
        test_labels = test_predictions.label_ids

        # Sigmoid 적용 & 최적 임계값으로 0/1 변환
        test_logits_tensor = torch.from_numpy(test_logits)
        test_probs = F.sigmoid(test_logits_tensor).numpy()

        final_preds = (test_probs > manual_thresholds).astype(int)

        # [전체 지표] 계산 및 저장
        test_metrics = {
            "accuracy_subset": accuracy_score(test_labels, final_preds),
            "f1_macro": f1_score(test_labels, final_preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(test_labels, final_preds, average="weighted", zero_division=0)
        }
        final_results["test_metrics"] = test_metrics  # 딕셔너리에 추가

        # [클래스별 지표] 계산 및 저장
        # average=None을 주면 각 클래스별 점수가 리스트로 나옵니다.
        per_class_f1 = f1_score(test_labels, final_preds, average=None, zero_division=0)

        class_performance = {}
        for i, score in enumerate(per_class_f1):
            class_name = id2label[i]  # ID(0) -> 이름(Alarm) 변환
            th = manual_thresholds[i]

            class_performance[class_name] = {
                'f1_score': float(score),
                'threshold': float(th),
            }

        # 점수가 높은 순서대로 정렬해서 저장
        sorted_performance = dict(sorted(class_performance.items(), key=lambda item: item[1]['f1_score'], reverse=True))
        final_results["per_class_f1"] = sorted_performance

        # 터미널 출력
        print(f"최종 테스트 결과: {test_metrics}")
        print("상위 5개 클래스 성능:", list(sorted_performance.items())[:5])
        print("하위 5개 클래스 성능:", list(sorted_performance.items())[-5:])

    else:
        print("Test Set이 없어 최종 평가를 건너뜁니다.")
        final_results["error"] = "No Test Set found"

    # 파일로 저장(JSON)
    save_path = os.path.join(args.output_dir, "final_results.json")

    # JSON 파일 쓰기
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n모든 평가 결과가 '{save_path}' 파일에 저장되었습니다.")

    # 테스트(추론) 실행 (test.py)
    print("\n--- 단일 파일 추론 테스트 ---")
    if not test_df.empty:
        test_file_name = test_df.iloc[0]['fname']
        test_file_path = f"{config.FSD50K_EVAL_AUDIO_DIR}/{test_file_name}.wav"

        test_single_audio(
            model_path=best_model_dir,
            audio_file=test_file_path,
            target_sr=feature_extractor.sampling_rate,
            id2label=id2label
        )


# 스크립트 실행
if __name__ == "__main__":
    # 4. argparse 파서 설정
    parser = argparse.ArgumentParser(description="AST 모델 파인튜닝 스크립트")

    # config.py의 값들을 'default'로 사용
    parser.add_argument(
        '--output_dir',
        type=str,
        default=config.OUTPUT_DIR,
        help='모델 출력 및 로그 저장 디렉토리'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=config.LEARNING_RATE,
        help='학습률'
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=config.NUM_TRAIN_EPOCHS,
        help='총 학습 에포크 수'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE_PER_DEVICE,
        help='디바이스당 배치 크기'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=config.GRADIENT_ACCUMULATION_STEPS,
        help='그래디언트 축적 스텝'
    )

    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=config.EARLY_STOPPING_PATIENCE,
        help='조기 종료 Paitence'
    )

    args = parser.parse_args()
    main(args)  # 파싱된 인자를 main 함수에 전달