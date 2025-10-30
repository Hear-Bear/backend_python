# main.py
# 모든 모듈을 임포트하여 파이프라인을 실행한다.

import config
from dataset import prepare_data
from model import load_model_and_extractor
from train import train_model
from test import test_single_audio
import argparse


def main(args):
    # 전체 파인튜닝 파이프라인을 순서대로 실행.

    # 1. 데이터 준비 (dataset.py)
    train_df, eval_df, label2id, id2label = prepare_data(
        config.FSD50K_VOCAB_CSV,
        config.FSD50K_METADATA_CSV,
        config.TARGET_CLASS_NAMES,
        config.FSD50K_AUDIO_DIR
    )

    if train_df is None:
        print("데이터 준비에 실패하여 프로그램을 종료합니다.")
        return

    # 2. 모델 로드 (model.py)
    num_labels = len(config.TARGET_CLASS_NAMES)
    model, feature_extractor = load_model_and_extractor(
        config.MODEL_ID, num_labels, label2id, id2label
    )

    # 3. 모델 학습 (train.py)
    trainer = train_model(
        model,
        train_df,
        eval_df,
        config.FSD50K_AUDIO_DIR,
        feature_extractor,
        args.output_dir,
        args.learning_rate,
        args.num_train_epochs,
        args.batch_size,
        args.gradient_accumulation_steps
    )

    # 4. 베스트 모델 저장
    best_model_dir = f"{args.output_dir}/best_model"
    print(f"학습 완료. 베스트 모델을 {best_model_dir}에 저장합니다.")
    trainer.save_model(best_model_dir)
    # 피처 추출기도 함께 저장해야 추론 시 동일한 전처리 보장
    feature_extractor.save_pretrained(best_model_dir)

    # 5. 베스트 모델로 전체 평가 데이터셋 최종 평가
    print("\n--- 5. 베스트 모델 최종 평가 (전체 Eval Set) ---")
    eval_metrics = trainer.evaluate()
    print(f"최종 평가 결과: {eval_metrics}")

    # 6. 테스트(추론) 실행 (test.py)
    print("\n--- 6. 단일 파일 추론 테스트 ---")
    if not eval_df.empty:
        test_file_name = eval_df.iloc[0]['file_name']
        test_file_path = f"{config.FSD50K_AUDIO_DIR}/{test_file_name}.wav"

        test_single_audio(
            model_path=config.BEST_MODEL_DIR,
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

    args = parser.parse_args()  # 5. 커맨드 라인 인자 파싱

    main(args)  # 6. 파싱된 인자를 main 함수에 전달