# config.py
# 모든 경로, 상수, 하이퍼파라미터를 여기에 저장합니다.
import os

# 1. 데이터셋 경로
FSD50K_AUDIO_DIR = "D:/dataset/FSD50K.dev_audio"
FSD50K_METADATA_CSV = "D:/dataset/FSD50K.ground_truth/dev.csv"
FSD50K_VOCAB_CSV = "D:/dataset/FSD50K.metadata/vocabulary.csv"

# 2. 타겟 클래스
TARGET_CLASS_NAMES = [
    "Siren", "Knock", "Doorbell", "Dog", "Hiss", "Alarm"
]

# 3. 모델 및 출력 설정
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUTPUT_DIR = "./ast-finetuned-fsd50k-custom"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# 4. 학습 하이퍼파라미터
MAX_AUDIO_LENGTH = 1024
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 10
BATCH_SIZE_PER_DEVICE = 4
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 3