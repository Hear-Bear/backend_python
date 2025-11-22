# config.py
# 모든 경로, 상수, 하이퍼파라미터를 여기에 저장합니다.
import os

# 데이터셋 경로
FSD50K_DEV_AUDIO_DIR = "D:/dataset/FSD50K/dev_audio"
FSD50K_EVAL_AUDIO_DIR = "D:/dataset/FSD50K/eval_audio"

# 메타데이터
FSD50K_DEV_CSV = "D:/dataset/FSD50K/ground_truth/dev.csv"
FSD50K_EVAL_CSV = "D:/dataset/FSD50K/ground_truth/eval.csv"

# 데이터 증강용 데이터 경로
BACKGROUND_NOISE_DIR = "D:/dataset/augmentation/background_noise_16k"
IMPULSE_RESPONSE_DIR = "D:/dataset/augmentation/impulse_response_16k"

# 타겟 클래스
CLASS_GROUPS = {
    # 안전 및 위급 상황 (Safety)
    'Alarm': ['Alarm', 'Buzzer'],
    'Boom': ['Boom', 'Explosion'],
    'Crushing': ['Crushing'],
    'Crying': ['Crying_and_sobbing'],
    'Fire': ['Fire', 'Crackle'],
    'Glass': ['Glass', 'Shatter'],
    'Gunshot': ['Gunshot_and_gunfire'],
    'Thunder': ['Thunder', 'Thunderstorm'], # 사이렌 등과 소리가 비슷해, 오탐 방지용
    'Respiratory': ['Breathing', 'Gasp', 'Sigh'], # 과호흡 등의 상황
    'Screaming': ['Screaming', 'Shout'],
    'Siren': ['Siren'],
    'Vehicle_horn': ['Vehicle_horn_and_car_horn_and_honking'],

    # 집안일 및 생활 (Household)
    'Boiling': ['Boiling'],
    'Clock': ['Clock', 'Tick', 'Tick-tock'], # 만약 이 소리가 우선적으로 감지된다면, 매우 조용한 상태
    'Keys': ['Keys_jangling'], # 외출/귀가 소리 감지
    'Dishes': ['Dishes_and_pots_and_pans', 'Cutlery_and_silverware'],
    'Door': ['Sliding_door', 'Slam'],
    'Doorbell': ['Doorbell'],
    'Frying': ['Frying_(food)'],
    'Furniture': ['Cupboard_open_or_close', 'Drawer_open_or_close'], # 수납장/서랍 여닫는 소리
    'Hiss': ['Hiss'], # 가스 소리, 물/바람 등의 소리와 구분이 어려워 오탐 방지를 위해 household에 배치
    'Knock': ['Knock'],
    'Microwave_oven': ['Microwave_oven'],
    'Telephone': ['Telephone', 'Ringtone'],
    'Toilet': ['Toilet_flush'],
    'Typing': ['Typing', 'Computer_keyboard', 'Typewriter'],
    'Water': ['Water', 'Sink_(filling_or_washing)', 'Splash_and_splatter', 'Stream'],

    # 사람 (Human)
    'Clapping': ['Clapping', 'Applause', 'Hands'],
    'Footsteps': ['Walk_and_footsteps'],
    'Running': ['Run'],
    'Sneeze': ['Sneeze', 'Cough'],
    'Speech': ['Speech', 'Child_speech_and_kid_speaking', 'Female_speech_and_woman_speaking', 'Male_speech_and_man_speaking'],
    'Laughter': ['Giggle', 'Laughter', 'Chuckle_and_chortle'],

    # 동물 (Animal)
    'Bird': ['Bird', 'Bird_vocalization_and_bird_call_and_bird_song'],
    'Cat': ['Cat', 'Meow', 'Purr'],
    'Dog': ['Dog', 'Bark'],
    
    # 기타
    'Tools': ['Drill', 'Hammer', 'Power_tool', 'Sawing', 'Tools'] # 건설/수리 등 매우 시끄러운 상황
}

CATEGORY_THRESHOLDS = {
    'Safety': 0.40,      # [중요] 놓치면 안 됨. 오탐이 있더라도 일단 알람을 띄움.
    'Household': 0.65,   # [일상] 너무 자주 울리면 시끄러우니 확실할 때만.
    'Human': 0.70,       # [사람] 말소리 등은 잡음일 확률이 높으므로 기준을 높임.
    'Animal': 0.70,      # [동물] 짖는 소리 등.
    'Etc': 0.50,         # [기타] 분류하기 어려운 것들
}

# 클래스별 카테고리 매핑 (User Defined)
CLASS_CATEGORY_MAP = {
    # --- Safety (기준: 0.35) ---
    'Alarm': 'Safety',
    'Boom': 'Safety',
    'Crushing': 'Safety',
    'Crying': 'Safety',
    'Fire': 'Safety',
    'Glass': 'Safety',
    'Gunshot': 'Safety',
    'Respiratory': 'Safety',
    'Screaming': 'Safety',
    'Siren': 'Safety',
    'Vehicle_horn': 'Safety',

    # --- Household (기준: 0.60) ---
    'Boiling': 'Household',
    'Clock': 'Household',
    'Keys': 'Household',
    'Dishes': 'Household',
    'Door': 'Household',
    'Doorbell': 'Household',
    'Frying': 'Household',
    'Furniture': 'Household',
    'Hiss': 'Household',
    'Knock': 'Household',
    'Microwave_oven': 'Household',
    'Telephone': 'Household',
    'Toilet': 'Household',
    'Typing': 'Household',
    'Water': 'Household',

    # --- Human (기준: 0.70) ---
    'Clapping': 'Human',
    'Footsteps': 'Human',
    'Running': 'Human',
    'Sneeze': 'Human',
    'Speech': 'Human',
    'Laughter': 'Human',

    # --- Animal (기준: 0.60) ---
    'Bird': 'Animal',
    'Cat': 'Animal',
    'Dog': 'Animal',

    # --- Tools (기준: 0.50) ---
    'Tools': 'Etc',
    'Thunder': 'Etc',
}

# 3. 모델 및 출력 설정
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUTPUT_DIR = "./ast-finetuned-fsd50k-focal-loss-weight-max-50"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# 4. 학습 하이퍼파라미터
MAX_AUDIO_LENGTH = 1024
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 10
BATCH_SIZE_PER_DEVICE = 4
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 3
VALIDATION_SPLIT_SIZE = 0.1
RANDOM_SEED = 42