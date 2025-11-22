import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import json
import os
from transformers import AutoFeatureExtractor
from pathlib import Path
import librosa  # 오디오 로드용 (없으면 pip install librosa)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ==========================================
# 1. 설정 및 상수 정의
# ==========================================

# 모델 파일이 있는 디렉토리 (HuggingFace 다운로드 경로)
MODEL_DIR = "./model_download"

# 카테고리별 임계값 (Safety는 민감하게, 나머지는 보수적으로)
CATEGORY_THRESHOLDS = {
    'Safety': 0.40,  # 비명, 유리창 깨짐 등
    'Household': 0.65,  # 생활 소음
    'Human': 0.70,  # 말소리 등
    'Animal': 0.70,  # 동물 소리
    'Etc': 0.50  # 기타
}

# 클래스 -> 카테고리 매핑 정보
CLASS_CATEGORY_MAP = {
    'Alarm': 'Safety', 'Boom': 'Safety', 'Crushing': 'Safety', 'Crying': 'Safety',
    'Fire': 'Safety', 'Glass': 'Safety', 'Gunshot': 'Safety', 'Respiratory': 'Safety',
    'Screaming': 'Safety', 'Siren': 'Safety', 'Vehicle_horn': 'Safety',

    'Boiling': 'Household', 'Clock': 'Household', 'Keys': 'Household', 'Dishes': 'Household',
    'Door': 'Household', 'Doorbell': 'Household', 'Frying': 'Household', 'Furniture': 'Household',
    'Hiss': 'Household', 'Knock': 'Household', 'Microwave_oven': 'Household',
    'Telephone': 'Household', 'Toilet': 'Household', 'Typing': 'Household', 'Water': 'Household',

    'Clapping': 'Human', 'Footsteps': 'Human', 'Running': 'Human',
    'Sneeze': 'Human', 'Speech': 'Human', 'Laughter': 'Human',

    'Bird': 'Animal', 'Cat': 'Animal', 'Dog': 'Animal',

    'Tools': 'Etc', 'Thunder': 'Etc'
}

# 전처리기 로드 (한 번만 로드)
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
except Exception as e:
    print(f"Warning: 로컬에서 feature_extractor 로드 실패. 인터넷에서 기본 설정을 가져옵니다. ({e})")
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


# id2label 로드 함수
def load_id2label(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return {int(k): v for k, v in config.get("id2label", {}).items()}
    return {}


ID2LABEL = load_id2label(MODEL_DIR)


# ==========================================
# 2. 헬퍼 함수 (전처리, 후처리)
# ==========================================

def sigmoid(x):
    """Multi-label 확률 계산을 위한 Sigmoid"""
    return 1 / (1 + np.exp(-x))


def prepare_input(wav_path: str):
    # Librosa를 사용하여 오디오 로드 (자동 리샘플링)
    try:
        audio, _ = librosa.load(wav_path, sr=feature_extractor.sampling_rate)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.zeros((1, 1024, 128), dtype=np.float32)

    # Feature Extractor 적용
    inputs = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="np",
        padding="max_length",  # 1024 프레임 길이 맞춤
        max_length=1024,
        truncation=True
    )

    # (Batch, Time, Freq) 형태로 변환 및 타입 캐스팅
    input_values = inputs["input_values"].astype(np.float32)
    return input_values


def post_process_output(logits):
    """로짓 -> Sigmoid -> 임계값 적용 -> 최종 결과 반환"""
    probs = sigmoid(logits).flatten()  # 1차원 배열로 변환

    detected = []

    for idx, prob in enumerate(probs):
        label = ID2LABEL.get(idx, f"Class_{idx}")
        category = CLASS_CATEGORY_MAP.get(label, 'Etc')
        threshold = CATEGORY_THRESHOLDS.get(category, 0.5)

        if prob >= threshold:
            detected.append({
                "label": label,
                "category": category,
                "prob": float(prob)
            })

    # 확률 높은 순 정렬
    detected.sort(key=lambda x: x['prob'], reverse=True)

    if not detected:
        return "None", "None"

    # 가장 중요한 이벤트(Safety 우선) 또는 확률 1순위 반환
    # 여기서는 단순히 확률 1순위를 반환하도록 함.
    top_event = detected[0]
    return top_event['label'], top_event['category']


# ==========================================
# 3. TensorRT 추론 클래스
# ==========================================

class TRTInference:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT Engine not found: {engine_path}")

        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.tensors = [None] * self.engine.num_io_tensors

    def infer(self, input_array: np.ndarray):
        # 입력 바인딩 (Index 0)
        input_idx = 0
        if self.engine.get_tensor_mode(self.engine[input_idx]) != trt.TensorIOMode.INPUT:
            # 만약 0번이 입력이 아니라면 이름을 찾아야 함 (일반적으로 0번임)
            pass

        self.context.set_input_shape(self.engine[input_idx], input_array.shape)

        # 버퍼 할당 및 주소 매핑
        for idx in range(self.engine.num_io_tensors):
            if self.tensors[idx] is None:
                name = self.engine[idx]
                shape = self.context.get_tensor_shape(name)
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                # GPU 메모리 할당
                buf = cuda.mem_alloc(size * dtype().nbytes)
                self.tensors[idx] = buf
                self.context.set_tensor_address(name, int(buf))

        # Host -> Device 복사
        cuda.memcpy_htod_async(self.tensors[0], input_array.ravel(), self.stream)

        # 비동기 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Device -> Host 복사 준비
        output_idx = 1
        output_name = self.engine[output_idx]
        output_shape = tuple(self.context.get_tensor_shape(output_name))
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))

        host_output = np.empty(output_shape, dtype=output_dtype)
        cuda.memcpy_dtoh_async(host_output, self.tensors[output_idx], self.stream)

        # 동기화
        self.stream.synchronize()
        return host_output


# ==========================================
# 4. 실행용 래퍼 함수
# ==========================================

_runner_cache = None


def _get_runner(model_path: str):
    global _runner_cache
    if _runner_cache is None:
        print(f"Loading TRT Engine from {model_path}...")
        _runner_cache = TRTInference(model_path)
    return _runner_cache


def classify_noise(wav_path: str,
                   model_path: str = os.path.join(MODEL_DIR, "model.trt")
                   ) -> tuple[str, str]:
    """
    외부에서 호출하는 메인 함수
    Return: (Label, Category) 예: ('Glass', 'Safety')
    """
    runner = _get_runner(model_path)

    # 1. 전처리
    input_tensor = prepare_input(wav_path)

    # 2. 추론
    logits = runner.infer(input_tensor)

    # 3. 후처리 (Sigmoid + Threshold)
    label, category = post_process_output(logits)

    return label, category


if __name__ == "__main__":
    # 테스트 코드
    test_wav = "/home/nvidia/turtlebot3_ws/recordings/test_audio.wav"  # 테스트할 파일 경로

    if os.path.exists(test_wav):
        lbl, cat = classify_noise(test_wav)
        print(f"Detected Result: {lbl} (Category: {cat})")
    else:
        print("Test file not found.")