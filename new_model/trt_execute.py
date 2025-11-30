# import pycuda.autoinit  # <--- [수정] autoinit은 ROS 환경에서 위험할 수 있어 제거 권장
import json
import os

import librosa
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from transformers import AutoFeatureExtractor

# 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ==========================================
# 1. 설정 및 상수 정의
# ==========================================

# 모델 경로 (절대 경로로 지정하는 것이 안전함)
BASE_DIR = os.path.expanduser("~/turtlebot3_ws/backend_python")
MODEL_DIR = os.path.join(BASE_DIR, "model_download")

# 카테고리별 임계값 (로봇 환경에 맞게 조정됨)
CATEGORY_THRESHOLDS = {
    'Safety': 0.35,  # 위급 상황 (민감하게)
    'Household': 0.50,  # 생활 소음
    'Human': 0.55,  # 사람 소리 (로봇 노이즈 고려하여 낮춤)
    'Animal': 0.55,  # 동물 소리
    'Etc': 0.50
}

# 클래스 -> 카테고리 매핑 (동일)
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

# 전처리기 로드
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
except Exception:
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


# id2label 로드
def load_id2label(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return {int(k): v for k, v in config.get("id2label", {}).items()}
    return {}


ID2LABEL = load_id2label(MODEL_DIR)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prepare_input(wav_path: str):
    # 오디오 파일을 로드하여 모델 입력 형태(np.float32)로 변환
    try:
        # sr=None이 아니라 feature_extractor의 sr로 강제 리샘플링
        audio, _ = librosa.load(wav_path, sr=feature_extractor.sampling_rate)
    except Exception as e:
        print(f"[Error] Audio load failed: {e}")
        # 실패 시 0으로 채운 더미 데이터 반환 (Shape 주의: 1, 1024, 128)
        return np.zeros((1, 1024, 128), dtype=np.float32)

    # 특징 추출
    inputs = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="np",
        padding="max_length",
        max_length=1024,
        truncation=True
    )

    # Type Casting
    # AST 모델은 보통 (Batch, Time, Freq) 입력을 받음
    input_values = inputs["input_values"].astype(np.float32)

    return input_values


def post_process_output(logits):
    probs = sigmoid(logits).flatten()
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

    detected.sort(key=lambda x: x['prob'], reverse=True)

    if not detected:
        return "None", "None"

    # 가장 확률 높은 것 반환
    top = detected[0]
    return top['label'], top['category']


class TRTInference:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT Engine not found: {engine_path}")

        # CUDA Context 초기화
        try:
            cuda.init()
            self.device = cuda.Device(0)
            self.cuda_ctx = self.device.make_context()
        except Exception:
            # 이미 컨텍스트가 있다면 패스 (ROS 노드에서 생성한 경우)
            pass

        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # IO 버퍼 할당
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True

            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)

            # 동적 배치가 있다면 -1을 1로 가정 (여기선 고정 배치 1 가정)
            if shape[0] < 0: shape = (1,) + shape[1:]

            size = trt.volume(shape) * dtype.itemsize
            allocation = cuda.mem_alloc(size)
            self.allocations.append(int(allocation))

            binding = {
                'index': i,
                'name': name,
                'dtype': np.float32,  # AST 기본
                'shape': shape,
                'allocation': allocation,
            }

            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def infer(self, input_array: np.ndarray):
        # 입력이 하나라고 가정 (AST 모델)
        cuda.memcpy_htod_async(self.inputs[0]['allocation'], input_array.ravel(), self.stream)

        # Context에 텐서 주소 설정
        for i in range(len(self.allocations)):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.allocations[i])

        # 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 출력 복사 (Device -> Host)
        output_binding = self.outputs[0]
        output = np.empty(output_binding['shape'], dtype=output_binding['dtype'])
        cuda.memcpy_dtoh_async(output, output_binding['allocation'], self.stream)

        # 동기화
        self.stream.synchronize()
        return output

    def __del__(self):
        # 소멸자에서 컨텍스트 해제 (선택 사항)
        try:
            self.cuda_ctx.pop()
        except:
            pass


# 실행 래퍼 (Singleton)
_runner_instance = None


def get_runner():
    global _runner_instance
    if _runner_instance is None:
        model_path = os.path.join(MODEL_DIR, "model.trt")
        print(f"[TRT] Loading engine from {model_path}...")
        _runner_instance = TRTInference(model_path)
    return _runner_instance


# wav 경로를 받아 (Label, Category) 반환
def classify_noise(wav_path: str) -> tuple[str, str]:
    # 파일 존재 여부 확인
    if not os.path.exists(wav_path):
        print(f"[Error] File not found: {wav_path}")
        return "Error", "File_Not_Found"

    runner = get_runner()

    # 전처리
    input_tensor = prepare_input(wav_path)

    # 추론
    try:
        logits = runner.infer(input_tensor)

        # 후처리
        label, category = post_process_output(logits)
        return label, category

    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        return "Error", "Inference_Failed"