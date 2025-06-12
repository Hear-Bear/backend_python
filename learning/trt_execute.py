import tensorrt as trt
from tensorrt import TensorIOMode
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA 컨텍스트 초기화
import torchaudio
from transformers import AutoFeatureExtractor
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


def prepare_input(wav_path: str):
    waveform, sr = torchaudio.load(wav_path)
    if sr != feature_extractor.sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=feature_extractor.sampling_rate
        )
    audio = waveform.mean(dim=0).numpy()

    inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="np")
    return inputs["input_values"].astype(np.float32)


def mapping_output(target):
    target = int(target)
    csv_path = "/home/nvidia/turtlebot3_ws/backend_python/learning/esc50_sorted.csv"
    df_esc50 = pd.read_csv(csv_path)

    target_to_category = dict(zip(df_esc50['target'], df_esc50['category']))

    return target, target_to_category.get(target, 'Unknown')


class TRTInference:
    def __init__(self, engine_path: str):
        # 1) 엔진 로드
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # 2) 실행 컨텍스트 생성
        self.context = self.engine.create_execution_context()

        # 3) CUDA 스트림 생성
        self.stream = cuda.Stream()

        # 3) 입출력 버퍼 할당
        self.tensors = [None] * self.engine.num_io_tensors

    def infer(self, input_array: np.ndarray):
        # 4) 입력 텐서 이름 가져오기
        input_name = self.engine[0]  # 바인딩 인덱스 0의 이름

        # 5) 동적 배치(shape) 설정
        # print('input_name: ', input_name, type(input_name))
        # print('input_array.shape: ', input_array.shape, type(input_array.shape))
        self.context.set_input_shape(input_name, input_array.shape)

        # 6) 각 바인딩마다 크기와 타입을 얻어서 GPU 메모리 할당
        for idx in range(self.engine.num_io_tensors):
            # 동적 shape 또는 static shape 조회
            if self.tensors[idx] is None:
                name = self.engine[idx]
                tensor_shape = self.context.get_tensor_shape(name)
                size = trt.volume(tensor_shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                buf = cuda.mem_alloc(size * dtype().nbytes)
                self.tensors[idx] = buf

                # context에 (이름, 장치 포인터) 매핑
                self.context.set_tensor_address(name, int(buf))

        # 6) 입력 데이터를 GPU로 복사
        cuda.memcpy_htod_async(self.tensors[0], input_array.ravel(), self.stream)

        # 7) 비동기 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 8) 출력 결과를 호스트로 복사
        output_name = self.engine[1]
        output_shape = tuple(self.context.get_tensor_shape(output_name))
        #항상 새로 계산
        output_dtype  = trt.nptype(self.engine.get_tensor_dtype(output_name))
        host_output = np.empty(output_shape, dtype=output_dtype)
        cuda.memcpy_dtoh_async(host_output, self.tensors[1], self.stream)

        # 9) 스트림 동기화
        self.stream.synchronize()
        
        return host_output
        
        
# ───────────────────────────────────────────────────────
# trt_execute.py 마지막 부분(또는 예제 위) 추가/수정
# ───────────────────────────────────────────────────────
_runner_cache = None                          # 전역 캐시

def _get_runner(model_path: str):
    """TRTInference 객체를 한 번만 생성해 재사용"""
    global _runner_cache
    if _runner_cache is None:
        _runner_cache = TRTInference(model_path)
    return _runner_cache


def classify_noise(wav_path: str,
                   model_path: str = str(Path(__file__).parent / "model" / "model.trt")
                   ) -> tuple[int, str]:
    """
    WAV 파일을 분석해 (target_id, category) 튜플을 반환한다.
    외부 모듈에서 import 해 쓰기 위해 만든 래퍼(runner 캐싱 포함).
    """
    runner = _get_runner(model_path)
    input_tensor = prepare_input(wav_path)          # (B, T)
    logits = runner.infer(input_tensor)
    preds = np.argmax(logits, axis=1)
    return mapping_output(preds)

# 사용 예시
if __name__ == "__main__":
    wav = "/home/nvidia/turtlebot3_ws/recordings/noise_1747577573.wav"
    target, category = classify_noise(wav)          # ← 간단!
    print(f"Target Class: {target}, Category: {category}")
