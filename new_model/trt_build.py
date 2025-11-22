import tensorrt as trt
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    print(f"Building TensorRT engine from {onnx_path}...")
    builder = trt.Builder(TRT_LOGGER)

    # Explicit Batch 모드 설정
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1) ONNX 파싱
    if not os.path.exists(onnx_path):
        sys.exit(f"Error: ONNX file not found at {onnx_path}")

    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for err in range(parser.num_errors):
                print(parser.get_error(err))
            sys.exit(1)

    # 2) 빌더 설정
    config = builder.create_builder_config()
    # 메모리 풀 설정 (구버전 TRT 호환성 고려)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    except AttributeError:
        config.max_workspace_size = 1 << 30

    # FP16 모드 설정 (Jetson 성능 향상 핵심)
    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision.")
        config.set_flag(trt.BuilderFlag.FP16)

    # 3) 최적화 프로필 설정
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Input Tensor: {input_name}, Shape: {input_tensor.shape}")

    # AST 모델 입력: (Batch, Time, Freq) = (Batch, 1024, 128)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(1, 1024, 128),  # 최소 배치
        opt=(1, 1024, 128),  # 최적 배치 (실시간 추론용 1)
        max=(4, 1024, 128)  # 최대 배치
    )
    config.add_optimization_profile(profile)

    # 4) 엔진 빌드 및 저장
    print("Building serialized network... (This may take a while)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        sys.exit("Failed to build serialized network.")

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Success! TRT engine saved to {engine_path}")


if __name__ == "__main__":
    # 경로 설정 (다운로드 받은 폴더 구조에 맞게 수정하세요)
    ONNX_MODEL = "./model_download/model.onnx"
    TRT_ENGINE = "./model_download/model.trt"

    build_engine(ONNX_MODEL, TRT_ENGINE)