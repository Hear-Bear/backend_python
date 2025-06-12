import tensorrt as trt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str, max_batch_size: int = 1, fp16: bool = True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1) ONNX 파싱
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for err in range(parser.num_errors):
                print(parser.get_error(err))
            sys.exit('ONNX parsing failed')

    # 2) 빌더 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30  # 1GB
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 3) 최적화 프로필 추가
    input_name = network.get_input(0).name
    
    input_tensor = network.get_input(0)
    print("input_name: ", input_tensor.name)
    print("input_shape: ", tuple(input_tensor.shape))

    profile = builder.create_optimization_profile()
    profile.set_shape(
                input_name,
                min=(1, 1024, 128),
                opt=(4, 1024, 128),
                max=(8, 1024, 128)
            )
    config.add_optimization_profile(profile)

    # 4) 엔진 빌드 및 저장
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        sys.exit("Falied to build serialized network")

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f'Serialiezed TRT engine saved to {engine_path}')

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        sys.exit("Failed to deserialize CUDA engine.")
    
    return engine

if __name__ == "__main__":
    onnx_model = "./model/model.onnx"
    trt_engine  = "./model/model.trt"
    build_engine(onnx_model, trt_engine)

