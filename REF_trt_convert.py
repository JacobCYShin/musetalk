import os
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from models import Wav2Lip

def export_to_onnx(checkpoint_path, onnx_path):
    """
    PyTorch 모델을 ONNX 형식으로 변환
    """
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 예제 입력 데이터: img_batch (이미지), mel_batch (음성 특징)
    dummy_img = torch.randn(1, 6, 256, 256)  # (batch, channels, height, width)
    dummy_mel = torch.randn(1, 1, 80, 16)  # (batch, channels, mel_bins, mel_frames)

    torch.onnx.export(
        model,
        (dummy_mel, dummy_img),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["mel", "img"],
        output_names=["output"],
        dynamic_axes={
            "mel": {0: "batch_size"},  # batch_size에 대해 동적 처리
            "img": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
    )
    print(f"ONNX 모델이 {onnx_path}에 저장되었습니다.")

def build_trt_engine(onnx_path, engine_path, max_batch_size=1024):
    """
    ONNX 파일을 기반으로 TensorRT 엔진 생성
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # TensorRT Builder 및 Network 생성
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 파일 파싱
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: ONNX 파일을 파싱하는 데 실패했습니다.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Optimization Profile 설정
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * (1 << 30)  # 4GB 워크스페이스 크기 설정

    profile = builder.create_optimization_profile()
    profile.set_shape("mel", (1, 1, 80, 16), (8, 1, 80, 16), (max_batch_size, 1, 80, 16))  # mel 크기 설정
    profile.set_shape("img", (1, 6, 256, 256), (8, 6, 256, 256), (max_batch_size, 6, 256, 256))  # img 크기 설정
    config.add_optimization_profile(profile)

    # FP16 최적화 활성화
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 엔진 생성 및 저장
    print("TensorRT 엔진 생성 중...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("ERROR: TensorRT 엔진 생성 실패!")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT 엔진이 {engine_path}에 저장되었습니다.")
    return engine

def infer_with_trt(engine_path, mel_input, img_input):
    """
    TensorRT 엔진을 사용하여 추론 수행
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 입력 크기 설정
    context.set_binding_shape(0, mel_input.shape)
    context.set_binding_shape(1, img_input.shape)

    # 입력/출력 바인딩
    bindings = []
    inputs, outputs = [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    # 입력 데이터 복사
    np.copyto(inputs[0][0], mel_input.ravel())
    np.copyto(inputs[1][0], img_input.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    cuda.memcpy_htod_async(inputs[1][1], inputs[1][0], stream)

    # 추론 실행
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # 출력 복사 및 반환
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
    return outputs[0][0].reshape(context.get_binding_shape(2))

if __name__ == "__main__":
    # 파일 경로 설정
    checkpoint_path = "./[CHECKPOINT]_250116_L1_GAN_KRW_scracth_exp1/checkpoint_step000924000.pth"
    onnx_path = "wav2lip_KRW_b1024.onnx"
    engine_path = "wav2lip_KRW_b1024.trt"

    # 1. PyTorch -> ONNX 변환
    export_to_onnx(checkpoint_path, onnx_path)

    # 2. ONNX -> TensorRT 변환
    build_trt_engine(onnx_path, engine_path)

    # 3. TensorRT 추론 (테스트 입력)
    dummy_mel = np.random.randn(1, 1, 80, 16).astype(np.float32)
    dummy_img = np.random.randn(1, 6, 256, 256).astype(np.float32)

    output = infer_with_trt(engine_path, dummy_mel, dummy_img)
    print("추론 결과:", output)