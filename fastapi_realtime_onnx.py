#!/usr/bin/env python3
"""
MuseTalk ONNX Realtime FastAPI Server

ONNX 모델을 사용하여 최고 속도의 inference를 제공합니다.

특징:
- UNet, VAE Encoder/Decoder를 ONNX로 가속화
- ONNXRuntime GPU provider 사용
- test_avatar 사전 로드
- 최대 성능 최적화

사용법:
1. ONNX 모델 변환: python convert_to_onnx.py
2. 서버 실행: python fastapi_realtime_onnx.py
3. API 호출: curl -X POST "http://localhost:8000/onnx-infer" -F "audio_file=@audio.wav"
"""

import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile, os, torch
import json
import pickle
import glob
import time
import numpy as np
import cv2
from tqdm import tqdm
import copy

# ONNX Runtime
try:
    import onnxruntime as ort
    print("✅ ONNXRuntime available")
except ImportError:
    print("❌ ONNXRuntime not found. Install with: pip install onnxruntime-gpu")
    exit(1)

# MuseTalk imports
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.utils import datagen
from musetalk.utils.blending import get_image_blending
from transformers import WhisperModel
import scripts.realtime_inference as realtime_inference

# ---- 1) 환경 설정 ----
class Args:
    pass

args = Args()
args.version = "v15"
args.extra_margin = 10
args.parsing_mode = "jaw"
args.audio_padding_length_left = 2
args.audio_padding_length_right = 2
args.skip_save_images = False

# realtime_inference 모듈에 args 전역 등록
realtime_inference.args = args

# ---- 2) ONNX 설정 ----
ONNX_DIR = "./onnx_models_fp16"  # Float16 모델 사용
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_onnx_providers():
    """ONNX Runtime provider 설정"""
    providers = []
    
    if torch.cuda.is_available():
        providers.append(('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }))
        print("🚀 Using CUDA Execution Provider")
    
    providers.append('CPUExecutionProvider')
    return providers

# ONNX Provider 설정
onnx_providers = setup_onnx_providers()

class ONNXModelManager:
    """ONNX 모델들을 관리하는 클래스"""
    
    def __init__(self, onnx_dir=ONNX_DIR):
        self.onnx_dir = onnx_dir
        self.sessions = {}
        self.load_models()
    
    def load_models(self):
        """ONNX 모델들을 로드"""
        print("🔄 Loading ONNX models...")
        
        models = {
            'unet': 'unet_fp16.onnx',
            'vae_encoder': 'vae_encoder_fp16.onnx', 
            'vae_decoder': 'vae_decoder_fp16.onnx',
            'positional_encoding': 'positional_encoding.onnx'  # 이건 아직 fp32
        }
        
        for model_name, model_file in models.items():
            model_path = os.path.join(self.onnx_dir, model_file)
            if os.path.exists(model_path):
                try:
                    session = ort.InferenceSession(model_path, providers=onnx_providers)
                    self.sessions[model_name] = session
                    print(f"✅ {model_name} loaded")
                    
                    # 모델 정보 출력
                    input_names = [input.name for input in session.get_inputs()]
                    output_names = [output.name for output in session.get_outputs()]
                    print(f"   Inputs: {input_names}")
                    print(f"   Outputs: {output_names}")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
            else:
                print(f"❌ Model file not found: {model_path}")
        
        if len(self.sessions) != len(models):
            print("⚠️  Some ONNX models are missing. Run convert_to_onnx.py first")
    
    def run_unet(self, latent_model_input, timesteps, encoder_hidden_states):
        """UNet ONNX 모델 실행"""
        session = self.sessions['unet']
        
        # 입력 데이터 타입을 float16으로 변환 (fp16 모델 사용)
        inputs = {
            'latent_model_input': latent_model_input.cpu().half().numpy(),
            'timesteps': timesteps.cpu().numpy(),
            'encoder_hidden_states': encoder_hidden_states.cpu().half().numpy()
        }
        
        # 추론 실행
        outputs = session.run(['sample'], inputs)
        
        # Torch tensor로 변환
        return torch.from_numpy(outputs[0]).to(DEVICE)
    
    def run_vae_encoder(self, input_image):
        """VAE Encoder ONNX 모델 실행"""
        session = self.sessions['vae_encoder']
        
        inputs = {'input_image': input_image.cpu().half().numpy()}
        outputs = session.run(['latents'], inputs)
        
        return torch.from_numpy(outputs[0]).to(DEVICE)
    
    def run_vae_decoder(self, input_latents):
        """VAE Decoder ONNX 모델 실행"""
        session = self.sessions['vae_decoder']
        
        inputs = {'input_latents': input_latents.cpu().half().numpy()}
        outputs = session.run(['output_image'], inputs)
        
        return torch.from_numpy(outputs[0]).to(DEVICE)
    
    def run_positional_encoding(self, input_features):
        """PositionalEncoding ONNX 모델 실행"""
        session = self.sessions['positional_encoding']
        
        inputs = {'input_features': input_features.cpu().float().numpy()}
        outputs = session.run(['encoded_features'], inputs)
        
        return torch.from_numpy(outputs[0]).to(DEVICE)

# ONNX 모델 매니저 생성
onnx_manager = ONNXModelManager()

# ---- 3) 기존 모듈들 로드 (Whisper, Audio processor 등) ----
print(f"🔧 Using device: {DEVICE}")

if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not available!")

# Audio processor와 Whisper 로드 - Float16 사용 (PyTorch와 동일)
audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=DEVICE, dtype=torch.float16).eval()
whisper.requires_grad_(False)

fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

# 전역 객체를 realtime_inference 모듈에 주입 (호환성을 위해)
realtime_inference.fp = fp
realtime_inference.audio_processor = audio_processor
realtime_inference.whisper = whisper
realtime_inference.device = DEVICE

# GPU 메모리 최적화
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ---- 4) ONNX Optimized Avatar 클래스 ----
@torch.no_grad()
class ONNXOptimizedAvatar:
    """ONNX 모델을 사용하는 최적화된 Avatar 클래스"""
    
    def __init__(self, avatar_id, batch_size=8):
        self.avatar_id = avatar_id
        self.batch_size = batch_size
        
        # 경로 설정
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        
        self.load_precomputed_data()

    def load_precomputed_data(self):
        """기존 처리된 데이터를 빠르게 로드"""
        if not os.path.exists(self.avatar_path):
            raise FileNotFoundError(f"Avatar '{self.avatar_id}' does not exist.")
        
        load_start = time.time()
        print(f"⚡ Loading precomputed data for ONNX avatar: {self.avatar_id}")
        
        # 필수 파일들 체크
        required_files = [self.coords_path, self.latents_out_path, self.mask_coords_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Latents 로드
        self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location=DEVICE)
        
        # Coordinates 로드
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        
        # 이미지 경로 준비 (lazy loading)
        self.input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        # 이미지 캐시
        self.frame_cache = {}
        self.mask_cache = {}
        
        load_time = time.time() - load_start
        print(f"✅ ONNX Avatar loaded in {load_time:.2f}s")
        print(f"   - Latents: {len(self.input_latent_list_cycle)}")
        print(f"   - Frame paths: {len(self.input_img_list)}")

    def inference(self, audio_path, out_vid_name, fps=25, skip_save_images=False):
        """ONNX를 사용한 초고속 inference"""
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        total_start = time.time()
        print("🚀 Starting ONNX ultra-fast inference...")
        
        # 오디오 처리 - Float16 사용 (PyTorch와 동일)
        audio_start = time.time()
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, weight_dtype=torch.float16)
        
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, DEVICE, torch.float16, whisper,
            librosa_length, fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        audio_time = time.time() - audio_start
        print(f"⚡ Audio processing: {audio_time*1000:.1f}ms")
        
        # ONNX Inference
        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        
        inference_start = time.time()
        res_frame_list = []
        
        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)), desc="ONNX Inference")):
            
            # Positional Encoding (PyTorch fallback)
            from musetalk.models.unet import PositionalEncoding
            pe_torch = PositionalEncoding(d_model=384).to(DEVICE).eval()
            audio_feature_batch = pe_torch(whisper_batch.to(DEVICE))
            
            # UNet inference (ONNX) - float16으로 통일 (PyTorch와 동일)
            latent_batch = latent_batch.to(device=DEVICE, dtype=torch.float16)
            timesteps = torch.tensor([0], device=DEVICE)
            
            pred_latents = onnx_manager.run_unet(
                latent_batch, timesteps, audio_feature_batch
            )
            
            # VAE Decoder (ONNX) - float16으로 통일
            pred_latents = pred_latents.to(dtype=torch.float16)
            recon_images = onnx_manager.run_vae_decoder(pred_latents)
            
            # 이미지 후처리
            recon_images = (recon_images / 2 + 0.5).clamp(0, 1)
            recon_images = recon_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            recon_images = (recon_images * 255).round().astype("uint8")
            recon_images = recon_images[...,::-1]  # RGB to BGR
            
            for res_frame in recon_images:
                res_frame_list.append(res_frame)
        
        inference_time = time.time() - inference_start
        
        # 프레임 블렌딩 (필요한 경우에만)
        if not skip_save_images:
            print("🎨 Processing frame blending...")
            for idx, res_frame in enumerate(tqdm(res_frame_list, desc="Frame blending")):
                coord_idx = idx % len(self.coord_list_cycle)
                img_idx = idx % len(self.input_img_list)
                mask_idx = idx % len(self.input_mask_list)
                
                # 캐시에서 가져오거나 새로 로드
                if img_idx not in self.frame_cache:
                    self.frame_cache[img_idx] = cv2.imread(self.input_img_list[img_idx])
                if mask_idx not in self.mask_cache:
                    self.mask_cache[mask_idx] = cv2.imread(self.input_mask_list[mask_idx])
                
                bbox = self.coord_list_cycle[coord_idx]
                ori_frame = copy.deepcopy(self.frame_cache[img_idx])
                mask = self.mask_cache[mask_idx]
                
                x1, y1, x2, y2 = bbox
                try:
                    res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    mask_crop_box = self.mask_coords_list_cycle[coord_idx]
                    combine_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
                    cv2.imwrite(f"{self.avatar_path}/tmp/{str(idx).zfill(8)}.png", combine_frame)
                except Exception as e:
                    print(f"Frame {idx} processing error: {e}")
                    continue
        
        # 성능 통계
        total_time = time.time() - total_start
        fps_actual = video_num / total_time if total_time > 0 else 0
        
        print(f"🏆 ONNX Performance Summary:")
        print(f"   - Audio processing: {audio_time*1000:.1f}ms")
        print(f"   - ONNX inference: {inference_time:.2f}s")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Processed {video_num} frames")
        print(f"   - ONNX FPS: {fps_actual:.1f}")
        print(f"   - Speed ratio: {fps_actual/fps:.2f}x" if fps > 0 else "")
        
        # 비디오 생성
        if out_vid_name and not skip_save_images:
            video_start = time.time()
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            import shutil
            shutil.rmtree(f"{self.avatar_path}/tmp")
            video_time = time.time() - video_start
            print(f"🎬 Video generation: {video_time:.2f}s")

# ---- 5) 테스트용 ONNX Avatar 미리 로드 ----
print("🎭 Preloading ONNX test avatar...")
try:
    global_onnx_avatar = ONNXOptimizedAvatar(avatar_id="test_avatar", batch_size=8)
    print("✅ ONNX Test avatar loaded successfully!")
    print("🚀 Ready for ONNX ultra-fast inference!")
except Exception as e:
    print(f"⚠️  Failed to load ONNX test avatar: {e}")
    global_onnx_avatar = None

# ---- 6) FastAPI 서버 ----
app = FastAPI(
    title="MuseTalk ONNX Ultra-Fast API",
    description="ONNX 가속화된 초고속 MuseTalk inference 서버",
    version="3.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "MuseTalk ONNX Ultra-Fast API Server",
        "acceleration": "ONNX Runtime",
        "models_loaded": list(onnx_manager.sessions.keys()),
        "test_avatar_ready": global_onnx_avatar is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "acceleration": "ONNX",
        "models_loaded": list(onnx_manager.sessions.keys()),
        "test_avatar_loaded": global_onnx_avatar is not None,
        "providers": [provider for provider, _ in onnx_providers] if hasattr(ort, 'get_available_providers') else []
    }

@app.post("/onnx-infer")
async def onnx_infer(
    audio_file: UploadFile = None,
    fps: int = Form(25),
    skip_save_images: bool = Form(False),
):
    """
    ONNX 가속화된 초고속 inference API
    
    사용법:
    curl -X POST "http://localhost:8000/onnx-infer" -F "audio_file=@your_audio.wav"
    """
    try:
        if global_onnx_avatar is None:
            raise HTTPException(
                status_code=503,
                detail="ONNX test avatar not loaded. Please restart server."
            )
        
        # 오디오 파일 저장
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, audio_file.filename)
        
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        print(f"🚀 ONNX inference with test_avatar")
        print(f"🎵 Audio: {audio_file.filename}")

        # ONNX inference 실행
        global_onnx_avatar.inference(
            audio_path=audio_path,
            out_vid_name="onnx_ultra_fast",
            fps=fps,
            skip_save_images=skip_save_images,
        )

        # 결과 반환
        if not skip_save_images:
            output_file = os.path.join(global_onnx_avatar.video_out_path, "onnx_ultra_fast.mp4")
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Output file not found: {output_file}")
            
            os.remove(audio_path)
            os.rmdir(tmp_dir)
            
            return FileResponse(
                output_file,
                media_type="video/mp4",
                filename="onnx_ultra_fast.mp4"
            )
        else:
            os.remove(audio_path)
            os.rmdir(tmp_dir)
            return {"message": "ONNX ultra-fast inference completed", "avatar_id": "test_avatar"}
        
    except Exception as e:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
        
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting MuseTalk ONNX Ultra-Fast API Server")
    print("⚡ Features:")
    print("   - ONNX Runtime acceleration")
    print("   - GPU-optimized inference")
    print("   - test_avatar preloaded")
    print("   - Maximum performance")
    print("")
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("")
    print("🎯 ONNX usage:")
    print('   curl -X POST "http://localhost:8000/onnx-infer" \\')
    print('        -F "audio_file=@your_audio.wav"')
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
