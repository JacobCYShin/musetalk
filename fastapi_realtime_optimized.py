#!/usr/bin/env python3
"""
MuseTalk Realtime Optimized FastAPI Server

이 스크립트는 기존에 처리된 얼굴 프레임과 detection 정보를 로드하여
메인 모델 inference만 수행하는 최적화된 realtime 서버입니다.

사용법:
1. 먼저 normal 모드로 avatar를 준비: sh inference.sh v1.5 normal
2. 이 서버 실행: python fastapi_realtime_optimized.py
3. API 호출하여 realtime inference 수행

차이점:
- Normal mode: 얼굴 프레임 추출 + face detection + inference
- Realtime mode: 기존 프레임/detection 로드 + inference만 수행
"""

import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile, os, torch
import json
import pickle
import glob
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import read_imgs
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

# ---- 2) 모델 로드 ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not available! This will be very slow.")
    print("   Please ensure you have:")
    print("   - NVIDIA GPU with CUDA support")
    print("   - PyTorch with CUDA installed")

vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device,
)

# GPU 최적화 - half precision 강제 적용 (normal 모드와 동일)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)

# GPU 메모리 최적화
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"💾 Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f}GB")

timesteps = torch.tensor([0], device=device)

audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
weight_dtype = unet.model.dtype

whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

# ---- 3) 전역 객체를 realtime_inference 모듈에 주입 ----
realtime_inference.vae = vae
realtime_inference.unet = unet
realtime_inference.pe = pe
realtime_inference.fp = fp
realtime_inference.audio_processor = audio_processor
realtime_inference.weight_dtype = weight_dtype
realtime_inference.whisper = whisper
realtime_inference.device = device
realtime_inference.timesteps = timesteps

# ---- 4) Optimized Avatar 클래스 ----
@torch.no_grad()
class OptimizedAvatar:
    """
    기존에 처리된 데이터를 로드하여 inference만 수행하는 최적화된 Avatar 클래스
    """
    def __init__(self, avatar_id, batch_size=8):  # normal 모드와 동일한 배치 크기
        self.avatar_id = avatar_id
        self.batch_size = batch_size
        
        # 버전에 따른 경로 설정
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        
        self.idx = 0
        self.load_precomputed_data()

    def load_precomputed_data(self):
        """기존에 처리된 데이터를 로드합니다."""
        if not os.path.exists(self.avatar_path):
            raise FileNotFoundError(f"Avatar '{self.avatar_id}' does not exist. Please run normal mode first to create avatar data.")
        
        # avatar 정보 로드
        if not os.path.exists(self.avatar_info_path):
            raise FileNotFoundError(f"Avatar info file not found: {self.avatar_info_path}")
        
        with open(self.avatar_info_path, "r") as f:
            self.avatar_info = json.load(f)
        
        print(f"Loading precomputed data for avatar: {self.avatar_id}")
        print(f"Avatar version: {self.avatar_info.get('version', 'unknown')}")
        
        # 필수 파일들 존재 확인
        required_files = [
            self.coords_path,
            self.latents_out_path,
            self.mask_coords_path
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required precomputed file not found: {file_path}")
        
        # latents 로드
        print("Loading latents...")
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        
        # coordinates 로드
        print("Loading coordinates...")
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        
        # 마스크 coordinates 로드
        print("Loading mask coordinates...")
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        
        # 이미지 프레임 로드 (lazy loading으로 최적화)
        print("Preparing frame and mask paths...")
        self.input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        # 이미지 캐시 초기화 (필요시에만 로드)
        self.frame_cache = {}
        self.mask_cache = {}
        
        print(f"✅ Successfully loaded precomputed data:")
        print(f"   - Frame paths: {len(self.input_img_list)}")
        print(f"   - Latents: {len(self.input_latent_list_cycle)}")
        print(f"   - Coordinates: {len(self.coord_list_cycle)}")
        print(f"   - Mask paths: {len(self.input_mask_list)}")

    def inference(self, audio_path, out_vid_name, fps=25, skip_save_images=False):
        """오디오 기반 inference 수행 (preprocessing 제외)"""
        import time
        import copy
        import numpy as np
        import cv2
        from tqdm import tqdm
        from musetalk.utils.utils import datagen
        from musetalk.utils.blending import get_image_blending
        
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("🚀 Starting optimized realtime inference...")
        
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"⏱️  Audio processing time: {(time.time() - start_time) * 1000:.2f}ms")
        
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        
        # 단순화된 inference - threading 제거하고 직접 처리
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()
        
        res_frame_list = []  # 결과 프레임들을 먼저 모두 생성
        
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)), desc="Neural inference")):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            
            # 결과 프레임 리스트에 추가
            for res_frame in recon:
                res_frame_list.append(res_frame)
        
        # 이제 프레임 블렌딩 처리 (필요한 경우에만)
        if not skip_save_images:
            print("🎨 Processing frame blending...")
            for idx, res_frame in enumerate(tqdm(res_frame_list, desc="Frame blending")):
                # 필요한 이미지만 로드 (lazy loading)
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

        inference_time = time.time() - start_time
        
        # 성능 통계
        fps_actual = video_num / inference_time if inference_time > 0 else 0
        print(f"📊 Performance Summary:")
        print(f"   - Neural inference time: {inference_time:.2f}s")
        print(f"   - Processed {video_num} frames")
        print(f"   - Actual FPS: {fps_actual:.1f}")
        print(f"   - Target FPS: {fps}")
        print(f"   - Speed ratio: {fps_actual/fps:.2f}x" if fps > 0 else "")
        
        if args.skip_save_images is True:
            print(f'⚡ Total inference time (no image saving): {inference_time:.2f}s')
        else:
            print(f'🎬 Total processing time (with image saving): {inference_time:.2f}s')

        if out_vid_name is not None and args.skip_save_images is False:
            # Video generation
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print("🎥 Generating video...")
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print("🔊 Combining audio...")
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            import shutil
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"✅ Result saved to: {output_vid}")
        print("")

# ---- 5) 테스트용 Avatar 미리 로드 ----
print("🎭 Preloading test avatar...")
try:
    global_avatar = OptimizedAvatar(avatar_id="test_avatar", batch_size=8)
    print("✅ Test avatar loaded successfully!")
    print(f"📦 Cached in VRAM: {len(global_avatar.input_latent_list_cycle)} latents")
    print(f"🖼️  Cached frames: {len(global_avatar.input_img_list)}")
    print("⚡ Ready for ultra-fast inference!")
except Exception as e:
    print(f"⚠️  Failed to load test avatar: {e}")
    print("💡 Please ensure 'test_avatar' exists by running normal mode first")
    global_avatar = None

# ---- 6) FastAPI 서버 ----
app = FastAPI(
    title="MuseTalk Realtime Optimized API",
    description="최적화된 실시간 MuseTalk inference 서버 - 기존 처리된 데이터를 로드하여 inference만 수행",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "MuseTalk Realtime Optimized API Server",
        "version": args.version,
        "description": "기존에 처리된 얼굴 프레임과 detection 정보를 로드하여 메인 모델 inference만 수행"
    }

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "version": args.version,
        "model_ready": True,
        "mode": "realtime_optimized",
        "test_avatar_loaded": global_avatar is not None,
        "preloaded_latents": len(global_avatar.input_latent_list_cycle) if global_avatar else 0
    }

@app.post("/fast-infer")
async def fast_infer(audio_file: UploadFile = None):
    """
    초간편 inference API (test_avatar 전용, 최소 파라미터)
    
    사용법:
    curl -X POST "http://localhost:8000/fast-infer" -F "audio_file=@your_audio.wav"
    """
    return await realtime_infer(audio_file=audio_file, fps=25, skip_save_images=False)

@app.get("/avatars")
async def list_avatars():
    """사용 가능한 avatar 목록 조회"""
    if args.version == "v15":
        avatars_dir = f"./results/{args.version}/avatars"
    else:
        avatars_dir = "./results/avatars"
    
    if not os.path.exists(avatars_dir):
        return {"avatars": [], "message": "No avatars directory found"}
    
    avatars = []
    for item in os.listdir(avatars_dir):
        avatar_path = os.path.join(avatars_dir, item)
        if os.path.isdir(avatar_path):
            avatar_info_path = os.path.join(avatar_path, "avator_info.json")
            if os.path.exists(avatar_info_path):
                try:
                    with open(avatar_info_path, "r") as f:
                        info = json.load(f)
                    avatars.append({
                        "avatar_id": item,
                        "info": info,
                        "ready": True
                    })
                except:
                    avatars.append({
                        "avatar_id": item,
                        "ready": False,
                        "error": "Failed to load avatar info"
                    })
    
    return {"avatars": avatars}

@app.post("/realtime-infer")
async def realtime_infer(
    audio_file: UploadFile = None,
    fps: int = Form(25),
    skip_save_images: bool = Form(False),
):
    """
    초고속 실시간 inference API (test_avatar 전용)
    
    Args:
        audio_file: 입력 오디오 파일
        fps: 프레임률 (기본값: 25)
        skip_save_images: 이미지 저장 건너뛰기 (기본값: False)
    
    Note: 
        - test_avatar가 미리 VRAM에 로드되어 있어 매우 빠름
        - avatar_id와 video_path 파라미터 불필요
    """
    try:
        # 전역 avatar 사용 가능 여부 확인
        if global_avatar is None:
            raise HTTPException(
                status_code=503, 
                detail="Test avatar not loaded. Please restart server with test_avatar available."
            )
        
        # 오디오 파일 임시 저장
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, audio_file.filename)
        
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        print(f"🎭 Using preloaded test_avatar")
        print(f"🎵 Audio file: {audio_file.filename}")
        print(f"⚡ VRAM preloaded - ultra-fast inference mode!")

        # inference 실행 (미리 로드된 global_avatar 사용)
        global_avatar.inference(
            audio_path=audio_path,
            out_vid_name="ultra_fast_result",
            fps=fps,
            skip_save_images=skip_save_images,
        )

        # 결과 파일 반환
        if not skip_save_images:
            output_file = os.path.join(global_avatar.video_out_path, "ultra_fast_result.mp4")
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Output file not found: {output_file}")
            
            # 임시 파일 정리
            os.remove(audio_path)
            os.rmdir(tmp_dir)
            
            return FileResponse(
                output_file, 
                media_type="video/mp4", 
                filename=f"test_avatar_ultra_fast.mp4"
            )
        else:
            # 이미지 저장을 건너뛴 경우
            os.remove(audio_path)
            os.rmdir(tmp_dir)
            return {"message": "Ultra-fast inference completed (images not saved)", "avatar_id": "test_avatar"}
        
    except FileNotFoundError as e:
        # 임시 파일 정리
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
        
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        # 임시 파일 정리
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
        
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting MuseTalk Ultra-Fast Realtime API Server")
    print("⚡ Features:")
    print("   - test_avatar preloaded in VRAM")
    print("   - Zero loading time for inference")
    print("   - Maximum speed optimization")
    print("")
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("❤️  Health: http://localhost:8000/health")
    print("")
    print("🎯 Quick usage:")
    print('   curl -X POST "http://localhost:8000/fast-infer" \\')
    print('        -F "audio_file=@your_audio.wav"')
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
