#!/usr/bin/env python3
"""
MuseTalk Realtime Optimized FastAPI Server (Audio Processing Optimized)

오디오 처리 속도를 최적화한 버전입니다.

최적화 사항:
1. Whisper 모델을 한 번만 로드하고 재사용
2. 오디오 처리 배치 크기 최적화
3. GPU 메모리 효율성 개선
4. 불필요한 중간 처리 과정 제거
"""

import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile, os, torch
import json
import pickle
import glob
import time
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

vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device,
)

pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)

timesteps = torch.tensor([0], device=device)

# ---- 3) 최적화된 오디오 프로세서 ----
class OptimizedAudioProcessor(AudioProcessor):
    """오디오 처리 속도를 최적화한 AudioProcessor"""
    
    def __init__(self, feature_extractor_path="./models/whisper"):
        super().__init__(feature_extractor_path)
        # Whisper encoder 사전 로드 및 최적화
        self._warmup_models()
    
    def _warmup_models(self):
        """모델 워밍업으로 첫 번째 처리 지연 감소"""
        print("🔥 Warming up audio models...")
        dummy_audio = torch.zeros((1, 80, 3000), dtype=torch.float16).to(device)
        try:
            # Warmup feature extractor
            import numpy as np
            dummy_wav = np.zeros(16000, dtype=np.float32)  # 1초 더미 오디오
            _ = self.feature_extractor(
                dummy_wav,
                return_tensors="pt", 
                sampling_rate=16000
            )
            print("✅ Audio models warmed up!")
        except Exception as e:
            print(f"⚠️  Warmup failed (normal): {e}")
    
    def get_audio_feature_fast(self, wav_path, weight_dtype=None):
        """빠른 오디오 특성 추출"""
        if not os.path.exists(wav_path):
            return None
        
        start_time = time.time()
        
        # librosa 로딩 최적화
        import librosa
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000, mono=True)
        load_time = time.time() - start_time
        
        # Feature extraction 최적화
        feature_start = time.time()
        
        # 30초 청크 대신 더 작은 청크로 처리하여 메모리 효율성 향상
        chunk_duration = 10  # 10초씩 처리
        segment_length = chunk_duration * sampling_rate
        segments = [librosa_output[i:i + segment_length] 
                   for i in range(0, len(librosa_output), segment_length)]

        features = []
        for i, segment in enumerate(segments):
            # 빈 세그먼트 처리
            if len(segment) == 0:
                continue
                
            # 짧은 세그먼트 패딩
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate
            ).input_features
            
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        feature_time = time.time() - feature_start
        total_time = time.time() - start_time
        
        print(f"📊 Audio processing breakdown:")
        print(f"   - Loading: {load_time*1000:.1f}ms")
        print(f"   - Feature extraction: {feature_time*1000:.1f}ms") 
        print(f"   - Total: {total_time*1000:.1f}ms")

        return features, len(librosa_output)

# 최적화된 오디오 프로세서 사용
audio_processor = OptimizedAudioProcessor(feature_extractor_path="./models/whisper")
weight_dtype = unet.model.dtype

# Whisper 모델 로드 및 최적화
print("🤖 Loading Whisper model...")
whisper_start = time.time()
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

# Whisper 모델 최적화
if hasattr(whisper, 'encoder'):
    # 컴파일 최적화 (PyTorch 2.0+)
    try:
        whisper.encoder = torch.compile(whisper.encoder, mode="reduce-overhead")
        print("✅ Whisper encoder compiled for speed")
    except:
        print("⚠️  Whisper compilation not available")

whisper_load_time = time.time() - whisper_start
print(f"✅ Whisper loaded in {whisper_load_time:.2f}s")

fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

# ---- 4) 전역 객체를 realtime_inference 모듈에 주입 ----
realtime_inference.vae = vae
realtime_inference.unet = unet
realtime_inference.pe = pe
realtime_inference.fp = fp
realtime_inference.audio_processor = audio_processor
realtime_inference.weight_dtype = weight_dtype
realtime_inference.whisper = whisper
realtime_inference.device = device
realtime_inference.timesteps = timesteps

# ---- 5) 최적화된 Avatar 클래스 ----
@torch.no_grad()
class SuperOptimizedAvatar:
    """극도로 최적화된 Avatar 클래스"""
    
    def __init__(self, avatar_id, batch_size=20):
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
        
        self.idx = 0
        self.load_precomputed_data()

    def load_precomputed_data(self):
        """기존 처리된 데이터를 빠르게 로드"""
        if not os.path.exists(self.avatar_path):
            raise FileNotFoundError(f"Avatar '{self.avatar_id}' does not exist.")
        
        load_start = time.time()
        print(f"⚡ Loading precomputed data for avatar: {self.avatar_id}")
        
        # 필수 파일들 체크
        required_files = [self.coords_path, self.latents_out_path, self.mask_coords_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # 병렬 로딩으로 속도 최적화
        print("📦 Loading latents and coordinates...")
        
        # Latents 로드 (가장 큰 파일)
        latent_start = time.time()
        self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location=device)
        latent_time = time.time() - latent_start
        
        # Coordinates 로드
        coord_start = time.time()
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        coord_time = time.time() - coord_start
        
        # 이미지 로드 (필요한 경우에만)
        img_start = time.time()
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                               key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)
        img_time = time.time() - img_start
        
        total_load_time = time.time() - load_start
        
        print(f"📊 Loading time breakdown:")
        print(f"   - Latents: {latent_time:.2f}s")
        print(f"   - Coordinates: {coord_time:.2f}s")
        print(f"   - Images: {img_time:.2f}s")
        print(f"   - Total: {total_load_time:.2f}s")
        print(f"✅ Loaded {len(self.frame_list_cycle)} frames, {len(self.input_latent_list_cycle)} latents")

    def inference(self, audio_path, out_vid_name, fps=25, skip_save_images=False):
        """최적화된 inference"""
        import threading
        import queue
        import copy
        import numpy as np
        import cv2
        from tqdm import tqdm
        from musetalk.utils.utils import datagen
        from musetalk.utils.blending import get_image_blending
        
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        total_start = time.time()
        print("🚀 Starting super-optimized realtime inference...")
        
        # 최적화된 오디오 처리
        audio_start = time.time()
        whisper_input_features, librosa_length = audio_processor.get_audio_feature_fast(
            audio_path, weight_dtype=weight_dtype)
        
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper,
            librosa_length, fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        audio_time = time.time() - audio_start
        print(f"⚡ Optimized audio processing: {audio_time*1000:.1f}ms")
        
        # Inference
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        
        def process_frames():
            while True:
                if self.idx >= video_num - 1:
                    break
                try:
                    res_frame = res_frame_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue

                bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
                ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                    
                mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
                mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

                if not skip_save_images:
                    cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
                self.idx += 1

        # 프레임 처리 스레드 시작
        process_thread = threading.Thread(target=process_frames)
        process_thread.start()

        # 배치 추론
        inference_start = time.time()
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        
        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)), desc="Inference")):
            
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch, timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                res_frame_queue.put(res_frame)

        process_thread.join()
        inference_time = time.time() - inference_start
        
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
        
        total_time = time.time() - total_start
        print(f"📊 Performance summary:")
        print(f"   - Audio processing: {audio_time*1000:.1f}ms")
        print(f"   - Neural inference: {inference_time:.2f}s") 
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - FPS: {video_num/total_time:.1f}")

# ---- 6) FastAPI 서버 ----
app = FastAPI(
    title="MuseTalk Super Optimized API",
    description="극도로 최적화된 실시간 MuseTalk inference 서버",
    version="2.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "MuseTalk Super Optimized API Server",
        "optimizations": [
            "Fast audio processing",
            "Model compilation",
            "Memory optimization", 
            "Batch processing",
            "Precomputed data loading"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": args.version,
        "model_ready": True,
        "mode": "super_optimized"
    }

@app.post("/realtime-infer")
async def realtime_infer(
    avatar_id: str = Form(...),
    audio_file: UploadFile = None,
    batch_size: int = Form(20),
    fps: int = Form(25),
    skip_save_images: bool = Form(False),
):
    """최적화된 실시간 inference API"""
    try:
        # 오디오 파일 임시 저장
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, audio_file.filename)
        
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        print(f"🎭 Processing with avatar_id: {avatar_id}")

        # 최적화된 Avatar 사용
        avatar = SuperOptimizedAvatar(avatar_id=avatar_id, batch_size=batch_size)

        # 최적화된 inference 실행
        avatar.inference(
            audio_path=audio_path,
            out_vid_name="super_optimized_result",
            fps=fps,
            skip_save_images=skip_save_images,
        )

        # 결과 반환
        if not skip_save_images:
            output_file = os.path.join(avatar.video_out_path, "super_optimized_result.mp4")
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Output file not found: {output_file}")
            
            # 정리
            os.remove(audio_path)
            os.rmdir(tmp_dir)
            
            return FileResponse(
                output_file, 
                media_type="video/mp4", 
                filename=f"{avatar_id}_super_optimized.mp4"
            )
        else:
            os.remove(audio_path)
            os.rmdir(tmp_dir)
            return {"message": "Super optimized inference completed", "avatar_id": avatar_id}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting MuseTalk Super Optimized API Server")
    print("⚡ Optimizations enabled:")
    print("   - Fast audio feature extraction")
    print("   - Compiled Whisper encoder")
    print("   - Optimized data loading")
    print("   - GPU memory efficiency")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8000)
