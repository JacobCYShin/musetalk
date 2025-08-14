#!/usr/bin/env python3
"""
MuseTalk PyTorch 모델을 ONNX로 변환하는 스크립트

이 스크립트는 다음 모델들을 ONNX로 변환합니다:
1. UNet2DConditionModel (메인 추론 모델)
2. VAE Encoder (이미지 → latents)
3. VAE Decoder (latents → 이미지)
4. PositionalEncoding (오디오 특성 인코딩)

사용법:
    python convert_to_onnx.py

출력:
    ./onnx_models/ 디렉토리에 ONNX 모델들이 저장됩니다.
"""

import torch
import torch.onnx
import numpy as np
import os
import json
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import WhisperModel

# MuseTalk 모델 import
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.models.unet import PositionalEncoding

# 설정
ONNX_DIR = "./onnx_models"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_onnx_dir():
    """ONNX 모델 저장 디렉토리 생성"""
    os.makedirs(ONNX_DIR, exist_ok=True)
    print(f"📁 ONNX models will be saved to: {ONNX_DIR}")

class UNetWrapper(torch.nn.Module):
    """UNet 모델을 ONNX 변환용으로 래핑"""
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        
    def forward(self, latent_model_input, timesteps, encoder_hidden_states):
        """
        Args:
            latent_model_input: [batch_size, 8, 32, 32] - concatenated latents
            timesteps: [batch_size] - timestep tensor
            encoder_hidden_states: [batch_size, seq_len, 384] - audio features
        
        Returns:
            sample: [batch_size, 4, 32, 32] - predicted latents
        """
        return self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

class VAEEncoderWrapper(torch.nn.Module):
    """VAE Encoder를 ONNX 변환용으로 래핑"""
    def __init__(self, vae):
        super().__init__()
        self.vae_encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.scaling_factor = vae.config.scaling_factor
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, 256, 256] - input image
        
        Returns:
            latents: [batch_size, 4, 32, 32] - encoded latents
        """
        h = self.vae_encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # 샘플링 대신 mean 사용 (deterministic)
        latents = mean * self.scaling_factor
        return latents

class VAEDecoderWrapper(torch.nn.Module):
    """VAE Decoder를 ONNX 변환용으로 래핑"""
    def __init__(self, vae):
        super().__init__()
        self.vae_decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.scaling_factor = vae.config.scaling_factor
        
    def forward(self, latents):
        """
        Args:
            latents: [batch_size, 4, 32, 32] - input latents
        
        Returns:
            image: [batch_size, 3, 256, 256] - decoded image
        """
        latents = latents / self.scaling_factor
        z = self.post_quant_conv(latents)
        image = self.vae_decoder(z)
        return image

class PositionalEncodingWrapper(torch.nn.Module):
    """PositionalEncoding을 ONNX 변환용으로 래핑"""
    def __init__(self, pe_model):
        super().__init__()
        self.pe = pe_model
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, 384] - input audio features
        
        Returns:
            x: [batch_size, seq_len, 384] - positionally encoded features
        """
        return self.pe(x)

def convert_unet_to_onnx():
    """UNet 모델을 ONNX로 변환"""
    print("🔄 Converting UNet to ONNX...")
    
    # UNet 모델 로드
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=DEVICE,
    )
    
    # Float32 사용 (ONNX 호환성을 위해)
    unet.model = unet.model.float().to(DEVICE)
    unet_wrapper = UNetWrapper(unet.model).eval()
    
    # 샘플 입력 생성
    batch_size = 1
    latent_model_input = torch.randn(batch_size, 8, 32, 32, dtype=torch.float32, device=DEVICE)
    timesteps = torch.tensor([0], dtype=torch.long, device=DEVICE)
    encoder_hidden_states = torch.randn(batch_size, 10, 384, dtype=torch.float32, device=DEVICE)
    
    # ONNX 변환
    onnx_path = os.path.join(ONNX_DIR, "unet.onnx")
    
    with torch.no_grad():
        torch.onnx.export(
            unet_wrapper,
            (latent_model_input, timesteps, encoder_hidden_states),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['latent_model_input', 'timesteps', 'encoder_hidden_states'],
            output_names=['sample'],
            dynamic_axes={
                'latent_model_input': {0: 'batch_size'},
                'timesteps': {0: 'batch_size'},
                'encoder_hidden_states': {0: 'batch_size', 1: 'sequence'},
                'sample': {0: 'batch_size'}
            }
        )
    
    print(f"✅ UNet saved to: {onnx_path}")
    return onnx_path

def convert_vae_to_onnx():
    """VAE Encoder/Decoder를 ONNX로 변환"""
    print("🔄 Converting VAE to ONNX...")
    
    # VAE 모델 로드
    vae = AutoencoderKL.from_pretrained("./models/sd-vae")
    vae = vae.float().to(DEVICE).eval()
    
    # VAE Encoder 변환
    vae_encoder_wrapper = VAEEncoderWrapper(vae)
    
    batch_size = 1
    input_image = torch.randn(batch_size, 3, 256, 256, dtype=torch.float32, device=DEVICE)
    
    encoder_path = os.path.join(ONNX_DIR, "vae_encoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            vae_encoder_wrapper,
            input_image,
            encoder_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['latents'],
            dynamic_axes={
                'input_image': {0: 'batch_size'},
                'latents': {0: 'batch_size'}
            }
        )
    
    print(f"✅ VAE Encoder saved to: {encoder_path}")
    
    # VAE Decoder 변환
    vae_decoder_wrapper = VAEDecoderWrapper(vae)
    
    input_latents = torch.randn(batch_size, 4, 32, 32, dtype=torch.float32, device=DEVICE)
    
    decoder_path = os.path.join(ONNX_DIR, "vae_decoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            vae_decoder_wrapper,
            input_latents,
            decoder_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_latents'],
            output_names=['output_image'],
            dynamic_axes={
                'input_latents': {0: 'batch_size'},
                'output_image': {0: 'batch_size'}
            }
        )
    
    print(f"✅ VAE Decoder saved to: {decoder_path}")
    return encoder_path, decoder_path

def convert_positional_encoding_to_onnx():
    """PositionalEncoding을 ONNX로 변환"""
    print("🔄 Converting PositionalEncoding to ONNX...")
    
    pe = PositionalEncoding(d_model=384).to(DEVICE).eval()
    pe_wrapper = PositionalEncodingWrapper(pe)
    
    batch_size = 1
    seq_len = 10
    input_features = torch.randn(batch_size, seq_len, 384, dtype=torch.float32, device=DEVICE)
    
    pe_path = os.path.join(ONNX_DIR, "positional_encoding.onnx")
    with torch.no_grad():
        torch.onnx.export(
            pe_wrapper,
            input_features,
            pe_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['encoded_features'],
            dynamic_axes={
                'input_features': {0: 'batch_size', 1: 'sequence'},
                'encoded_features': {0: 'batch_size', 1: 'sequence'}
            }
        )
    
    print(f"✅ PositionalEncoding saved to: {pe_path}")
    return pe_path

def verify_onnx_models():
    """변환된 ONNX 모델들을 검증"""
    print("🔍 Verifying ONNX models...")
    
    try:
        import onnx
        
        models = [
            "unet.onnx",
            "vae_encoder.onnx", 
            "vae_decoder.onnx",
            "positional_encoding.onnx"
        ]
        
        for model_name in models:
            model_path = os.path.join(ONNX_DIR, model_name)
            if os.path.exists(model_path):
                onnx_model = onnx.load(model_path)
                onnx.checker.check_model(onnx_model)
                print(f"✅ {model_name} is valid")
                
                # 모델 크기 확인
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   Size: {size_mb:.1f} MB")
            else:
                print(f"❌ {model_name} not found")
                
    except ImportError:
        print("⚠️  onnx package not installed. Install with: pip install onnx")
    except Exception as e:
        print(f"❌ Verification failed: {e}")

def create_onnx_config():
    """ONNX 모델 설정 파일 생성"""
    config = {
        "models": {
            "unet": {
                "path": "unet.onnx",
                "inputs": ["latent_model_input", "timesteps", "encoder_hidden_states"],
                "outputs": ["sample"],
                "input_shapes": {
                    "latent_model_input": ["batch_size", 8, 32, 32],
                    "timesteps": ["batch_size"],
                    "encoder_hidden_states": ["batch_size", "sequence", 384]
                }
            },
            "vae_encoder": {
                "path": "vae_encoder.onnx",
                "inputs": ["input_image"],
                "outputs": ["latents"],
                "input_shapes": {
                    "input_image": ["batch_size", 3, 256, 256]
                }
            },
            "vae_decoder": {
                "path": "vae_decoder.onnx",
                "inputs": ["input_latents"],
                "outputs": ["output_image"],
                "input_shapes": {
                    "input_latents": ["batch_size", 4, 32, 32]
                }
            },
            "positional_encoding": {
                "path": "positional_encoding.onnx",
                "inputs": ["input_features"],
                "outputs": ["encoded_features"],
                "input_shapes": {
                    "input_features": ["batch_size", "sequence", 384]
                }
            }
        },
        "version": "v15",
        "precision": "fp16",
        "opset_version": 17
    }
    
    config_path = os.path.join(ONNX_DIR, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Config saved to: {config_path}")

def main():
    """메인 변환 함수"""
    print("🚀 Starting MuseTalk PyTorch → ONNX conversion")
    print(f"🔧 Device: {DEVICE}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Conversion will be slower.")
    
    # 준비
    ensure_onnx_dir()
    
    try:
        # 모델 변환
        convert_unet_to_onnx()
        convert_vae_to_onnx()
        convert_positional_encoding_to_onnx()
        
        # 검증
        verify_onnx_models()
        
        # 설정 파일 생성
        create_onnx_config()
        
        print("\n🎉 ONNX conversion completed successfully!")
        print(f"📁 All models saved in: {ONNX_DIR}")
        print("\n📋 Next steps:")
        print("   1. Install ONNXRuntime: pip install onnxruntime-gpu")
        print("   2. Run: python fastapi_realtime_onnx.py")
        print("   3. Compare performance with PyTorch version")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
