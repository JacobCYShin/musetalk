#!/usr/bin/env python3
"""
MuseTalk PyTorch 모델을 Float16 ONNX로 변환 (성능 최적화)

PyTorch와 동일한 float16 정밀도로 변환하여 성능을 극대화합니다.
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
from musetalk.models.unet import PositionalEncoding

# 설정
ONNX_DIR = "./onnx_models_fp16"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_onnx_dir():
    """ONNX 모델 저장 디렉토리 생성"""
    os.makedirs(ONNX_DIR, exist_ok=True)
    print(f"📁 Float16 ONNX models will be saved to: {ONNX_DIR}")

class UNetWrapper(torch.nn.Module):
    """UNet 모델을 ONNX 변환용으로 래핑 (Float16 최적화)"""
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        
    def forward(self, latent_model_input, timesteps, encoder_hidden_states):
        return self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae_encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.scaling_factor = vae.config.scaling_factor
        
    def forward(self, x):
        h = self.vae_encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latents = mean * self.scaling_factor
        return latents

class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae_decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.scaling_factor = vae.config.scaling_factor
        
    def forward(self, latents):
        latents = latents / self.scaling_factor
        z = self.post_quant_conv(latents)
        image = self.vae_decoder(z)
        return image

def convert_unet_to_onnx_fp16():
    """UNet 모델을 Float16 ONNX로 변환"""
    print("🔄 Converting UNet to Float16 ONNX...")
    
    # UNet 모델 로드
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=DEVICE,
    )
    
    # Half precision 적용 (PyTorch와 동일)
    unet.model = unet.model.half().to(DEVICE)
    unet_wrapper = UNetWrapper(unet.model).eval()
    
    # 샘플 입력 생성 (Float16)
    batch_size = 1
    latent_model_input = torch.randn(batch_size, 8, 32, 32, dtype=torch.float16, device=DEVICE)
    timesteps = torch.tensor([0], dtype=torch.long, device=DEVICE)
    encoder_hidden_states = torch.randn(batch_size, 10, 384, dtype=torch.float16, device=DEVICE)
    
    # ONNX 변환
    onnx_path = os.path.join(ONNX_DIR, "unet_fp16.onnx")
    
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
    
    print(f"✅ Float16 UNet saved to: {onnx_path}")
    return onnx_path

def convert_vae_to_onnx_fp16():
    """VAE를 Float16 ONNX로 변환"""
    print("🔄 Converting VAE to Float16 ONNX...")
    
    # VAE 모델 로드
    vae = AutoencoderKL.from_pretrained("./models/sd-vae")
    vae = vae.half().to(DEVICE).eval()  # Float16 사용
    
    # VAE Encoder 변환
    vae_encoder_wrapper = VAEEncoderWrapper(vae)
    
    batch_size = 1
    input_image = torch.randn(batch_size, 3, 256, 256, dtype=torch.float16, device=DEVICE)
    
    encoder_path = os.path.join(ONNX_DIR, "vae_encoder_fp16.onnx")
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
    
    print(f"✅ Float16 VAE Encoder saved to: {encoder_path}")
    
    # VAE Decoder 변환
    vae_decoder_wrapper = VAEDecoderWrapper(vae)
    input_latents = torch.randn(batch_size, 4, 32, 32, dtype=torch.float16, device=DEVICE)
    
    decoder_path = os.path.join(ONNX_DIR, "vae_decoder_fp16.onnx")
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
    
    print(f"✅ Float16 VAE Decoder saved to: {decoder_path}")
    return encoder_path, decoder_path

def convert_positional_encoding_to_onnx():
    """PositionalEncoding을 ONNX로 변환"""
    print("🔄 Converting PositionalEncoding to ONNX...")
    
    pe = PositionalEncoding(d_model=384).to(DEVICE).eval()
    
    batch_size = 1
    seq_len = 10
    input_features = torch.randn(batch_size, seq_len, 384, dtype=torch.float32, device=DEVICE)
    
    pe_path = os.path.join(ONNX_DIR, "positional_encoding.onnx")
    with torch.no_grad():
        torch.onnx.export(
            pe,
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

def main():
    """메인 변환 함수"""
    print("🚀 Starting MuseTalk PyTorch → Float16 ONNX conversion")
    print("⚡ Optimized for speed (same precision as PyTorch)")
    print(f"🔧 Device: {DEVICE}")
    
    # 준비
    ensure_onnx_dir()
    
    try:
        # 모델 변환
        convert_unet_to_onnx_fp16()
        convert_vae_to_onnx_fp16()
        convert_positional_encoding_to_onnx()
        
        # 설정 파일 생성
        config = {
            "models": {
                "unet": {
                    "path": "unet_fp16.onnx",
                    "precision": "fp16"
                },
                "vae_encoder": {
                    "path": "vae_encoder_fp16.onnx", 
                    "precision": "fp16"
                },
                "vae_decoder": {
                    "path": "vae_decoder_fp16.onnx",
                    "precision": "fp16"
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
        print("\n🎉 Float16 ONNX conversion completed!")
        print("⚡ This should be much faster than Float32 version")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
