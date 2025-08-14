#!/usr/bin/env python3
"""
MuseTalk ONNX vs PyTorch 성능 벤치마크

PyTorch와 ONNX 버전의 성능을 비교합니다.

사용법:
    python benchmark_onnx_vs_pytorch.py

출력:
    - 각 모델의 inference 시간
    - FPS 비교
    - 메모리 사용량
    - 속도 향상 비율
"""

import requests
import time
import os
import subprocess
import psutil
import torch

def check_server_health(url):
    """서버 상태 확인"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def measure_inference_time(url, endpoint, audio_file, iterations=3):
    """inference 시간 측정"""
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio_file': (os.path.basename(audio_file), f, 'audio/wav')}
                response = requests.post(f"{url}/{endpoint}", files=files, timeout=60)
            
            if response.status_code == 200:
                end_time = time.time()
                inference_time = end_time - start_time
                times.append(inference_time)
                print(f"  Iteration {i+1}: {inference_time:.2f}s")
            else:
                print(f"  Iteration {i+1}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"  Iteration {i+1}: Error - {e}")
    
    return times

def get_gpu_memory():
    """GPU 메모리 사용량 조회"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def start_server(script_name, port=8000):
    """서버 시작"""
    print(f"🚀 Starting {script_name}...")
    
    # 기존 프로세스 종료
    try:
        subprocess.run(["pkill", "-f", script_name], check=False)
        time.sleep(2)
    except:
        pass
    
    # 새 서버 시작
    process = subprocess.Popen(
        ["python", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 서버 준비 대기
    url = f"http://localhost:{port}"
    max_wait = 30
    wait_time = 0
    
    while wait_time < max_wait:
        if check_server_health(url):
            print(f"✅ {script_name} ready!")
            return process, url
        time.sleep(1)
        wait_time += 1
    
    print(f"❌ {script_name} failed to start")
    process.terminate()
    return None, None

def run_benchmark():
    """벤치마크 실행"""
    print("🏁 MuseTalk ONNX vs PyTorch Benchmark")
    print("=" * 50)
    
    # 테스트 오디오 파일
    test_audio_files = [
        "data/audio/sun.wav",
        "data/audio/yongen.wav",
        "data/audio/eng.wav"
    ]
    
    # 사용 가능한 오디오 파일 찾기
    audio_file = None
    for audio in test_audio_files:
        if os.path.exists(audio):
            audio_file = audio
            break
    
    if not audio_file:
        print("❌ No test audio files found!")
        print("Please ensure one of these files exists:")
        for audio in test_audio_files:
            print(f"   - {audio}")
        return
    
    print(f"🎵 Using audio file: {audio_file}")
    print()
    
    results = {}
    
    # ---- PyTorch 벤치마크 ----
    print("📊 Testing PyTorch version...")
    pytorch_process, pytorch_url = start_server("fastapi_realtime_optimized.py")
    
    if pytorch_process and pytorch_url:
        pytorch_times = measure_inference_time(
            pytorch_url, "fast-infer", audio_file, iterations=3
        )
        
        if pytorch_times:
            results['pytorch'] = {
                'times': pytorch_times,
                'avg_time': sum(pytorch_times) / len(pytorch_times),
                'min_time': min(pytorch_times),
                'memory_gb': get_gpu_memory()
            }
            print(f"✅ PyTorch avg: {results['pytorch']['avg_time']:.2f}s")
        
        pytorch_process.terminate()
        time.sleep(3)
    
    # ---- ONNX 벤치마크 ----
    print("\n📊 Testing ONNX version...")
    
    # ONNX 모델 존재 확인
    if not os.path.exists("./onnx_models/unet.onnx"):
        print("❌ ONNX models not found!")
        print("Please run: python convert_to_onnx.py")
        return
    
    onnx_process, onnx_url = start_server("fastapi_realtime_onnx.py")
    
    if onnx_process and onnx_url:
        onnx_times = measure_inference_time(
            onnx_url, "onnx-infer", audio_file, iterations=3
        )
        
        if onnx_times:
            results['onnx'] = {
                'times': onnx_times,
                'avg_time': sum(onnx_times) / len(onnx_times),
                'min_time': min(onnx_times),
                'memory_gb': get_gpu_memory()
            }
            print(f"✅ ONNX avg: {results['onnx']['avg_time']:.2f}s")
        
        onnx_process.terminate()
    
    # ---- 결과 분석 ----
    print("\n" + "=" * 50)
    print("🏆 BENCHMARK RESULTS")
    print("=" * 50)
    
    if 'pytorch' in results and 'onnx' in results:
        pytorch_avg = results['pytorch']['avg_time']
        onnx_avg = results['onnx']['avg_time']
        speedup = pytorch_avg / onnx_avg
        
        print(f"📈 Performance Comparison:")
        print(f"   PyTorch: {pytorch_avg:.2f}s (avg)")
        print(f"   ONNX:    {onnx_avg:.2f}s (avg)")
        print(f"   Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '🐌'}")
        print()
        
        print(f"⚡ Best Times:")
        print(f"   PyTorch: {results['pytorch']['min_time']:.2f}s")
        print(f"   ONNX:    {results['onnx']['min_time']:.2f}s")
        print()
        
        print(f"💾 Memory Usage:")
        print(f"   PyTorch: {results['pytorch']['memory_gb']:.2f}GB")
        print(f"   ONNX:    {results['onnx']['memory_gb']:.2f}GB")
        print()
        
        # 추천
        if speedup > 1.2:
            print("✅ RECOMMENDATION: Use ONNX for better performance!")
        elif speedup > 0.8:
            print("⚖️  RECOMMENDATION: Both versions have similar performance")
        else:
            print("⚠️  RECOMMENDATION: PyTorch might be better for your setup")
            
    else:
        print("❌ Benchmark incomplete. Check server logs for errors.")
    
    print("\n" + "=" * 50)

def main():
    """메인 함수"""
    print("🔧 System Information:")
    print(f"   CPU: {psutil.cpu_count()} cores")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("   GPU: Not available")
    
    print()
    
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
