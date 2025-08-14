#!/usr/bin/env python3
"""
MuseTalk Realtime Optimized API 테스트 스크립트

사용 전 준비:
1. 먼저 normal 모드로 avatar 준비: sh inference.sh v1.5 normal
2. 최적화된 서버 시작: python fastapi_realtime_optimized.py
3. 이 테스트 스크립트 실행: python test_realtime_optimized.py
"""

import requests
import os
import json

# API 서버 URL
API_URL = "http://localhost:8000"

def test_health_check():
    """서버 상태 확인"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ 서버 상태 확인:")
            print(f"   - Status: {data['status']}")
            print(f"   - Version: {data['version']}")
            print(f"   - Mode: {data.get('mode', 'unknown')}")
            print(f"   - Model Ready: {data['model_ready']}")
            return True
        else:
            print(f"❌ 서버 상태 확인 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return False

def test_list_avatars():
    """사용 가능한 avatar 목록 확인"""
    try:
        response = requests.get(f"{API_URL}/avatars")
        if response.status_code == 200:
            data = response.json()
            avatars = data.get('avatars', [])
            
            print(f"📋 사용 가능한 Avatar 목록 ({len(avatars)}개):")
            if not avatars:
                print("   ⚠️  사용 가능한 avatar가 없습니다.")
                print("   💡 먼저 normal 모드로 avatar를 생성하세요: sh inference.sh v1.5 normal")
                return []
            
            ready_avatars = []
            for avatar in avatars:
                avatar_id = avatar['avatar_id']
                is_ready = avatar.get('ready', False)
                status = "✅ Ready" if is_ready else "❌ Not Ready"
                print(f"   - {avatar_id}: {status}")
                
                if is_ready:
                    ready_avatars.append(avatar_id)
                    info = avatar.get('info', {})
                    if info:
                        print(f"     📹 Video: {info.get('video_path', 'N/A')}")
                        print(f"     🔢 Version: {info.get('version', 'N/A')}")
            
            return ready_avatars
        else:
            print(f"❌ Avatar 목록 조회 실패: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Avatar 목록 조회 중 오류: {e}")
        return []

def test_realtime_inference(avatar_id, audio_file_path):
    """최적화된 실시간 추론 테스트"""
    if not os.path.exists(audio_file_path):
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file_path}")
        return False
    
    try:
        # 요청 데이터 준비
        data = {
            "avatar_id": avatar_id,
            "batch_size": 20,
            "fps": 25,
            "skip_save_images": False
        }
        
        # 오디오 파일 업로드
        with open(audio_file_path, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_file_path), f, "audio/wav")}
            
            print("🚀 최적화된 실시간 추론 요청 중...")
            print(f"   - Avatar ID: {avatar_id}")
            print(f"   - Audio: {audio_file_path}")
            print(f"   - Mode: Realtime Optimized (기존 데이터 로드)")
            
            response = requests.post(f"{API_URL}/realtime-infer", data=data, files=files)
            
            if response.status_code == 200:
                # 결과 비디오 저장
                output_filename = f"{avatar_id}_realtime_optimized_result.mp4"
                with open(output_filename, "wb") as out_file:
                    out_file.write(response.content)
                
                print(f"✅ 최적화된 추론 완료! 결과 저장됨: {output_filename}")
                print(f"📊 파일 크기: {len(response.content) / 1024 / 1024:.2f} MB")
                return True
            else:
                print(f"❌ 추론 실패: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   오류 내용: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   응답: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ 추론 중 오류 발생: {e}")
        return False

def main():
    print("=== MuseTalk Realtime Optimized API 테스트 ===")
    print("💡 이 모드는 기존에 처리된 얼굴 프레임과 detection 정보를 로드하여")
    print("   메인 모델 inference만 수행하는 최적화된 모드입니다.")
    print()
    
    # 1. 서버 상태 확인
    if not test_health_check():
        print()
        print("💡 서버를 시작하려면: python fastapi_realtime_optimized.py")
        return
    
    print()
    
    # 2. Avatar 목록 확인
    print("📋 Avatar 목록 확인 중...")
    ready_avatars = test_list_avatars()
    
    if not ready_avatars:
        print()
        print("💡 사용법:")
        print("   1. 먼저 normal 모드로 avatar 생성: sh inference.sh v1.5 normal")
        print("   2. 그 다음 이 최적화된 서버 사용")
        return
    
    print()
    
    # 3. 추론 테스트
    print("🧪 추론 테스트를 시작합니다...")
    
    # 첫 번째 사용 가능한 avatar 사용
    test_avatar_id = ready_avatars[0]
    
    # 테스트 오디오 파일 경로들 (우선순위 순)
    test_audio_paths = [
        "data/audio/yongen.wav",
        "data/audio/eng.wav",
        "test.wav",
        "sample.wav"
    ]
    
    test_audio_path = None
    for path in test_audio_paths:
        if os.path.exists(path):
            test_audio_path = path
            break
    
    if test_audio_path:
        success = test_realtime_inference(test_avatar_id, test_audio_path)
        if success:
            print()
            print("🎉 테스트 완료!")
            print("⚡ 최적화된 realtime 모드의 장점:")
            print("   - 얼굴 프레임 추출 과정 생략")
            print("   - Face detection 과정 생략") 
            print("   - 기존 처리된 데이터 재사용")
            print("   - 빠른 inference 시간")
    else:
        print("⚠️  테스트용 오디오 파일을 찾을 수 없습니다.")
        print("   다음 경로 중 하나에 오디오 파일을 배치하세요:")
        for path in test_audio_paths:
            print(f"   - {path}")

if __name__ == "__main__":
    main()
