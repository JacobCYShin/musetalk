curl.exe -X POST "http://127.0.0.1:8000/realtime-infer" -F "avatar_id=test_avatar" -F "video_path=./data/video/sun.mp4" -F "audio_file=@C:/CODE/MuseTalk/data/audio/sun.wav" --output result.mp4

curl.exe -X POST "http://127.0.0.1:8000/realtime-infer" -F "audio_file=@C:/CODE/MuseTalk/data/audio/yongen.wav" --output result.mp4



python3 fastapi_realtime_optimized_fast.py


curl.exe -X POST "http://localhost:8000/onnx-infer" -F "audio_file=@C:/CODE/MuseTalk/data/audio/sun.wav"
