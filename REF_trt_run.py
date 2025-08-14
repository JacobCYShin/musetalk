from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
# import face_detection
from models import Wav2Lip
import platform
from face_parsing import init_parser, swap_regions, swap_regions_fastapi
from basicsr.apply_sr import init_sr_model, enhance
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import asyncio

# parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

# parser.add_argument('--face', type=str, default='/share/jacob/Wav2Lip-master/ordinary_1_2_front_longest_silence_interval_clip_boomerang2_upperbody_25fps_rembg.mp4',
# 					help='Filepath of video/image that contains faces to use')

# parser.add_argument('--static', type=bool, 
# 					help='If True, then use only first video frame for inference', default=False)
# parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
# 					default=25., required=False)

# parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0], 
# 					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

# parser.add_argument('--face_det_batch_size', type=int, 
# 					help='Batch size for face detection', default=16)
# parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

# parser.add_argument('--resize_factor', default=1, type=int, 
# 			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

# parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
# 					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
# 					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

# parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
# 					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
# 					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

# parser.add_argument('--rotate', default=False, action='store_true',
# 					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
# 					'Use if you get a flipped result, despite feeding a normal looking video')

# parser.add_argument('--nosmooth', default=False, action='store_true',
# 					help='Prevent smoothing face detections over a short temporal window')
# parser.add_argument('--no_segmentation', default=False, action='store_true',
# 					help='Prevent using face segmentation')
# parser.add_argument('--no_sr', default=False, action='store_true',
# 					help='Prevent using super resolution')

# parser.add_argument('--save_frames', default=False, action='store_true',
# 					help='Save each frame as an image. Use with caution')
# parser.add_argument('--gt_path', type=str, default='data/gt',
# 					help='Where to store saved ground truth frames', required=False)
# parser.add_argument('--pred_path', type=str, default='data/lq',
# 					help='Where to store frames produced by algorithm', required=False)
# parser.add_argument('--save_as_video', action="store_true", default=False,
# 					help='Whether to save frames as video', required=False)
# parser.add_argument('--image_prefix', type=str, default="",
# 					help='Prefix to save frames with', required=False)

# parser.add_argument("--margin", help="Adding margin of cropped images", default=True)

# args = parser.parse_args()
# args.img_size = 512

# 그림의 argparse를 전역 변수로 변경한 코드

# FACE = "/share/jacob/Wav2Lip-master/ordinary_1_2_front_longest_silence_interval_clip_boomerang2_upperbody_25fps_rembg.mp4"  # Filepath of video/image that contains faces to use
# FACE = "/share/jacob/wav2lip_live/preprocess_source_data/source_data/source_video"
STATIC = False  # If True, then use only first video frame for inference
FPS = 25.0  # Can be specified only if input is a static image (default: 25)
PADS = [0, 0, 0, 0]  # Padding (top, bottom, left, right). Please adjust to include chin at least
FACE_DET_BATCH_SIZE = 16  # Batch size for face detection
WAV2LIP_BATCH_SIZE = 128  # Batch size for wav2Lip model(s)
RESIZE_FACTOR = 1  # Reduce the resolution by this factor
CROP = [0, 1, 0, 1]  # Crop video to a smaller region
BOX = [-1, -1, -1, -1]  # Specify a constant bounding box for the face
ROTATE = False  # Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.
NOSMOOTH = False  # Prevent smoothing face detections over a short temporal window
NO_SEGMENTATION = False  # Prevent using face segmentation
NO_SR = True  # Prevent using super resolution
SAVE_FRAMES = False  # Save each frame as an image. Use with caution
GT_PATH = "data/gt"  # Where to store saved ground truth frames
PRED_PATH = "data/lq"  # Where to store frames produced by algorithm
SAVE_AS_VIDEO = False  # Whether to save frames as video
IMAGE_PREFIX = ""  # Prefix to save frames with
IMG_SIZE = 256
MARGIN=True

# 이후 코드에서 위의 전역 변수를 사용

# if os.path.isfile(FACE_crop) and FACE_crop.split('.')[1] in ['jpg', 'png', 'jpeg']: # 240204 필요없음
# 	STATIC = True


def infer_with_trt(engine, mel_input, img_input):
	if mel_input.shape[0] == 0 or img_input.shape[0] == 0:
		raise ValueError(f"Invalid input shapes: mel_input.shape={mel_input.shape}, img_input.shape={img_input.shape}")

	print("engine:", engine)  # None이 아닌지 확인

	# start = time.time()
	# TRT_LOGGER = trt.Logger(trt.Logger.INFO)
	# with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	# 	engine = runtime.deserialize_cuda_engine(f.read())
	# 	print("TensorRT 엔진이 성공적으로 로드되었습니다.")

	# loading_time = time.time() - start

	# print(f'Loading time : {loading_time}')

	context = engine.create_execution_context()
	# 입력 크기 설정
	print(f"[DEBUG] mel_input shape: {mel_input.shape}, img_input shape: {img_input.shape}")
	context.set_binding_shape(0, mel_input.shape)
	context.set_binding_shape(1, img_input.shape)

	# 입력/출력 바인딩
	bindings = []
	inputs, outputs = [], []
	stream = cuda.Stream()

	for i, binding in enumerate(engine):
		size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
		dtype = trt.nptype(engine.get_binding_dtype(binding))
		host_mem = cuda.pagelocked_empty(size, dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)
		bindings.append(int(device_mem))
		if engine.binding_is_input(binding):
			inputs.append((host_mem, device_mem))
		else:
			outputs.append((host_mem, device_mem))

	print(f"[DEBUG] bindings: {bindings}, inputs: {inputs}, outputs: {outputs}")

	# 입력 데이터 복사
	np.copyto(inputs[0][0], mel_input.ravel())
	np.copyto(inputs[1][0], img_input.ravel())
	cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
	cuda.memcpy_htod_async(inputs[1][1], inputs[1][0], stream)

	# 스트림 동기화
	stream.synchronize()

	# 추론 실행
	context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

	# 출력 복사 및 반환
	cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
	stream.synchronize()

	output_shape = context.get_binding_shape(2)
	print(f"[DEBUG] Output shape: {output_shape}")
	return outputs[0][0].reshape(output_shape)

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, margin):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = FACE_DET_BATCH_SIZE
	
	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = PADS
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		# y1 = 0
		# y2 = image.shape[0]
		# x1 = 0
		# x2 = image.shape[1]

		# x1, y1, x2, y2 = rect
		# w_ = y2 - y1
		# h_ = x2 - x1

		# y1 = max(0, rect[1] - int(w_ * 0.1))
		# y2 = min(image.shape[0], rect[3] + int(w_ * 0.1))
		# x1 = max(0, rect[0] - int(h_ * 0.1))
		# x2 = min(image.shape[1], rect[2] + int(h_ * 0.1))

		# y1 = max(0, rect[1] - int(w_ * 0.1))
		# y2 = min(image.shape[0], rect[3] + int(w_ * 0.1))
		# x1 = max(0, rect[0] - int(h_ * 0.1))
		# x2 = min(image.shape[1], rect[2] + int(h_ * 0.1))
		
		results.append([x1, y1, x2, y2])

	if margin == False:
		boxes = np.array(results)
	else:
		# boxes = np.array([(x1 - int((x2 - x1) * 0.1), y1 - int((y2 - y1) * 0.1), x2 + int((x2 - x1) * 0.1), y2 + int((y2 - y1) * 0.1)) for (x1, y1, x2, y2) in results])
		boxes = np.array([(x1, y1 - int((y2 - y1) * 0.12), x2, y2 + int((y2 - y1) * 0.12))  for (x1, y1, x2, y2) in results])


	if not NOSMOOTH: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	# cv2.imwrite('test.jpg', results[0][0])
	# exit()

	del detector
	return results 

def datagen(mels, src_data, margin=False):
	img_batch, mel_batch, frame_batch, coords_batch, masks_batch = [], [], [], [], []

	"""
	if BOX[0] == -1:
		if not STATIC:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = BOX
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
	"""

	reader = read_frames(src_data)

	bbx = np.load(src_data["BBX"])

	for i, m in enumerate(mels):
		try:
			frame_to_save, face, mask = next(reader)
		except StopIteration:
			reader = read_frames(src_data)
			frame_to_save, face, mask = next(reader)

		# face, coords = face_detect([frame_to_save], margin)[0]
		coords = bbx[i % len(mels)]

		face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)
		masks_batch.append(mask)

		if len(img_batch) >= WAV2LIP_BATCH_SIZE:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, IMG_SIZE//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch, masks_batch
			img_batch, mel_batch, frame_batch, coords_batch, masks_batch = [], [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, IMG_SIZE//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch, masks_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

# def load_model(path):
# 	model = Wav2Lip()
# 	print("Load checkpoint from: {}".format(path))
# 	checkpoint = _load(path)
# 	s = checkpoint["state_dict"]
# 	new_s = {}
# 	for k, v in s.items():
# 		new_s[k.replace('module.', '')] = v
# 	model.load_state_dict(new_s)

# 	model = model.to(device)
# 	return model.eval()

def read_frames(src_data):
	if os.path.isdir(src_data["FACE_orig"]):  # FACE가 폴더 경로인지 확인
		image_files = [os.path.join(src_data["FACE_orig"], f) for f in os.listdir(src_data["FACE_orig"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
		image_files_crop = [os.path.join(src_data["FACE_crop"], f) for f in os.listdir(src_data["FACE_crop"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
		image_files_mask = [os.path.join(src_data["FACE_mask"], f) for f in os.listdir(src_data["FACE_mask"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
		
		for image_file, image_file_crop, image_file_mask in zip(image_files, image_files_crop, image_files_mask):
			frame = cv2.imread(image_file)
			frame_crop = cv2.imread(image_file_crop)
			frame_mask = cv2.imread(image_file_mask)
			yield frame, frame_crop, frame_mask
		return  # 폴더 내 이미지를 다 읽었으면 함수 종료

	elif FACE.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:  # 단일 이미지 파일일 때
		face = cv2.imread(FACE)
		while True:
			yield face
		return
    
	# 비디오 파일 처리
	video_stream = cv2.VideoCapture(FACE)
	fps = video_stream.get(cv2.CAP_PROP_FPS)

	print('Reading video frames from start...')

	while True:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		if RESIZE_FACTOR > 1:
			frame = cv2.resize(frame, (frame.shape[1] // RESIZE_FACTOR, frame.shape[0] // RESIZE_FACTOR))
		if ROTATE:
			frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

		y1, y2, x1, x2 = CROP
		if x2 == -1: x2 = frame.shape[1]
		if y2 == -1: y2 = frame.shape[0]

		frame = frame[y1:y2, x1:x2]
		yield frame


def main(engine, seg_net, src_data, wav_path, save_path, device):
	if not os.path.isfile(src_data["FACE_crop"]):
		fps = FPS
		# raise ValueError('--face argument must be a valid path to video/image file')

	elif src_data["FACE_crop"].split('.')[1] in ['jpg', 'png', 'jpeg']:
		fps = FPS
	else:
		video_stream = cv2.VideoCapture(src_data["FACE_crop"])
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		video_stream.release()


	if not wav_path.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(wav_path, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		wav_path = 'temp/temp.wav'

	wav = audio.load_wav(wav_path, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	batch_size = WAV2LIP_BATCH_SIZE
	gen = datagen(mel_chunks, src_data, MARGIN)

	print('load data is done!')
	abs_idx = 0
	for i, (img_batch, mel_batch, frames, coords, masks) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			# print("Loading segmentation network...")
			# seg_net = init_parser(args.segmentation_path)

			# print("Loading super resolution model...")
			# sr_net = init_sr_model(args.sr_path)

			# model = load_model(args.checkpoint_path) 
			# print ("Model loaded")

			frame_h, frame_w = next(read_frames(src_data))[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		# img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		# mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		# with torch.no_grad():
		# 	pred = model(mel_batch, img_batch)

		# pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		# NumPy 배열로 변환
		mel_input = mel_batch.transpose(0, 3, 1, 2).astype(np.float32)
		img_input = img_batch.transpose(0, 3, 1, 2).astype(np.float32)


		# cv2.imwrite('temp.png', (img_input.transpose(0, 2, 3, 1) * 255.)[0].astype(np.uint8)[:,:,:3])
		# exit()

		# TensorRT 엔진으로 추론
		pred = infer_with_trt(engine, mel_input, img_input)
		
		pred = pred.transpose(0, 2, 3, 1) * 255.

		# cv2.imwrite('temp.png', pred[0].astype(np.uint8))
		# exit()

		
		for p, f, c, m in zip(pred, frames, coords, masks):

			x1, y1, x2, y2 = c

			if not NO_SR:
				p = enhance(sr_net, p)

			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))			
			if not NO_SEGMENTATION:
				# p = swap_regions(f[y1:y2, x1:x2], p, seg_net)
				p = swap_regions_fastapi(f[y1:y2, x1:x2], p, m)
			
			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(wav_path, 'temp/result.avi', save_path)
	subprocess.call(command, shell=platform.system() != 'Windows')

	# if SAVE_FRAMES and SAVE_AS_VIDEO:
	# 	gt_out.release()
	# 	pred_out.release()

	# 	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(wav_path, 'temp/gt.avi', GT_PATH)
	# 	subprocess.call(command, shell=platform.system() != 'Windows')

	# 	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(wav_path, 'temp/pred.avi', PRED_PATH)
	# 	subprocess.call(command, shell=platform.system() != 'Windows')



from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI 앱 생성
app = FastAPI()

# 전역 변수로 모델 및 Face Detector
engines = {}  # 두 개의 모델을 저장할 딕셔너리
seg_net = None
src_datas = {
	"HTR": {
		"FACE_crop" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data_crop/source_video_boomerang",
		"FACE_orig" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data_orig/source_video_boomerang",
		"FACE_mask" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data_mask/source_video_boomerang",
		"BBX" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data/source_video_boomerang.npy"
	},
	"KRW": {
		"FACE_crop" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data2_crop/02047",
		"FACE_orig" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data2_orig/02047",
		"FACE_mask" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data2_mask/02047",
		"BBX" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data2/02047.npy"
	},
	"HTR_full": {
		"FACE_crop" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data3_crop/0430",
		"FACE_orig" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data3_orig/0430",
		"FACE_mask" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data3_mask/0430",
		"BBX" : "/share/jacob/wav2lip_live/preprocess_source_data/source_data3/0430.npy"
	}
	}
class SynthesizeRequest(BaseModel):
    character: str  # 'HTR' 또는 'KRW'
    wav_path: str
    save_path: str

@app.on_event("startup")
async def load_model():
    """
    서버가 시작될 때 두 개의 모델 (HTR, KRW)을 GPU에 올림.
    """
    global engines, seg_net

    # 모델 경로 설정
    engine_paths = {
        "HTR": "wav2lip_HTR_b1024.trt",
        "KRW": "wav2lip_KRW_b1024.trt",
		"HTR_full": "wav2lip_HTR_b1024.trt",
    }

    segmentation_path = "./pretrained_ckpt/face_segmentation.pth"

    # Face segmentation 모델 로드
    seg_net = init_parser(segmentation_path)
    print("모델과 seg_net가 성공적으로 로드되었습니다.")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # 두 개의 TensorRT 엔진 로드
    for character, path in engine_paths.items():
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engines[character] = runtime.deserialize_cuda_engine(f.read())
            print(f"TensorRT 엔진 {character} 성공적으로 로드됨.")

@app.post("/synthesize/")
async def synthesize(request: SynthesizeRequest):
	"""
	주어진 character, wav_path를 사용하여 영상을 합성하고 결과를 반환.
	"""
	global engines, seg_net

	# 요청된 인물의 엔진 가져오기
	if request.character not in engines:
		return {"error": f"잘못된 character 선택: {request.character}. 'HTR' 또는 'KRW' 중 선택하세요."}

	engine = engines[request.character]
	src_data = src_datas[request.character]

	if seg_net is None:
		return {"error": "seg_net가 로드되지 않았습니다."}
	if engine is None:
		return {"error": f"{request.character}의 TensorRT 엔진이 로드되지 않았습니다."}


	wav_path = request.wav_path
	save_path = request.save_path

	# 합성 실행
	with torch.no_grad():
		main(engine, seg_net, src_data, wav_path, save_path, device)

	return {"message": "합성 완료", "output_path": save_path}


'''
source /database/venv/talklip/bin/activate

curl -X POST "http://localhost:8001/synthesize/" -H "Content-Type: application/json" -d '{"wav_path": "./sample_voice2_16khz_5s.wav", "save_path": "./250123_test1.mp4"}'
curl -X POST "http://localhost:8002/synthesize/" -H "Content-Type: application/json" -d '{"wav_path": "./sample_voice2_16khz_5s.wav", "save_path": "./250123_test1.mp4", "character": "HTR"}'
curl -X POST "http://localhost:8002/synthesize/" -H "Content-Type: application/json" -d '{"wav_path": "./sample.wav", "save_path": "./250123_test1.mp4", "character": "KRW"}'
curl -X POST "http://localhost:8001/synthesize/" -H "Content-Type: application/json" -d '{"wav_path": "./sample_2.wav", "save_path": "./250414_test.mp4", "character": "HTR"}'

curl -X POST "http://localhost:8001/synthesize/" -H "Content-Type: application/json" -d '{"wav_path": "./001_aud.mp3", "save_path": "./250430_test_v1.mp4", "character": "HTR"}'

CUDA_VISIBLE_DEVICES=7 uvicorn inference_hq_fastapi:app --host 0.0.0.0 --port 8001 --reload
CUDA_VISIBLE_DEVICES=7 uvicorn inference_hq_fastapi_v2:app --host 0.0.0.0 --port 8001 --reload
CUDA_VISIBLE_DEVICES=5 uvicorn inference_hq_fastapi_v3:app --host 0.0.0.0 --port 8001 --reload
CUDA_VISIBLE_DEVICES=3 uvicorn inference_hq_fastapi_v3:app --host 0.0.0.0 --port 8001 --reload

포트 사용확인
sudo netstat -tuln | grep 8001

백그라운드 재생중인 uvicorn 확인
ps aux | grep uvicorn
'''

