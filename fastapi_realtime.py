import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import tempfile, os, torch
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing

# ---- 1) 가짜 args 생성 ----
class Args:
    pass

args = Args()
args.version = "v15"
args.extra_margin = 10
args.parsing_mode = "jaw"
args.audio_padding_length_left = 2
args.audio_padding_length_right = 2
args.skip_save_images = False

# ---- 2) Avatar 및 모듈 로드 ----
import scripts.realtime_inference as realtime_inference
realtime_inference.args = args  # args 전역 등록
from scripts.realtime_inference import Avatar

# ---- 3) 모델 로드 ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
weight_dtype = unet.model.dtype

whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

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

# ---- 5) FastAPI 서버 ----
app = FastAPI()

@app.post("/realtime-infer")
async def realtime_infer(
    avatar_id: str = Form(...),
    video_path: str = Form(...),
    audio_file: UploadFile = None,
):
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, audio_file.filename)
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    avatar = Avatar(
        avatar_id=avatar_id,
        video_path=video_path,
        bbox_shift=0,
        batch_size=20,
        preparation=True,
    )

    avatar.inference(
        audio_path=audio_path,
        out_vid_name="result",
        fps=25,
        skip_save_images=False,
    )

    output_file = os.path.join(avatar.video_out_path, "result.mp4")
    return FileResponse(output_file, media_type="video/mp4", filename="result.mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
