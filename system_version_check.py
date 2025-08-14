import sys
import os
import subprocess
import torch

print("=== Python and Package Versions ===")
print(f"Python version: {sys.version}")

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"ONNX Runtime available providers: {ort.get_available_providers()}")
except ImportError:
    print("ONNX Runtime is not installed.")
    
try:
    import onnx
    print(f"ONNX library version: {onnx.__version__}")
except ImportError:
    print("ONNX library is not installed.")

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA is available in PyTorch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"PyTorch cuDNN version: {torch.backends.cudnn.version()}")
except ImportError:
    print("PyTorch is not installed.")

print("\n=== CUDA & Driver Versions (System) ===")

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"Command not found or error: {e}"

# NVIDIA Driver Version
print(f"NVIDIA Driver version: {run_cmd(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])}")

# CUDA Toolkit Version (if installed)
print(f"CUDA Toolkit version: {run_cmd(['nvcc', '--version'])}")

# Check for cuDNN library files
print("\n=== cuDNN Library Check ===")
cudnn_lib_paths = [
    '/usr/lib/x86_64-linux-gnu/',
    '/usr/local/cuda/lib64/'
]
found_cudnn = False
for path in cudnn_lib_paths:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if 'libcudnn' in f]
        if files:
            print(f"Found cuDNN files in {path}: {', '.join(files)}")
            found_cudnn = True
if not found_cudnn:
    print("No cuDNN library files found in common locations.")

print("\n=== GPU Information ===")
print(f"GPU Name: {run_cmd(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])}")