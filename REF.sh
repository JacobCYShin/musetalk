docker run --rm -it --gpus all -p 8000:8000   -v $(pwd):/workspace   -w /workspace --entrypoint /bin/bash  musetalk
