FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential \
    ffmpeg git curl unzip libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy==1.26.4 cython && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Separate environment: FaceFusion's dependencies do not replace the scanner's.
ARG FACEFUSION_REF=3.3.0
RUN git clone --depth 1 --branch "${FACEFUSION_REF}" \
    https://github.com/facefusion/facefusion.git /opt/facefusion && \
    git -C /opt/facefusion rev-parse HEAD > /opt/facefusion/BUILD_COMMIT
RUN python3 -m venv /opt/ffenv && \
    /opt/ffenv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /opt/ffenv/bin/python -m pip install --no-cache-dir \
        -r /opt/facefusion/requirements.txt && \
    /opt/ffenv/bin/python -m pip uninstall -y onnxruntime onnxruntime-gpu && \
    /opt/ffenv/bin/python -m pip install --no-cache-dir \
        onnxruntime-gpu==1.22.0 nvidia-cuda-runtime-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 'nvidia-cudnn-cu12==9.*'
RUN /opt/ffenv/bin/python -c "import pathlib,sysconfig; pathlib.Path(sysconfig.get_paths()['purelib'],'sitecustomize.py').write_text('import onnxruntime as ort\nort.preload_dlls()\n')"

# Fail the build instead of silently running an incompatible CLI release.
RUN cd /opt/facefusion && /opt/ffenv/bin/python facefusion.py headless-run --help \
    > /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1a_256' /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1b_256' /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1c_256' /tmp/ff-help.txt && \
    grep -q -- '--reference-frame-number' /tmp/ff-help.txt && \
    grep -q -- '--expression-restorer-factor' /tmp/ff-help.txt && \
    grep -q -- '--face-mask-types' /tmp/ff-help.txt

# Bake the scanner model into the image; fail if download/unzip fails.
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    curl -fL --retry 5 \
      https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
      -o /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip

# Optional mounted RunPod network volume for FaceFusion's .assets directory.
ENV FF_ASSETS_DIR="" JOB_TIMEOUT_SECONDS=1800 \
    EXPRESSION_FACTOR=60 REFERENCE_FACE_DISTANCE=0.30 \
    ANCHOR_MATCH_THRESHOLD=0.50
COPY scanner.py handler.py /app/
CMD ["python3", "-u", "/app/handler.py"]
