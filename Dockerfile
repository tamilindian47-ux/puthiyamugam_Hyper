# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


# System dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    git \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*


# Main environment: RunPod handler and InsightFace scanner.
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        numpy==1.26.4 \
        cython && \
    python3 -m pip install --no-cache-dir \
        -r /app/requirements.txt


# Separate FaceFusion environment.
ARG FACEFUSION_REF=3.3.0

RUN git clone --depth 1 --branch "${FACEFUSION_REF}" \
    https://github.com/facefusion/facefusion.git \
    /opt/facefusion && \
    git -C /opt/facefusion rev-parse HEAD \
        > /opt/facefusion/BUILD_COMMIT

RUN python3 -m venv /opt/ffenv && \
    /opt/ffenv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /opt/ffenv/bin/python -m pip install --no-cache-dir \
        -r /opt/facefusion/requirements.txt && \
    /opt/ffenv/bin/python -m pip uninstall -y \
        onnxruntime \
        onnxruntime-gpu && \
    /opt/ffenv/bin/python -m pip install --no-cache-dir \
        numpy==1.26.4 \
        onnxruntime-gpu==1.22.0 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        'nvidia-cudnn-cu12==9.*'


# Register NVIDIA libraries from both Python installations with
# the system dynamic linker. FaceFusion paths are listed first.
RUN python3 - <<'PY'
import json
import subprocess
from pathlib import Path

discovery_code = """
import json
import site
import sysconfig

roots = set(site.getsitepackages())
roots.add(sysconfig.get_paths()["purelib"])
print(json.dumps(sorted(roots)))
"""

interpreters = [
    "/opt/ffenv/bin/python",
    "/usr/bin/python3",
]

library_dirs = []
cudnn_files = []

for interpreter in interpreters:
    output = subprocess.check_output(
        [interpreter, "-c", discovery_code],
        text=True,
    )

    roots = json.loads(output)

    for root in roots:
        nvidia_root = Path(root) / "nvidia"

        if not nvidia_root.is_dir():
            continue

        for library_dir in sorted(nvidia_root.glob("*/lib")):
            directory = str(library_dir.resolve())

            if directory not in library_dirs:
                library_dirs.append(directory)

        cudnn_files.extend(
            nvidia_root.glob("cudnn/lib/libcudnn.so.9")
        )

if not library_dirs:
    raise SystemExit("No pip-installed NVIDIA library directories found.")

if not cudnn_files:
    raise SystemExit("libcudnn.so.9 was not installed.")

config = Path("/etc/ld.so.conf.d/00-puthiyamugam-nvidia.conf")
config.write_text("\n".join(library_dirs) + "\n")

print("Registered NVIDIA library directories:")
for directory in library_dirs:
    print(directory)

print("Found cuDNN:")
for library in cudnn_files:
    print(library)
PY

RUN ldconfig


# Preload from NVIDIA pip packages when FaceFusion Python starts.
# The linker configuration above also makes lazy-loaded dependencies
# discoverable by their shared-library names.
RUN /opt/ffenv/bin/python - <<'PY'
import sysconfig
from pathlib import Path

destination = (
    Path(sysconfig.get_paths()["purelib"])
    / "sitecustomize.py"
)

destination.write_text(
    "import onnxruntime as ort\n"
    "ort.preload_dlls(directory='')\n"
)

print("Created:", destination)
PY


# Build-time library checks.
# These verify library loading, not GPU inference: Docker image
# builders normally do not have an NVIDIA GPU attached.
RUN python3 - <<'PY'
import ctypes
import onnxruntime as ort

ort.preload_dlls(directory="")

cudnn = ctypes.CDLL("libcudnn.so.9")
cudnn.cudnnGetVersion.restype = ctypes.c_size_t

version = int(cudnn.cudnnGetVersion())

if version < 90000:
    raise SystemExit(f"Expected cuDNN 9+, found version {version}")

print("Scanner ONNX Runtime:", ort.__version__)
print("Scanner cuDNN loaded:", version)
PY

RUN /opt/ffenv/bin/python - <<'PY'
import ctypes
import onnxruntime as ort

cudnn = ctypes.CDLL("libcudnn.so.9")
cudnn.cudnnGetVersion.restype = ctypes.c_size_t

version = int(cudnn.cudnnGetVersion())

if version < 90000:
    raise SystemExit(f"Expected cuDNN 9+, found version {version}")

print("FaceFusion ONNX Runtime:", ort.__version__)
print("FaceFusion cuDNN loaded:", version)
PY


# Check the pinned FaceFusion CLI before deployment.
RUN cd /opt/facefusion && \
    /opt/ffenv/bin/python facefusion.py headless-run --help \
        > /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1a_256' /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1b_256' /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1c_256' /tmp/ff-help.txt && \
    grep -q -- '--reference-frame-number' /tmp/ff-help.txt && \
    grep -q -- '--expression-restorer-factor' /tmp/ff-help.txt && \
    grep -q -- '--face-mask-types' /tmp/ff-help.txt


# Bake InsightFace scanner models into the image.
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    curl -fL --retry 5 \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
        -o /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip \
        -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip


# Optional network-volume asset directory.
ENV FF_ASSETS_DIR="" \
    JOB_TIMEOUT_SECONDS=1800 \
    EXPRESSION_FACTOR=60 \
    REFERENCE_FACE_DISTANCE=0.30 \
    ANCHOR_MATCH_THRESHOLD=0.50

COPY scanner.py handler.py /app/

CMD ["python3", "-u", "/app/handler.py"]
