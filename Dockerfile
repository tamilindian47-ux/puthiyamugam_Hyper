# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
        zlib1g && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Main environment: Runpod handler and media utilities
# ------------------------------------------------------------

COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        -r /app/requirements.txt

# ------------------------------------------------------------
# FaceFusion: pinned version in a separate virtual environment
# ------------------------------------------------------------

ARG FACEFUSION_REF=3.3.0

RUN git clone \
        --depth 1 \
        --branch "${FACEFUSION_REF}" \
        https://github.com/facefusion/facefusion.git \
        /opt/facefusion && \
    git -C /opt/facefusion rev-parse HEAD \
        > /opt/facefusion/BUILD_COMMIT

RUN python3 -m venv /opt/ffenv

# Prepare FaceFusion requirements without first installing
# a CPU ONNX Runtime or a NumPy version we immediately replace.

RUN python3 - <<'PY'
import re
from pathlib import Path

source = Path(
    "/opt/facefusion/requirements.txt"
).read_text(encoding="utf-8")

excluded = {
    "numpy",
    "onnxruntime",
    "onnxruntime-gpu",
}

kept = []

for line in source.splitlines():
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        kept.append(line)
        continue

    name = re.split(
        r"[<>=!~;\[\s]",
        stripped,
        maxsplit=1,
    )[0].lower().replace("_", "-")

    if name not in excluded:
        kept.append(line)

Path("/tmp/facefusion-requirements.txt").write_text(
    "\n".join(kept) + "\n",
    encoding="utf-8",
)
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/ffenv/bin/python -m pip install --upgrade pip && \
    /opt/ffenv/bin/python -m pip install \
        -r /tmp/facefusion-requirements.txt \
        numpy==1.26.4 \
        onnxruntime-gpu==1.22.0 \
        'requests>=2.32,<3' \
        nvidia-cuda-runtime-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        'nvidia-cudnn-cu12==9.*'

# ------------------------------------------------------------
# GPU telemetry fix only.
#
# Keep native content analysis and model pre-checks intact.
# Do not rewrite FaceFusion source from handler.py at runtime.
# ------------------------------------------------------------

RUN /opt/ffenv/bin/python - <<'PY'
import ast
from pathlib import Path

path = Path(
    "/opt/facefusion/facefusion/execution.py"
)

tree = ast.parse(
    path.read_text(encoding="utf-8")
)

matches = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "create_value_and_unit"
]

if len(matches) != 1:
    raise RuntimeError(
        "Expected one create_value_and_unit function."
    )

node = matches[0]

if len(node.args.args) != 1:
    raise RuntimeError(
        "Unexpected GPU telemetry parser signature."
    )

argument = node.args.args[0].arg

replacement = (
    "import re\n"
    "if not isinstance(VALUE, str):\n"
    "    return None\n"
    "match = re.fullmatch("
    "r'\\s*([0-9]+)\\s+(\\S+)\\s*', VALUE)\n"
    "if match is None:\n"
    "    return None\n"
    "return {\n"
    "    'value': int(match.group(1)),\n"
    "    'unit': match.group(2)\n"
    "}\n"
).replace("VALUE", argument)

node.body = ast.parse(replacement).body

ast.fix_missing_locations(tree)

patched = ast.unparse(tree) + "\n"

compile(
    patched,
    str(path),
    "exec",
)

path.write_text(
    patched,
    encoding="utf-8",
)

print(
    "BUILD CHECK: GPU telemetry parser patched. "
    "Native model/content pre-checks unchanged.",
    flush=True,
)
PY

# ------------------------------------------------------------
# Register pip-installed NVIDIA libraries with the Linux linker
# ------------------------------------------------------------

RUN python3 - <<'PY'
import json
import subprocess
from pathlib import Path

discovery = (
    "import json, site, sysconfig; "
    "roots = set(site.getsitepackages()); "
    "roots.add(sysconfig.get_paths()['purelib']); "
    "print(json.dumps(sorted(roots)))"
)

directories = []

for interpreter in (
    "/opt/ffenv/bin/python",
    "/usr/bin/python3",
):
    output = subprocess.check_output(
        [
            interpreter,
            "-c",
            discovery,
        ],
        text=True,
    )

    for root in json.loads(output):
        nvidia = Path(root) / "nvidia"

        for directory in sorted(
            nvidia.glob("*/lib")
        ):
            resolved = str(directory.resolve())

            if resolved not in directories:
                directories.append(resolved)

if not directories:
    raise RuntimeError(
        "No NVIDIA library directories were found."
    )

Path(
    "/etc/ld.so.conf.d/00-puthiyamugam-nvidia.conf"
).write_text(
    "\n".join(directories) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(directories),
    flush=True,
)
PY

RUN ldconfig

# ------------------------------------------------------------
# Preload CUDA/cuDNN libraries in the FaceFusion environment
# ------------------------------------------------------------

RUN /opt/ffenv/bin/python - <<'PY'
import sysconfig
from pathlib import Path

destination = Path(
    sysconfig.get_paths()["purelib"]
) / "sitecustomize.py"

destination.write_text(
    "import onnxruntime as ort\n"
    "ort.preload_dlls(directory='')\n",
    encoding="utf-8",
)
PY

RUN /opt/ffenv/bin/python -c \
    "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('BUILD CHECK: cuDNN 9 load OK')"

# ------------------------------------------------------------
# Copy the new multi-person application
# ------------------------------------------------------------

COPY handler.py scanner.py ff_bridge.py /app/

# ------------------------------------------------------------
# Main environment imports and Python syntax checks
# ------------------------------------------------------------

RUN python3 - <<'PY'
from pathlib import Path

import numpy
import PIL
import requests
import runpod

for name in (
    "handler.py",
    "scanner.py",
    "ff_bridge.py",
):
    source = Path(
        "/app",
        name,
    ).read_text(encoding="utf-8")

    compile(
        source,
        name,
        "exec",
    )

print(
    "BUILD CHECK: application imports and syntax OK.",
    flush=True,
)
PY

# ------------------------------------------------------------
# Verify the actual FaceFusion API used by ff_bridge.py
# ------------------------------------------------------------

RUN cd /opt/facefusion && \
    PYTHONPATH=/app:/opt/facefusion \
    /opt/ffenv/bin/python - <<'PY'
import inspect

import cv2
import numpy
import requests
import onnxruntime
import scanner

from facefusion import core
from facefusion.args import apply_args
from facefusion.face_analyser import get_many_faces
from facefusion.processors import choices
from facefusion.processors.modules import (
    face_swapper,
    expression_restorer,
)
from facefusion.program import create_program

required = {
    "source_face",
    "target_face",
    "temp_vision_frame",
}

parameters = set(
    inspect.signature(
        face_swapper.swap_face
    ).parameters
)

unexpected = (
    parameters
    - required
    - {"source_vision_frame"}
)

if (
    not required.issubset(parameters)
    or unexpected
):
    raise RuntimeError(
        "Unsupported swap_face signature: "
        + str(parameters)
    )

expected_expression = {
    "source_vision_frame",
    "target_face",
    "temp_vision_frame",
}

expression_parameters = set(
    inspect.signature(
        expression_restorer.restore_expression
    ).parameters
)

if expression_parameters != expected_expression:
    raise RuntimeError(
        "Unsupported expression-restorer signature: "
        + str(expression_parameters)
    )

required_models = {
    "hyperswap_1a_256",
    "hyperswap_1b_256",
    "hyperswap_1c_256",
}

available_models = set(
    choices.face_swapper_models
)

if not required_models.issubset(
    available_models
):
    raise RuntimeError(
        "Required HyperSwap models are missing."
    )

print(
    "BUILD CHECK: native multi-person adapter APIs "
    "and all three HyperSwap models supported.",
    flush=True,
)
PY

# ------------------------------------------------------------
# Runtime configuration
#
# Attach the SAME Runpod network volume to every worker,
# mounted at /runpod-volume.
# ------------------------------------------------------------

ENV FF_ASSETS_DIR="" \
    RESULTS_DIR=/runpod-volume/pm-results \
    RESULT_TTL_SECONDS=86400 \
    JOB_TIMEOUT_SECONDS=3600 \
    MAX_INPUT_BYTES=536870912 \
    MAX_DURATION_SECONDS=600 \
    MAX_VIDEO_PIXELS=8294400 \
    SCAN_FRAMES=60 \
    SCAN_IDENTITY_SIMILARITY=0.58 \
    MATCH_SIMILARITY=0.50 \
    MATCH_MARGIN=0.08 \
    OUTPUT_CRF=18 \
    GIF_FPS=25

CMD ["python3", "-u", "/app/handler.py"]
