# syntax=docker/dockerfile:1

# Reuse YOUR previously successful image instead of downloading
# and reinstalling all Python, CUDA, and FaceFusion dependencies.
#
# This exact digest comes from the working-image logs you supplied.
FROM registry.runpod.net/tamilindian47-ux-puthiyamugam-hyper-main-dockerfile@sha256:e695d761b713d4a84ce3c4822b09afd6f0fa6e759a99caf7335eaa08e92aa6a8

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------
# Restore native FaceFusion model/content pre-checks.
#
# The previous image contains modifications from the old build.
# Restore these two files from that image's own pinned checkout.
# Keep its existing CUDA environment and GPU telemetry fix.
# ------------------------------------------------------------

RUN /opt/ffenv/bin/python - <<'PY'
import ast
import subprocess
from pathlib import Path

root = Path("/opt/facefusion")

commit = subprocess.check_output(
    [
        "git",
        "-C",
        str(root),
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

expected = "26c701cd62068cbcee871fcf55c3689e197f5098"

if commit != expected:
    raise RuntimeError(
        "Unexpected FaceFusion commit in base image: "
        + commit
    )

pending = {}

for relative in (
    "facefusion/core.py",
    "facefusion/content_analyser.py",
):
    original = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            "HEAD:" + relative,
        ],
        text=True,
    )

    compile(original, relative, "exec")
    pending[root / relative] = original

for path, source in pending.items():
    path.write_text(source, encoding="utf-8")

cache = root / "facefusion" / "__pycache__"

if cache.is_dir():
    for path in cache.glob("*.pyc"):
        path.unlink()

print(
    "Restored native content/model pre-checks. "
    "Kept the existing CUDA environment and telemetry fix.",
    flush=True,
)
PY

# ------------------------------------------------------------
# Copy the NEW multi-person application.
#
# requirements.txt is retained as a record of app requirements.
# Dependencies are inherited from the successful base image;
# this Dockerfile deliberately does not reinstall them.
# ------------------------------------------------------------

COPY requirements.txt /app/requirements.txt

COPY handler.py scanner.py ff_bridge.py /app/

# ------------------------------------------------------------
# Verify the main environment and application syntax.
#
# These checks do not download packages or models.
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
    source = Path("/app", name).read_text(
        encoding="utf-8"
    )
    compile(source, name, "exec")

print(
    "BUILD CHECK: main environment imports "
    "and application syntax OK.",
    flush=True,
)
PY

# ------------------------------------------------------------
# Check the native FaceFusion APIs used by ff_bridge.py.
#
# This fails early if the inherited installation is incompatible.
# No GPU inference or model downloading is performed here.
# ------------------------------------------------------------

RUN cd /opt/facefusion && \
    PYTHONPATH=/app:/opt/facefusion \
    /opt/ffenv/bin/python - <<'PY'
import ast
import ctypes
import inspect
from pathlib import Path

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

# Confirm the existing cuDNN linker configuration works.
ctypes.CDLL("libcudnn.so.9")

# Support the native swapper signatures handled by our adapter.
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

# Confirm the expression-restorer interface.
expected = {
    "source_vision_frame",
    "target_face",
    "temp_vision_frame",
}

actual = set(
    inspect.signature(
        expression_restorer.restore_expression
    ).parameters
)

if actual != expected:
    raise RuntimeError(
        "Unsupported expression-restorer API: "
        + str(actual)
    )

# Confirm all requested HyperSwap models are supported.
models = {
    "hyperswap_1a_256",
    "hyperswap_1b_256",
    "hyperswap_1c_256",
}

if not models.issubset(
    set(choices.face_swapper_models)
):
    raise RuntimeError(
        "Required HyperSwap models are not supported."
    )

# Confirm core model pre-check calls remain present.
core_path = Path(
    "/opt/facefusion/facefusion/core.py"
)

tree = ast.parse(
    core_path.read_text(encoding="utf-8")
)

function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "common_pre_check"
)

has_pre_checks = any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "pre_check"
    for node in ast.walk(function)
)

if not has_pre_checks:
    raise RuntimeError(
        "Native core model pre-check calls missing."
    )

print(
    "BUILD CHECK: native adapter APIs, "
    "HyperSwap choices, and cuDNN 9 OK. "
    "No GPU inference was run.",
    flush=True,
)
PY

# ------------------------------------------------------------
# Runtime configuration.
#
# REQUIRED:
# Attach the SAME RunPod network volume to every worker,
# mounted at /runpod-volume.
#
# Input uploads and rendered outputs use that shared volume.
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
