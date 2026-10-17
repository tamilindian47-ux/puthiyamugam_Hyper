# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev build-essential \
        ffmpeg git curl unzip libgl1 libglib2.0-0 ca-certificates zlib1g && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

ARG FACEFUSION_REF=3.3.0

RUN git clone --depth 1 --branch "${FACEFUSION_REF}" \
        https://github.com/facefusion/facefusion.git /opt/facefusion && \
    git -C /opt/facefusion rev-parse HEAD > /opt/facefusion/BUILD_COMMIT

RUN python3 -m venv /opt/ffenv

RUN python3 - <<'PY'
import re
from pathlib import Path
source = Path("/opt/facefusion/requirements.txt").read_text(encoding="utf-8")
excluded = {"numpy", "onnxruntime", "onnxruntime-gpu"}
kept = []
for line in source.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        kept.append(line)
        continue
    name = re.split(r"[<>=!~;\[\s]", stripped, maxsplit=1)[0].lower().replace("_", "-")
    if name not in excluded:
        kept.append(line)
Path("/tmp/facefusion-requirements.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/ffenv/bin/python -m pip install --upgrade pip && \
    /opt/ffenv/bin/python -m pip install \
        -r /tmp/facefusion-requirements.txt \
        numpy==1.26.4 onnxruntime-gpu==1.22.0 'requests>=2.32,<3' \
        nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 \
        nvidia-cufft-cu12 nvidia-curand-cu12 'nvidia-cudnn-cu12==9.*'

# GPU-telemetry parser fix only. No content-analyser edits are needed because the
# bridge never calls the content analyser.
RUN /opt/ffenv/bin/python - <<'PY'
import ast
from pathlib import Path
path = Path("/opt/facefusion/facefusion/execution.py")
tree = ast.parse(path.read_text(encoding="utf-8"))
matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_value_and_unit"]
if len(matches) != 1 or len(matches[0].args.args) != 1:
    raise RuntimeError("Unsupported telemetry parser; refusing to guess.")
node = matches[0]
name = node.args.args[0].arg
body = (
    "import re\n"
    "if not isinstance(VALUE, str):\n    return None\n"
    "match = re.fullmatch(r'\\s*([0-9]+)\\s+(\\S+)\\s*', VALUE)\n"
    "if match is None:\n    return None\n"
    "return {'value': int(match.group(1)), 'unit': match.group(2)}\n"
).replace("VALUE", name)
node.body = ast.parse(body).body
ast.fix_missing_locations(tree)
patched = ast.unparse(tree) + "\n"
compile(patched, str(path), "exec")
path.write_text(patched, encoding="utf-8")
print("BUILD CHECK: telemetry parser patched", flush=True)
PY

RUN python3 - <<'PY'
import json, subprocess
from pathlib import Path
code = ("import json, site, sysconfig; roots = set(site.getsitepackages()); "
        "roots.add(sysconfig.get_paths()['purelib']); print(json.dumps(sorted(roots)))")
directories = []
for interpreter in ("/opt/ffenv/bin/python", "/usr/bin/python3"):
    for root in json.loads(subprocess.check_output([interpreter, "-c", code], text=True)):
        for directory in sorted((Path(root) / "nvidia").glob("*/lib")):
            value = str(directory.resolve())
            if value not in directories:
                directories.append(value)
if not directories:
    raise RuntimeError("No NVIDIA library directories found.")
Path("/etc/ld.so.conf.d/00-puthiyamugam-nvidia.conf").write_text("\n".join(directories) + "\n")
PY

RUN ldconfig

RUN /opt/ffenv/bin/python - <<'PY'
import sysconfig
from pathlib import Path
destination = Path(sysconfig.get_paths()["purelib"]) / "sitecustomize.py"
destination.write_text("import onnxruntime as ort\nort.preload_dlls(directory='')\n", encoding="utf-8")
PY

RUN /opt/ffenv/bin/python -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('BUILD CHECK: cuDNN 9 OK')"

# Everything below changes whenever your source changes, so dependency layers stay cached.
ARG SOURCE_SHA=unknown
ENV SOURCE_SHA=${SOURCE_SHA}
LABEL org.opencontainers.image.revision=${SOURCE_SHA}

COPY handler.py scanner.py ff_bridge.py /app/

# Refuse to build an image whose three source files belong to different bundles.
RUN python3 - <<'PY'
import re
from pathlib import Path
expected = "pm-bundle-20261017-1"
for name in ("handler.py", "scanner.py", "ff_bridge.py"):
    text = Path("/app", name).read_text(encoding="utf-8")
    compile(text, name, "exec")
    found = re.search(r"^BUNDLE_ID\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if not found or found.group(1) != expected:
        raise RuntimeError(f"{name} is not from bundle {expected}: {found.group(1) if found else 'missing marker'}")
bridge = Path("/app/ff_bridge.py").read_text(encoding="utf-8")
if "def checked_content" in bridge:
    raise RuntimeError("ff_bridge.py still contains the old content-analysis function.")
print("BUILD CHECK: all source files belong to", expected, flush=True)
PY

RUN python3 -c "import numpy, PIL, requests, runpod; print('BUILD CHECK: main imports OK')"

RUN cd /opt/facefusion && PYTHONPATH=/app:/opt/facefusion /opt/ffenv/bin/python - <<'PY'
import inspect
import cv2, numpy, requests, onnxruntime, scanner
from facefusion import core, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer
from facefusion.args import apply_args
from facefusion.face_analyser import get_many_faces
from facefusion.processors import choices
from facefusion.processors.modules import face_swapper, expression_restorer
from facefusion.program import create_program
for module in (face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, face_swapper, expression_restorer):
    if not callable(getattr(module, "pre_check", None)):
        raise RuntimeError("Missing pre_check in " + module.__name__)
required = {"source_face", "target_face", "temp_vision_frame"}
parameters = set(inspect.signature(face_swapper.swap_face).parameters)
if not required.issubset(parameters) or parameters - required - {"source_vision_frame"}:
    raise RuntimeError("Unsupported swap_face signature: " + str(parameters))
expected = {"source_vision_frame", "target_face", "temp_vision_frame"}
if set(inspect.signature(expression_restorer.restore_expression).parameters) != expected:
    raise RuntimeError("Unsupported expression-restorer signature.")
models = {"hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256"}
if not models.issubset(set(choices.face_swapper_models)):
    raise RuntimeError("Required HyperSwap models are missing.")
print("BUILD CHECK: FaceFusion adapter APIs and HyperSwap models OK", flush=True)
PY

ENV FF_ASSETS_DIR="" \
    JOB_TIMEOUT_SECONDS=3600 \
    MAX_INPUT_BYTES=536870912 \
    MAX_DURATION_SECONDS=600 \
    MAX_VIDEO_PIXELS=8294400 \
    SCAN_FRAMES=60 \
    SCAN_IDENTITY_SIMILARITY=0.58 \
    MATCH_SIMILARITY=0.50 \
    MATCH_MARGIN=0.08 \
    GIF_FPS=25

CMD ["python3", "-u", "/app/handler.py"]
