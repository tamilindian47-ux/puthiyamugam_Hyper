# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies
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

# Main environment: RunPod handler and InsightFace scanner
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        numpy==1.26.4 \
        cython && \
    python3 -m pip install --no-cache-dir \
        -r /app/requirements.txt

# FaceFusion setup
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

# BUILD-TIME PATCH: Fix GPU memory telemetry + Disable NSFW Filter + Bypass Core Hash Check
RUN /opt/ffenv/bin/python - <<'PY'
import ast
from pathlib import Path

ff_pkg = Path("/opt/facefusion/facefusion")

# 1. Patch GPU telemetry
exec_path = ff_pkg / "execution.py"
src = exec_path.read_text(encoding="utf-8")
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "create_value_and_unit":
        node.body = [
            ast.parse(
                "import re\n"
                "if not isinstance(text, str): return None\n"
                "m = re.fullmatch(r'\\s*([0-9]+)\\s+(\\S+)\\s*', text)\n"
                "if not m: return None\n"
                "return {'value': int(m.group(1)), 'unit': m.group(2)}"
            )
        ]
exec_path.write_text(ast.unparse(tree), encoding="utf-8")

# 2. Patch Content Analyser (AST rewrites all checks to return False)
analyser_path = ff_pkg / "content_analyser.py"
a_src = analyser_path.read_text(encoding="utf-8")
a_tree = ast.parse(a_src)
for node in a_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "analyse_frame", "analyse_image", "analyse_video", "detect_nsfw", "analyse_stream"
    ):
        node.body = [ast.Return(value=ast.Constant(value=False))]
analyser_path.write_text(ast.unparse(a_tree), encoding="utf-8")

# 3. Patch Core (common_pre_check returns True to bypass hash validation)
core_path = ff_pkg / "core.py"
c_src = core_path.read_text(encoding="utf-8")
c_tree = ast.parse(c_src)
for node in c_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "common_pre_check":
        node.body = [ast.Return(value=ast.Constant(value=True))]
core_path.write_text(ast.unparse(c_tree), encoding="utf-8")

print("SUCCESS: Build-time FaceFusion patches applied cleanly via AST!")
PY

# Register dynamic linker paths for cuDNN 9
RUN python3 - <<'PY'
import json
import subprocess
from pathlib import Path

discovery_code = """
import json, site, sysconfig
roots = set(site.getsitepackages())
roots.add(sysconfig.get_paths()["purelib"])
print(json.dumps(sorted(roots)))
"""

library_dirs = []
for interpreter in ["/opt/ffenv/bin/python", "/usr/bin/python3"]:
    out = subprocess.check_output([interpreter, "-c", discovery_code], text=True)
    for root in json.loads(out):
        n_root = Path(root) / "nvidia"
        if n_root.is_dir():
            for l_dir in sorted(n_root.glob("*/lib")):
                d = str(l_dir.resolve())
                if d not in library_dirs:
                    library_dirs.append(d)

Path("/etc/ld.so.conf.d/00-puthiyamugam-nvidia.conf").write_text("\n".join(library_dirs) + "\n")
PY

RUN ldconfig

# Preload DLLs
RUN /opt/ffenv/bin/python - <<'PY'
import sysconfig
from pathlib import Path
dest = Path(sysconfig.get_paths()["purelib"]) / "sitecustomize.py"
dest.write_text("import onnxruntime as ort\nort.preload_dlls(directory='')\n")
PY

# Bake InsightFace scanner models
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    curl -fL --retry 5 \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
        -o /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip \
        -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip

ENV FF_ASSETS_DIR="" \
    JOB_TIMEOUT_SECONDS=1800 \
    EXPRESSION_FACTOR=60 \
    REFERENCE_FACE_DISTANCE=0.30 \
    ANCHOR_MATCH_THRESHOLD=0.50

# Invalidate cache for handler and scanner
ARG CACHEBUST=1
COPY scanner.py handler.py /app/

CMD ["python3", "-u", "/app/handler.py"]
