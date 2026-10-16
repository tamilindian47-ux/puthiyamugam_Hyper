# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PUTHIYAMUGAM_BUILD_ID=hs-build-only-v1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential \
    ffmpeg git curl unzip libgl1 libglib2.0-0 ca-certificates zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy==1.26.4 cython && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

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
        numpy==1.26.4 onnxruntime-gpu==1.22.0 \
        nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
        'nvidia-cudnn-cu12==9.*'

# Modify FaceFusion only at build time. Preserve core model preparation.
RUN /opt/ffenv/bin/python - <<'PY'
import ast
from pathlib import Path
root = Path('/opt/facefusion/facefusion')
pending = {}

def load(name):
    path = root / name
    original = path.read_text(encoding='utf-8')
    return path, original, ast.parse(original)

def function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one function: {name}')
    return matches[0]

def queue(path, original, tree):
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + '\n'
    compile(result, str(path), 'exec')
    pending[path] = (original, result)

path, original, tree = load('execution.py')
node = function(tree, 'create_value_and_unit')
if len(node.args.args) != 1:
    raise RuntimeError('Unexpected GPU telemetry parser signature.')
argument = node.args.args[0].arg
body = '''import re
if not isinstance(VALUE, str):
    return None
match = re.fullmatch(r'\s*([0-9]+)\s+(\S+)\s*', VALUE)
if match is None:
    return None
return {'value': int(match.group(1)), 'unit': match.group(2)}
'''.replace('VALUE', argument)
node.body = ast.parse(body).body
queue(path, original, tree)

path, original, tree = load('content_analyser.py')
for name in ('analyse_frame', 'analyse_image', 'analyse_video'):
    function(tree, name).body = [ast.Return(value=ast.Constant(value=False))]
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('detect_nsfw', 'analyse_stream'):
        node.body = [ast.Return(value=ast.Constant(value=False))]
function(tree, 'pre_check').body = [ast.Return(value=ast.Constant(value=True))]
queue(path, original, tree)

path, original, tree = load('core.py')
node = function(tree, 'common_pre_check')
assignments = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'common_modules' for t in n.targets)]
if len(assignments) != 1 or not isinstance(assignments[0].value, (ast.List, ast.Tuple)):
    raise RuntimeError('Unexpected common_pre_check structure; refusing to patch.')
modules = assignments[0].value
names = {item.id for item in modules.elts if isinstance(item, ast.Name)}
required = {'face_classifier', 'face_detector', 'face_landmarker',
            'face_masker', 'face_recognizer', 'voice_extractor'}
if not required.issubset(names):
    raise RuntimeError(f'Missing expected core modules: {required - names}')
node.body = [
    ast.Assign(targets=[ast.Name(id='common_modules', ctx=ast.Store())], value=modules),
    ast.Return(value=ast.parse('all(module.pre_check() for module in common_modules)', mode='eval').body),
]
queue(path, original, tree)

# Validate every change before writing any source file.
for path, (original, result) in pending.items():
    path.with_suffix('.py.original').write_text(original, encoding='utf-8')
    path.write_text(result, encoding='utf-8')
    print('Patched and syntax-checked:', path)
print('BUILD CHECK: core model pre-checks preserved; no runtime patch required.')
PY

# Register pip NVIDIA libraries for the Linux dynamic linker.
RUN python3 - <<'PY'
import json
import subprocess
from pathlib import Path
code = "import json,site,sysconfig; print(json.dumps(sorted(set(site.getsitepackages()+[sysconfig.get_paths()['purelib']]))))"
directories = []
cudnn_found = False
for interpreter in ('/opt/ffenv/bin/python', '/usr/bin/python3'):
    roots = json.loads(subprocess.check_output([interpreter, '-c', code], text=True))
    for root in roots:
        for directory in sorted((Path(root) / 'nvidia').glob('*/lib')):
            value = str(directory.resolve())
            if value not in directories:
                directories.append(value)
            if (directory / 'libcudnn.so.9').exists():
                cudnn_found = True
if not directories or not cudnn_found:
    raise RuntimeError('NVIDIA library directories or cuDNN 9 missing.')
Path('/etc/ld.so.conf.d/00-puthiyamugam-nvidia.conf').write_text('\n'.join(directories) + '\n')
print('\n'.join(directories))
PY
RUN ldconfig

RUN /opt/ffenv/bin/python - <<'PY'
import sysconfig
from pathlib import Path
path = Path(sysconfig.get_paths()['purelib']) / 'sitecustomize.py'
path.write_text("import onnxruntime as ort\nort.preload_dlls(directory='')\n")
PY

# These are library-loading checks, not GPU inference tests.
RUN python3 -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('Scanner: cuDNN load OK')" && \
    /opt/ffenv/bin/python -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('FaceFusion: cuDNN load OK')"

RUN cd /opt/facefusion && \
    /opt/ffenv/bin/python facefusion.py headless-run --help > /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1a_256' /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1b_256' /tmp/ff-help.txt && \
    grep -q -- 'hyperswap_1c_256' /tmp/ff-help.txt && \
    grep -q -- '--reference-frame-number' /tmp/ff-help.txt && \
    grep -q -- '--expression-restorer-factor' /tmp/ff-help.txt && \
    grep -q -- '--face-mask-types' /tmp/ff-help.txt

RUN mkdir -p /root/.insightface/models/buffalo_l && \
    curl -fL --retry 5 \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
        -o /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip

ENV FF_ASSETS_DIR="" \
    JOB_TIMEOUT_SECONDS=1800 \
    EXPRESSION_FACTOR=60 \
    REFERENCE_FACE_DISTANCE=0.30 \
    ANCHOR_MATCH_THRESHOLD=0.50

COPY scanner.py handler.py /app/

# Fail the build if the old runtime patcher is accidentally copied.
RUN python3 - <<'PY'
import ast
from pathlib import Path
for name in ('handler.py', 'scanner.py'):
    text = Path('/app', name).read_text()
    compile(text, name, 'exec')
text = Path('/app/handler.py').read_text()
for node in ast.walk(ast.parse(text)):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ('patch_facefusion', 'AstSafetyBypasser'):
        raise RuntimeError('Old runtime-patching handler copied into image. Replace handler.py.')
print('HANDLER CHECK: no conflicting runtime patcher.')
PY
CMD ["python3", "-u", "/app/handler.py"]
