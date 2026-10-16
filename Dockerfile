# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential \
    ffmpeg git curl unzip libgl1 libglib2.0-0 ca-certificates zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
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

# Only telemetry is patched. Native content and model pre-checks stay intact.
RUN /opt/ffenv/bin/python - <<'PY'
import ast
from pathlib import Path
path=Path('/opt/facefusion/facefusion/execution.py')
tree=ast.parse(path.read_text())
matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_value_and_unit']
if len(matches)!=1 or len(matches[0].args.args)!=1:
    raise RuntimeError('Unsupported telemetry parser; refusing to guess.')
node=matches[0]
name=node.args.args[0].arg
body=(
    "import re\n"
    "if not isinstance(VALUE, str):\n    return None\n"
    "match = re.fullmatch(r'\\s*([0-9]+)\\s+(\\S+)\\s*', VALUE)\n"
    "if match is None:\n    return None\n"
    "return {'value': int(match.group(1)), 'unit': match.group(2)}\n"
).replace('VALUE',name)
node.body=ast.parse(body).body
ast.fix_missing_locations(tree)
result=ast.unparse(tree)+'\n'
compile(result,str(path),'exec')
path.write_text(result)
print('BUILD CHECK: telemetry fixed; native content and model pre-checks unchanged.')
PY
RUN python3 - <<'PY'
import json
import subprocess
from pathlib import Path
code="import json,site,sysconfig; print(json.dumps(sorted(set(site.getsitepackages()+[sysconfig.get_paths()['purelib']]))))"
directories=[]
for interpreter in ('/opt/ffenv/bin/python','/usr/bin/python3'):
    roots=json.loads(subprocess.check_output([interpreter,'-c',code],text=True))
    for root in roots:
        for directory in sorted((Path(root)/'nvidia').glob('*/lib')):
            value=str(directory.resolve())
            if value not in directories:
                directories.append(value)
if not directories:
    raise RuntimeError('NVIDIA library directories not found.')
Path('/etc/ld.so.conf.d/00-puthiyamugam-nvidia.conf').write_text('\n'.join(directories)+'\n')
PY
RUN ldconfig
RUN /opt/ffenv/bin/python - <<'PY'
import sysconfig
from pathlib import Path
path=Path(sysconfig.get_paths()['purelib'])/'sitecustomize.py'
path.write_text("import onnxruntime as ort\nort.preload_dlls(directory='')\n")
PY
RUN /opt/ffenv/bin/python -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('cuDNN 9 load OK')"
COPY handler.py scanner.py ff_bridge.py /app/

# Check the actual checked-out native API, not a different release's docs.
RUN cd /opt/facefusion && /opt/ffenv/bin/python - <<'PY'
import inspect
from facefusion import core
from facefusion.face_analyser import get_many_faces
from facefusion.processors.modules import face_swapper,expression_restorer
from facefusion.processors import choices
from facefusion.program import create_program
from facefusion.args import apply_args
required={'source_face','target_face','temp_vision_frame'}
parameters=set(inspect.signature(face_swapper.swap_face).parameters)
if not required.issubset(parameters) or parameters-required-{'source_vision_frame'}:
    raise RuntimeError('Unsupported swap_face signature: '+str(parameters))
expected={'source_vision_frame','target_face','temp_vision_frame'}
if set(inspect.signature(expression_restorer.restore_expression).parameters)!=expected:
    raise RuntimeError('Unsupported expression-restorer API.')
models={'hyperswap_1a_256','hyperswap_1b_256','hyperswap_1c_256'}
if not models.issubset(set(choices.face_swapper_models)):
    raise RuntimeError('Required HyperSwap models are not supported by this tag.')
print('BUILD CHECK: native multi-face adapter signatures and all HyperSwap models supported.')
PY
RUN python3 - <<'PY'
from pathlib import Path
for name in ('handler.py','scanner.py','ff_bridge.py'):
    compile(Path('/app',name).read_text(),name,'exec')
print('BUILD CHECK: application syntax OK.')
PY
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
CMD ["python3","-u","/app/handler.py"]
