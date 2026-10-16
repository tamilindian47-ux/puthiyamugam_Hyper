import os
import base64
import cv2
import numpy as np
import requests
import onnxruntime as ort
import insightface

ort.preload_dlls()
if 'CUDAExecutionProvider' not in ort.get_available_providers():
    raise RuntimeError('CUDAExecutionProvider is unavailable.')

app = insightface.app.FaceAnalysis(
    name='buffalo_l', root='/root/.insightface',
    allowed_modules=['detection', 'recognition'],
    providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'})],
)
app.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))
for name, model in app.models.items():
    active = model.session.get_providers()
    print(f'SCANNER {name}: {active}', flush=True)
    if 'CUDAExecutionProvider' not in active:
        raise RuntimeError(f'Scanner {name} has no CUDA provider.')
cv2.setNumThreads(1)


def normalise(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError('Invalid embedding.')
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError('Zero-length embedding.')
    return vector / norm


def download(value, path):
    if not isinstance(value, str) or not value:
        raise ValueError('Expected a URL or base64 media string.')
    if value.startswith(('https://', 'http://')):
        with requests.get(value, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with open(path, 'wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    else:
        if value.startswith('data:'):
            value = value.split(',', 1)[1]
        with open(path, 'wb') as handle:
            handle.write(base64.b64decode(value))


def frames(path, video, limit):
    if not video:
        image = cv2.imread(path)
        if image is None:
            raise ValueError('Cannot decode image.')
        yield 0, image
        return
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video.')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError('Cannot determine frame count. Re-export as MP4.')
        indices = np.unique(np.linspace(0, total - 1, min(total, limit), dtype=int))
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                yield int(index), frame
    finally:
        cap.release()


def scan(path, video):
    clusters = []
    for _, frame in frames(path, video, 30):
        for face in app.get(frame):
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if min(x2 - x1, y2 - y1) < 30:
                continue
            vector = face.normed_embedding
            cluster = next((c for c in clusters if float(np.dot(
                vector, normalise(c['sum']))) >= 0.58), None)
            if cluster is None:
                cluster = {'sum': np.zeros_like(vector), 'score': -1, 'crop': None}
                clusters.append(cluster)
            cluster['sum'] += vector
            score = (x2 - x1) * (y2 - y1)
            if score > cluster['score']:
                cluster['score'] = score
                cluster['crop'] = frame[y1:y2, x1:x2].copy()
    result = []
    for index, cluster in enumerate(clusters):
        ok, buffer = cv2.imencode('.jpg', cluster['crop'])
        if ok:
            result.append({
                'character_id': index,
                'embedding': normalise(cluster['sum']).tolist(),
                'thumbnail_base64': base64.b64encode(buffer).decode('ascii'),
            })
    return result


def find_anchor(path, video, embedding):
    threshold = float(os.environ.get('ANCHOR_MATCH_THRESHOLD', '0.50'))
    best = None
    for index, frame in frames(path, video, 120):
        faces = app.get(frame)
        if len(faces) != 1:
            continue
        score = float(np.dot(faces[0].normed_embedding, embedding))
        if score >= threshold and (best is None or score > best[1]):
            best = (index, score)
    if best is None:
        raise ValueError(
            'No unambiguous reference frame found. This test worker needs '
            'one clear frame with ONLY the selected person detectable. '
            'It will not guess face ordering in a group frame.'
        )
    return best
