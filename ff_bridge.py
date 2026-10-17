"""Native multi-person FaceFusion adapter. Content classification is not invoked."""
import base64
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, '/opt/facefusion')
from scanner import normalise, match_faces

BUNDLE_ID = 'pm-bundle-20261017-1'
BRIDGE_PATCH = 'content-filter-off-1'


def progress(**data):
    print('PM_PROGRESS:' + json.dumps(data), flush=True)


def initialise(config):
    import onnxruntime as ort
    from facefusion import core, state_manager, logger
    from facefusion import face_classifier, face_detector, face_landmarker
    from facefusion import face_masker, face_recognizer
    from facefusion.args import apply_args
    from facefusion.program import create_program
    from facefusion.processors.modules import face_swapper
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDAExecutionProvider unavailable in FaceFusion environment.')
    processors = ['face_swapper']
    if config.get('restore'):
        processors.append('expression_restorer')
    argv = [
        'headless-run', '--target-path', config['target'],
        '--output-path', config.get('output', '/tmp/pm-scan.png'),
        '--processors', *processors, '--face-swapper-model', config['model'],
        '--face-selector-mode', 'many', '--execution-providers', 'cuda',
        '--face-mask-types', 'box', 'occlusion',
    ]
    if config.get('sources'):
        argv += ['--source-paths', *config['sources'].values()]
    if config.get('restore'):
        argv += ['--expression-restorer-factor', str(config.get('expression_factor', 60))]
    sys.argv = ['facefusion.py', *argv]
    program = create_program()
    args = vars(program.parse_args())
    apply_args(args, state_manager.init_item)
    logger.init(state_manager.get_item('log_level'))
    if not core.pre_check():
        raise RuntimeError('FaceFusion environment pre-check failed.')
    for module in (face_classifier, face_detector, face_landmarker, face_masker, face_recognizer):
        if not module.pre_check():
            raise RuntimeError('Model pre-check failed: ' + module.__name__)
    if not face_swapper.pre_check():
        raise RuntimeError('Swapper model pre-check failed.')
    if config.get('restore'):
        from facefusion.processors.modules import expression_restorer
        if not expression_restorer.pre_check():
            raise RuntimeError('Expression-restorer model pre-check failed.')
    return 'pm-native-v1:' + str(state_manager.get_item('face_recognizer_model'))


def clear_faces():
    from facefusion import face_store
    clear = getattr(face_store, 'clear_static_faces', None)
    if clear:
        clear()


def faces_in(frame):
    from facefusion.face_analyser import get_many_faces
    return get_many_faces([frame])


def usable_fps(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and 0 < value <= 240 else None
    except (TypeError, ValueError):
        return None


def scan(config, identity_space):
    clusters = []
    if config['media_type'] == 'image':
        frame = cv2.imread(config['target'])
        if frame is None:
            raise ValueError('Cannot decode target image.')
        samples = [(0, frame)]
        cap = None
    else:
        cap = cv2.VideoCapture(config['target'])
        if not cap.isOpened():
            cap.release()
            raise ValueError('Cannot open target video or GIF.')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError('Cannot determine frame count.')
        limit = max(1, int(os.getenv('SCAN_FRAMES', '60')))
        samples = [(int(i), None) for i in np.unique(np.linspace(0, total - 1, min(total, limit), dtype=int))]
    threshold = float(os.getenv('SCAN_IDENTITY_SIMILARITY', '0.58'))
    try:
        for sample_number, (index, frame) in enumerate(samples):
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
            detected = faces_in(frame)
            used = set()
            for face in sorted(detected, key=lambda f: -float(np.prod(f.bounding_box[2:] - f.bounding_box[:2]))):
                x1, y1, x2, y2 = face.bounding_box.astype(int)
                if min(x2 - x1, y2 - y1) < 24:
                    continue
                vector = normalise(face.normed_embedding)
                scores = sorted([
                    (max(float(np.dot(vector, v)) for v in c['gallery']), i)
                    for i, c in enumerate(clusters) if i not in used
                ], reverse=True)
                if scores and scores[0][0] >= threshold:
                    ci = scores[0][1]
                else:
                    if len(clusters) >= 40:
                        raise ValueError('More than 40 identity clusters found. Use a shorter clip.')
                    clusters.append({'sum': np.zeros(512, dtype=np.float32), 'gallery': [], 'best': 0, 'crop': None, 'count': 0})
                    ci = len(clusters) - 1
                used.add(ci)
                cluster = clusters[ci]
                cluster['sum'] += vector
                cluster['count'] += 1
                gallery = cluster['gallery']
                if len(gallery) < 8 and (not gallery or max(float(np.dot(vector, v)) for v in gallery) < 0.94):
                    gallery.append(vector)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > cluster['best']:
                    px, py = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
                    crop = frame[max(0, y1 - py):min(frame.shape[0], y2 + py), max(0, x1 - px):min(frame.shape[1], x2 + px)]
                    if crop.size:
                        scale = min(1.0, 256 / max(crop.shape[:2]))
                        cluster['crop'] = cv2.resize(crop, None, fx=scale, fy=scale)
                        cluster['best'] = area
            clear_faces()
            progress(stage='scanning', sampled=sample_number + 1, total_samples=len(samples), identities=len(clusters))
    finally:
        if cap is not None:
            cap.release()
    result = []
    for cluster in clusters:
        if cluster['crop'] is None:
            continue
        ok, buffer = cv2.imencode('.jpg', cluster['crop'], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            continue
        result.append({
            'character_id': str(len(result)),
            'embedding': normalise(cluster['sum']).tolist(),
            'gallery': [v.tolist() for v in cluster['gallery']],
            'thumbnail_base64': base64.b64encode(buffer).decode('ascii'),
            'observations': cluster['count'],
        })
    return {'status': 'success', 'identity_space': identity_space, 'characters': result}


def call_swap(module, source_face, target_face, source_image, frame):
    values = {'source_face': source_face, 'target_face': target_face,
              'source_vision_frame': source_image, 'temp_vision_frame': frame}
    signature = inspect.signature(module.swap_face)
    if set(signature.parameters) - set(values):
        raise RuntimeError('Unsupported FaceFusion swap_face signature: ' + str(signature))
    output = module.swap_face(**{key: values[key] for key in signature.parameters})
    if not isinstance(output, np.ndarray):
        raise RuntimeError('Unsupported FaceFusion swap_face return type.')
    return output


def render(config, identity_space):
    from facefusion import state_manager
    from facefusion.processors.modules import face_swapper
    if config.get('identity_space') != identity_space:
        raise ValueError('Identity model changed. Scan this media again.')
    identities = config['identities']
    if not identities or len(identities) > 40:
        raise ValueError('Scan identities missing or too numerous.')
    for identity in identities:
        normalise(identity['embedding'])
        for vector in identity.get('gallery', []):
            normalise(vector)
    mappings = config['mappings']
    sources = {}
    for key, path in config['sources'].items():
        image = cv2.imread(path)
        if image is None:
            raise ValueError('Cannot decode replacement photo.')
        detected = faces_in(image)
        if len(detected) != 1:
            raise ValueError('Each replacement photo must contain exactly one detectable face.')
        sources[key] = (detected[0], image, path)
        clear_faces()
    threshold = float(os.getenv('MATCH_SIMILARITY', '0.50'))
    margin = float(os.getenv('MATCH_MARGIN', '0.08'))
    counts = {str(i['character_id']): 0 for i in identities if str(i['character_id']) in mappings}
    frame_count, detected_count = 0, 0

    def process(original):
        nonlocal frame_count, detected_count
        detected = faces_in(original)
        detected_count += len(detected)
        matches = match_faces([f.normed_embedding for f in detected], identities, threshold, margin)
        current = original.copy()
        for fi, ii, similarity in matches:
            identifier = str(identities[ii]['character_id'])
            source_key = mappings.get(identifier)
            if source_key is None:
                continue
            source_face, source_image, source_path = sources[source_key]
            state_manager.set_item('source_paths', [source_path])
            current = call_swap(face_swapper, source_face, detected[fi], source_image, current)
            if config.get('restore'):
                from facefusion.processors.modules import expression_restorer
                current = expression_restorer.restore_expression(original, detected[fi], current)
            counts[identifier] += 1
        frame_count += 1
        clear_faces()
        if frame_count == 1 or frame_count % 10 == 0:
            progress(stage='swapping', frames=frame_count, swapped_faces=sum(counts.values()), by_person=counts)
        return current

    if config['media_type'] == 'image':
        frame = cv2.imread(config['target'])
        if frame is None or not cv2.imwrite(config['output'], process(frame)):
            raise RuntimeError('Image processing failed.')
    else:
        info = config['probe']
        width, height = int(info['width']), int(info['height'])
        width += width % 2
        height += height % 2
        fps = usable_fps(info.get('fps'))
        if fps is None:
            raise ValueError('Invalid render FPS.')
        silent = str(Path(config['output']).with_name('silent.mp4'))
        decoder_command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', config['target'],
            '-an', '-vf', f'fps={fps},pad={width}:{height}:0:0',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1',
        ]
        encoder_command = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', os.getenv('OUTPUT_CRF', '18'),
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', silent,
        ]
        log_path = Path(config['output']).with_name('ffmpeg.log')
        with open(log_path, 'wb') as log:
            decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=log)
            encoder = None
            try:
                encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                length = width * height * 3
                while True:
                    data = bytearray()
                    while len(data) < length:
                        block = decoder.stdout.read(length - len(data))
                        if not block:
                            break
                        data.extend(block)
                    if not data:
                        break
                    if len(data) != length:
                        raise RuntimeError('Incomplete video frame during decoding.')
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                    encoder.stdin.write(process(frame).tobytes())
                encoder.stdin.close()
                decoder_code, encoder_code = decoder.wait(), encoder.wait()
                if decoder_code or encoder_code:
                    raise RuntimeError('FFmpeg failed: ' + log_path.read_text(errors='replace')[-2000:])
            finally:
                for child in (decoder, encoder):
                    if child is not None and child.poll() is None:
                        child.kill()
                        child.wait()
                decoder.stdout.close()
        if frame_count == 0:
            raise RuntimeError('No video frames decoded.')
        completed = subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', silent,
            '-i', config['target'], '-map', '0:v:0', '-map', '1:a:0?',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-t', str(frame_count / fps), '-movflags', '+faststart', config['output'],
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError('Audio mux failed: ' + completed.stderr.decode(errors='replace')[-2000:])
    if not sum(counts.values()):
        raise ValueError('No confident matches for assigned people. No swapped result was produced. Scan again or use clearer media.')
    return {
        'status': 'success', 'frames': frame_count, 'detected_faces': detected_count,
        'swapped_by_person': counts,
        'warnings': [f'Person {key} was not confidently matched.' for key, value in counts.items() if not value],
    }


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    started = time.monotonic()
    print('BUNDLE_ID=' + BUNDLE_ID + ' BRIDGE_PATCH=' + BRIDGE_PATCH, flush=True)
    progress(stage='content_filter_disabled', content_filter_enabled=False, bundle_id=BUNDLE_ID)
    identity_space = initialise(config)
    result = scan(config, identity_space) if config['action'] == 'detect' else render(config, identity_space)
    result['processing_seconds'] = round(time.monotonic() - started, 3)
    result['bridge_patch'] = BRIDGE_PATCH
    result['bundle_id'] = BUNDLE_ID
    Path(config['result_json']).write_text(json.dumps(result))


if __name__ == '__main__':
    main()
