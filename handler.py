import os
import sys
import json
import time
import base64
import tempfile
import subprocess
import traceback
from pathlib import Path
import cv2
import runpod
from scanner import app, download, normalise, scan, find_anchor

FF_DIR = '/opt/facefusion'
FF_PYTHON = '/opt/ffenv/bin/python'
FF_SCRIPT = FF_DIR + '/facefusion.py'
PIPELINES = {
    'hs_a': ('hyperswap_1a_256', False),
    'hs_a_expr': ('hyperswap_1a_256', True),
    'hs_b': ('hyperswap_1b_256', False),
    'hs_b_expr': ('hyperswap_1b_256', True),
    'hs_c': ('hyperswap_1c_256', False),
    'hs_c_expr': ('hyperswap_1c_256', True),
}


def configure_assets():
    destination = os.environ.get('FF_ASSETS_DIR', '').strip()
    if not destination:
        return
    # Only accept an existing mounted volume parent, not an accidental local path.
    parent = Path(destination).parent
    if not parent.is_dir():
        raise RuntimeError(f'Asset volume parent does not exist: {parent}')
    Path(destination).mkdir(parents=True, exist_ok=True)
    link = Path(FF_DIR) / '.assets'
    if link.is_symlink():
        if link.resolve() != Path(destination).resolve():
            raise RuntimeError('Existing FaceFusion asset link points elsewhere.')
    elif link.exists():
        raise RuntimeError(
            'FaceFusion .assets already exists. Leave FF_ASSETS_DIR empty '
            'or migrate it into the mounted volume before enabling this setting.'
        )
    else:
        link.symlink_to(destination, target_is_directory=True)


def tail(path, size=10000):
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - size))
        return handle.read().decode('utf-8', errors='replace')


def run_command(command, directory):
    log_path = os.path.join(directory, 'facefusion.log')
    timeout = int(os.environ.get('JOB_TIMEOUT_SECONDS', '1800'))
    started = time.monotonic()
    print('FACEFUSION COMMAND: ' + json.dumps(command), flush=True)
    with open(log_path, 'wb') as output:
        process = subprocess.Popen(
            command, cwd=FF_DIR, stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        previous = ''
        try:
            while True:
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    current = tail(log_path, 2500)
                    if current != previous:
                        print(current, flush=True)
                        previous = current
                    print(f'QUALITY JOB elapsed={elapsed:.1f}s', flush=True)
                    if elapsed > timeout:
                        raise TimeoutError(f'FaceFusion exceeded {timeout}s.\n{current}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    details = tail(log_path)
    print(details, flush=True)
    if code != 0:
        raise RuntimeError(f'FaceFusion exited with code {code}:\n{details}')
    return time.monotonic() - started


def handler(event):
    job = event.get('input', {})
    job_id = event.get('id', 'unknown')
    print(f'JOB START {job_id}', flush=True)
    try:
        if job.get('is_gif'):
            raise ValueError('GIF input is not supported. Convert to MP4.')
        video = bool(job.get('is_video', True))
        media_type = 'video' if video else 'image'
        with tempfile.TemporaryDirectory(prefix='hyperswap_') as directory:
            target = os.path.join(directory, 'target.mp4' if video else 'target.jpg')
            download(job.get('media'), target)
            action = job.get('action', 'swap')
            if action == 'detect':
                return {'status': 'success', 'characters': scan(target, video)}
            if action != 'swap':
                raise ValueError(f'Unknown action: {action}')
            swaps = job.get('swaps', [])
            if len(swaps) != 1:
                raise ValueError('Select exactly ONE replacement person per job.')
            pipeline = job.get('pipeline', 'hs_a')
            # Compatibility with the previous HTML, if it is used accidentally.
            pipeline = {'fast': 'hs_a', 'enhanced': 'hs_a_expr'}.get(pipeline, pipeline)
            if pipeline not in PIPELINES:
                raise ValueError(f'Unknown pipeline: {pipeline}')
            model, restore = PIPELINES[pipeline]
            factor = int(job.get('expression_factor', os.environ.get('EXPRESSION_FACTOR', '60')))
            if not 0 <= factor <= 100:
                raise ValueError('Expression factor must be between 0 and 100.')
            source_input = os.path.join(directory, 'source_input.img')
            download(swaps[0]['new_face'], source_input)
            image = cv2.imread(source_input)
            if image is None:
                raise ValueError('Cannot decode replacement image.')
            faces = app.get(image)
            if len(faces) != 1:
                raise ValueError('Replacement image must contain exactly one detectable face.')
            embedding = normalise(swaps[0]['target_embedding'])
            if embedding.shape != faces[0].normed_embedding.shape:
                raise ValueError('Invalid target embedding dimensions.')
            source = os.path.join(directory, 'source.png')
            if not cv2.imwrite(source, image):
                raise RuntimeError('Could not write replacement image.')
            print('Finding a single-person reference frame...', flush=True)
            reference, similarity = find_anchor(target, video, embedding)
            result = os.path.join(directory, 'result.mp4' if video else 'result.jpg')
            processors = ['face_swapper'] + (['expression_restorer'] if restore else [])
            command = [
                FF_PYTHON, FF_SCRIPT, 'headless-run',
                '--source-paths', source, '--target-path', target,
                '--output-path', result, '--processors', *processors,
                '--face-swapper-model', model,
                '--face-selector-mode', 'reference',
                '--reference-frame-number', str(reference),
                '--reference-face-position', '0',
                '--reference-face-distance', os.environ.get('REFERENCE_FACE_DISTANCE', '0.30'),
                '--face-mask-types', 'box', 'occlusion',
                '--execution-providers', 'cuda',
                '--output-video-encoder', 'libx264',
                '--output-video-preset', 'fast',
                '--output-video-quality', '90',
                '--output-image-quality', '97',
            ]
            if restore:
                command += ['--expression-restorer-factor', str(factor)]
            seconds = run_command(command, directory)
            if not os.path.isfile(result) or not os.path.getsize(result):
                raise RuntimeError('No output produced.\n' + tail(os.path.join(directory, 'facefusion.log')))
            with open(result, 'rb') as handle:
                encoded = base64.b64encode(handle.read()).decode('ascii')
            return {
                'status': 'success', 'media_type': media_type, 'base64': encoded,
                'metrics': {
                    'pipeline': pipeline, 'model': model,
                    'expression_factor': factor if restore else 0,
                    'reference_frame': reference,
                    'reference_similarity': round(similarity, 4),
                    'facefusion_seconds': round(seconds, 3),
                },
            }
    except Exception as exc:
        traceback.print_exc()
        return {'status': 'failed', 'error': str(exc)}
    finally:
        print(f'JOB END {job_id}', flush=True)


if __name__ == '__main__':
    configure_assets()
    runpod.serverless.start({'handler': handler})
