"""Identity matching and bounded media downloads. No face-order assumptions."""
import base64
import ipaddress
import os
import socket
import time
from urllib.parse import urljoin, urlsplit
import numpy as np
import requests

BUNDLE_ID = 'pm-bundle-20261017-1'


def normalise(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size != 512 or not np.isfinite(vector).all():
        raise ValueError('Expected a finite 512-dimensional embedding. Scan again.')
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError('Invalid zero embedding.')
    return vector / norm


def validate_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use an HTTP/HTTPS direct media URL without embedded credentials.')
    allowed = [v.strip().lower() for v in os.getenv('URL_ALLOWED_HOSTS', '').split(',') if v.strip()]
    if allowed and parsed.hostname.lower() not in allowed:
        raise ValueError('URL host is not in URL_ALLOWED_HOSTS.')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('Private, local, or non-public URL addresses are not allowed.')


def download(value, destination):
    if not isinstance(value, str) or not value:
        raise ValueError('Provide a media file or direct URL.')
    maximum = int(os.getenv('MAX_INPUT_BYTES', str(512 * 1024 * 1024)))
    if value.startswith(('https://', 'http://')):
        session = requests.Session()
        session.trust_env = False
        url = value
        started = time.monotonic()
        try:
            for _ in range(6):
                validate_url(url)
                with session.get(url, stream=True, timeout=(20, 90), allow_redirects=False) as response:
                    if response.is_redirect or response.is_permanent_redirect:
                        url = urljoin(url, response.headers.get('Location', ''))
                        continue
                    response.raise_for_status()
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise ValueError('That link returns a webpage, not a media file.')
                    length = response.headers.get('Content-Length')
                    if length and int(length) > maximum:
                        raise ValueError('Remote file exceeds MAX_INPUT_BYTES.')
                    size = 0
                    with open(destination, 'wb') as handle:
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            size += len(block)
                            if size > maximum or time.monotonic() - started > 600:
                                raise ValueError('Media download exceeded its size or time limit.')
                            handle.write(block)
                    if size == 0:
                        raise ValueError('Downloaded file is empty.')
                    return size
            raise ValueError('Too many redirects.')
        finally:
            session.close()
    if value.startswith('data:'):
        header, value = value.split(',', 1)
        if ';base64' not in header:
            raise ValueError('Expected a base64 data URI.')
    if len(value) > ((maximum + 2) // 3) * 4 + 8:
        raise ValueError('Encoded file exceeds MAX_INPUT_BYTES.')
    data = base64.b64decode(value, validate=True)
    if not data or len(data) > maximum:
        raise ValueError('Invalid file size.')
    with open(destination, 'wb') as handle:
        handle.write(data)
    return len(data)


def match_faces(vectors, identities, threshold=0.50, margin=0.08):
    """Match faces from the ORIGINAL frame against all scanned identities."""
    if not vectors or not identities:
        return []
    candidates = []
    for face_index, vector in enumerate(vectors):
        vector = normalise(vector)
        scores = []
        for identity_index, identity in enumerate(identities):
            gallery = identity.get('gallery') or [identity['embedding']]
            score = max(float(np.dot(vector, normalise(sample))) for sample in gallery)
            scores.append((score, identity_index))
        scores.sort(reverse=True)
        score, identity_index = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else -1.0
        if score >= threshold and score - runner_up >= margin:
            candidates.append((score, face_index, identity_index))
    matches, used_faces, used_ids = [], set(), set()
    for score, face_index, identity_index in sorted(candidates, reverse=True):
        if face_index not in used_faces and identity_index not in used_ids:
            matches.append((face_index, identity_index, score))
            used_faces.add(face_index)
            used_ids.add(identity_index)
    return matches
