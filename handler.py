"""Runpod entry point: native multi-person processing and chunked transfers."""

import base64
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
import uuid

from PIL import Image, ImageOps
import runpod

from scanner import download


BUILD_ID = "pm-multi-native-v2"

FF_DIR = "/opt/facefusion"
FF_PYTHON = "/opt/ffenv/bin/python"

CHUNK_BYTES = 1024 * 1024

PIPELINES = {
    "hs_a": ("hyperswap_1a_256", False),
    "hs_a_expr": ("hyperswap_1a_256", True),
    "hs_b": ("hyperswap_1b_256", False),
    "hs_b_expr": ("hyperswap_1b_256", True),
    "hs_c": ("hyperswap_1c_256", False),
    "hs_c_expr": ("hyperswap_1c_256", True),
}

ACTIONS = (
    "detect",
    "swap",
    "begin_upload",
    "upload_chunk",
    "finish_upload",
    "fetch_chunk",
    "result_manifest",
    "result_for_job",
    "capabilities",
)


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(CHUNK_BYTES),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.parent / (
        uuid.uuid4().hex + ".tmp"
    )

    try:
        temporary.write_text(
            json.dumps(value),
            encoding="utf-8",
        )

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)


def valid_id(value, label):
    if (
        not isinstance(value, str)
        or not re.fullmatch("[a-f0-9]{32}", value)
    ):
        raise ValueError("Invalid " + label + ".")

    return value


def results_root():
    volume = Path("/runpod-volume")

    if not volume.is_dir():
        raise RuntimeError(
            "Attach the SAME shared Runpod network volume "
            "to all workers at /runpod-volume. "
            "Chunk uploads/downloads require it."
        )

    root = Path(
        os.getenv(
            "RESULTS_DIR",
            "/runpod-volume/pm-results",
        )
    )

    if volume.resolve() not in root.resolve().parents:
        raise RuntimeError(
            "RESULTS_DIR must be inside /runpod-volume."
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return root


def upload_root():
    root = results_root().parent / "pm-uploads"

    root.mkdir(exist_ok=True)

    return root


def upload_request(job):
    action = job["action"]

    if action == "begin_upload":
        size = job.get("size_bytes")

        maximum = int(
            os.getenv(
                "MAX_INPUT_BYTES",
                str(512 * CHUNK_BYTES),
            )
        )

        if (
            type(size) is not int
            or not 0 < size <= maximum
        ):
            raise ValueError(
                "Invalid upload size; exceeds "
                "MAX_INPUT_BYTES or is empty."
            )

        identifier = uuid.uuid4().hex

        folder = upload_root() / identifier
        folder.mkdir()

        metadata = {
            "size_bytes": size,
            "chunk_bytes": CHUNK_BYTES,
            "chunk_count": math.ceil(
                size / CHUNK_BYTES
            ),
            "expires_at": int(time.time()) + 86400,
            "complete": False,
        }

        atomic_json(
            folder / "upload.json",
            metadata,
        )

        return {
            "status": "success",
            "upload_id": identifier,
            **metadata,
        }

    identifier = valid_id(
        job.get("upload_id"),
        "upload ID",
    )

    folder = upload_root() / identifier

    if not (folder / "upload.json").is_file():
        raise ValueError(
            "Upload not found. Check the shared "
            "volume or upload again."
        )

    # Serialize modifications to a single upload.
    with open(folder / ".lock", "a+b") as lock:
        fcntl.flock(
            lock,
            fcntl.LOCK_EX,
        )

        metadata = json.loads(
            (folder / "upload.json").read_text()
        )

        if time.time() > metadata["expires_at"]:
            raise ValueError(
                "Upload expired. Choose the file again."
            )

        if action == "upload_chunk":
            if metadata["complete"]:
                raise ValueError(
                    "Upload is already finalized."
                )

            index = job.get("index")

            if (
                type(index) is not int
                or not 0 <= index < metadata["chunk_count"]
            ):
                raise ValueError(
                    "Invalid upload chunk index."
                )

            encoded = job.get("base64")

            maximum_encoded = (
                (CHUNK_BYTES + 2) // 3
            ) * 4

            if (
                not isinstance(encoded, str)
                or len(encoded) > maximum_encoded
            ):
                raise ValueError(
                    "Invalid or oversized upload chunk."
                )

            data = base64.b64decode(
                encoded,
                validate=True,
            )

            expected = min(
                CHUNK_BYTES,
                metadata["size_bytes"]
                - index * CHUNK_BYTES,
            )

            if len(data) != expected:
                raise ValueError(
                    "Upload chunk has the wrong length."
                )

            temporary = folder / (
                uuid.uuid4().hex + ".tmp"
            )

            try:
                temporary.write_bytes(data)

                temporary.replace(
                    folder / f"{index}.part"
                )

            finally:
                temporary.unlink(missing_ok=True)

            return {
                "status": "success",
                "index": index,
                "size_bytes": len(data),
            }

        if action == "finish_upload":
            claimed = job.get("sha256")

            if claimed is not None and (
                not isinstance(claimed, str)
                or not re.fullmatch(
                    "[a-f0-9]{64}",
                    claimed,
                )
            ):
                raise ValueError(
                    "Invalid upload checksum."
                )

            if not metadata["complete"]:
                temporary = folder / (
                    uuid.uuid4().hex + ".tmp"
                )

                digest = hashlib.sha256()

                try:
                    with open(temporary, "wb") as handle:
                        for index in range(
                            metadata["chunk_count"]
                        ):
                            part = folder / f"{index}.part"

                            if not part.is_file():
                                raise ValueError(
                                    f"Upload chunk {index} "
                                    "is missing."
                                )

                            data = part.read_bytes()

                            expected = min(
                                CHUNK_BYTES,
                                metadata["size_bytes"]
                                - index * CHUNK_BYTES,
                            )

                            if len(data) != expected:
                                raise ValueError(
                                    f"Upload chunk {index} "
                                    "is incomplete."
                                )

                            digest.update(data)
                            handle.write(data)

                    checksum = digest.hexdigest()

                    if claimed and claimed != checksum:
                        raise ValueError(
                            "Upload checksum mismatch."
                        )

                    temporary.replace(
                        folder / "media.bin"
                    )

                    metadata.update(
                        complete=True,
                        sha256=checksum,
                    )

                    atomic_json(
                        folder / "upload.json",
                        metadata,
                    )

                finally:
                    temporary.unlink(missing_ok=True)

                for part in folder.glob("*.part"):
                    part.unlink(missing_ok=True)

            if (
                claimed
                and claimed != metadata["sha256"]
            ):
                raise ValueError(
                    "Upload checksum mismatch."
                )

            return {
                "status": "success",
                "upload_id": identifier,
                **metadata,
            }

    raise ValueError(
        "Unknown upload action: " + repr(action)
    )


def materialise(value, destination):
    """
    Accept either:
      - URL/base64 string
      - {"upload_id": "..."} from a completed chunk upload
    """

    if not isinstance(value, dict):
        return download(
            value,
            destination,
        )

    identifier = valid_id(
        value.get("upload_id"),
        "upload reference",
    )

    folder = upload_root() / identifier
    metadata_file = folder / "upload.json"

    if not metadata_file.is_file():
        raise ValueError(
            "Uploaded media not found on the shared volume."
        )

    metadata = json.loads(
        metadata_file.read_text()
    )

    if (
        not metadata.get("complete")
        or time.time() > metadata["expires_at"]
    ):
        raise ValueError(
            "Media upload is incomplete or expired."
        )

    source = folder / "media.bin"

    if (
        not source.is_file()
        or source.stat().st_size != metadata["size_bytes"]
    ):
        raise ValueError(
            "Uploaded media file is missing or incomplete."
        )

    shutil.copyfile(
        source,
        destination,
    )

    return Path(destination).stat().st_size


def store_result(
    path,
    media_type,
    mime,
    metrics,
    job_id,
):
    root = results_root()

    artifact = uuid.uuid4().hex

    folder = root / artifact
    folder.mkdir()

    destination = folder / (
        "result" + Path(path).suffix
    )

    shutil.copyfile(
        path,
        destination,
    )

    size = destination.stat().st_size

    manifest = {
        "status": "success",
        "delivery": "chunks",
        "artifact_id": artifact,
        "filename": destination.name,
        "mime_type": mime,
        "media_type": media_type,
        "size_bytes": size,
        "chunk_bytes": CHUNK_BYTES,
        "chunk_count": math.ceil(
            size / CHUNK_BYTES
        ),
        "sha256": sha256(destination),
        "expires_at": (
            int(time.time())
            + int(
                os.getenv(
                    "RESULT_TTL_SECONDS",
                    "86400",
                )
            )
        ),
        "metrics": metrics,
        "build_id": BUILD_ID,
        "job_id": job_id,
    }

    atomic_json(
        folder / "manifest.json",
        manifest,
    )

    # Save an index so the HTML can recover a stored
    # result using its processing job ID.
    index = root / "_jobs"
    index.mkdir(exist_ok=True)

    name = (
        hashlib.sha256(
            job_id.encode()
        ).hexdigest()
        + ".json"
    )

    atomic_json(
        index / name,
        {"artifact_id": artifact},
    )

    return manifest


def read_manifest(artifact):
    artifact = valid_id(
        artifact,
        "artifact ID",
    )

    folder = results_root() / artifact
    file = folder / "manifest.json"

    if not file.is_file():
        raise ValueError(
            "Result not found. Check that all workers "
            "mount the SAME network volume."
        )

    manifest = json.loads(
        file.read_text()
    )

    if time.time() > manifest["expires_at"]:
        raise ValueError(
            "Stored result has expired."
        )

    return folder, manifest


def result_for_job(job_id):
    if (
        not isinstance(job_id, str)
        or not re.fullmatch(
            "[A-Za-z0-9_-]{1,200}",
            job_id,
        )
    ):
        raise ValueError(
            "Invalid job ID."
        )

    name = (
        hashlib.sha256(
            job_id.encode()
        ).hexdigest()
        + ".json"
    )

    file = results_root() / "_jobs" / name

    if not file.is_file():
        raise ValueError(
            "No indexed result for this job. "
            "It may have failed, not finished, or used "
            "the previous handler. Older results can "
            "still be retrieved by artifact ID "
            "if available."
        )

    index = json.loads(
        file.read_text()
    )

    return read_manifest(
        index["artifact_id"]
    )[1]


def cleanup_expired():
    locations = (
        (
            results_root(),
            "manifest.json",
        ),
        (
            upload_root(),
            "upload.json",
        ),
    )

    for root, name in locations:
        files = list(
            root.glob("*/" + name)
        )[:1000]

        for file in files:
            try:
                data = json.loads(
                    file.read_text()
                )

                if (
                    time.time()
                    > data["expires_at"] + 3600
                ):
                    shutil.rmtree(
                        file.parent
                    )

                    if (
                        name == "manifest.json"
                        and data.get("job_id")
                    ):
                        index = (
                            hashlib.sha256(
                                data["job_id"].encode()
                            ).hexdigest()
                            + ".json"
                        )

                        (
                            root
                            / "_jobs"
                            / index
                        ).unlink(missing_ok=True)

            except (
                OSError,
                ValueError,
                KeyError,
            ):
                continue


def fetch_chunk(job):
    folder, manifest = read_manifest(
        job.get("artifact_id")
    )

    index = job.get("index")

    if (
        type(index) is not int
        or not 0 <= index < manifest["chunk_count"]
    ):
        raise ValueError(
            "Invalid result chunk index."
        )

    with open(
        folder / manifest["filename"],
        "rb",
    ) as handle:
        handle.seek(
            index * manifest["chunk_bytes"]
        )

        data = handle.read(
            manifest["chunk_bytes"]
        )

    expected = min(
        manifest["chunk_bytes"],
        manifest["size_bytes"]
        - index * manifest["chunk_bytes"],
    )

    if len(data) != expected:
        raise RuntimeError(
            "Stored output is incomplete."
        )

    return {
        "status": "success",
        "index": index,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(
            data
        ).hexdigest(),
        "base64": base64.b64encode(
            data
        ).decode("ascii"),
    }


def progress(event, **data):
    print(
        "PROGRESS " + json.dumps(data),
        flush=True,
    )

    try:
        runpod.serverless.progress_update(
            event,
            data,
        )

    except Exception as error:
        print(
            "Progress update unavailable: "
            + str(error)[:200],
            flush=True,
        )


def image_to_png(source, destination):
    with Image.open(source) as image:
        if (
            image.width * image.height
            > 50_000_000
        ):
            raise ValueError(
                "Image exceeds 50 megapixels."
            )

        image = ImageOps.exif_transpose(
            image
        )

        image.convert("RGB").save(
            destination,
            "PNG",
        )


def probe_media(path, directory):
    still = False
    is_gif = False

    try:
        with Image.open(path) as image:
            animated = bool(
                getattr(
                    image,
                    "is_animated",
                    False,
                )
            )

            still = not animated

            is_gif = (
                animated
                and image.format == "GIF"
            )

    except OSError:
        pass

    if still:
        destination = str(
            Path(directory) / "target.png"
        )

        image_to_png(
            path,
            destination,
        )

        with Image.open(destination) as image:
            return (
                destination,
                "image",
                {
                    "width": image.width,
                    "height": image.height,
                },
            )

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,"
                "avg_frame_rate,r_frame_rate:"
                "stream_side_data=rotation:"
                "format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if completed.returncode:
        raise ValueError(
            "Cannot decode media. Provide an actual "
            "media file or direct media URL."
        )

    info = json.loads(
        completed.stdout
    )

    streams = info.get(
        "streams",
        [],
    )

    if not streams:
        raise ValueError(
            "No image/video stream found."
        )

    stream = streams[0]

    width = int(
        stream["width"]
    )

    height = int(
        stream["height"]
    )

    rotation = next(
        (
            float(item["rotation"])
            for item in stream.get(
                "side_data_list",
                [],
            )
            if "rotation" in item
        ),
        0,
    )

    if abs(round(rotation)) % 180 == 90:
        width, height = height, width

    maximum_pixels = int(
        os.getenv(
            "MAX_VIDEO_PIXELS",
            str(3840 * 2160),
        )
    )

    if width * height > maximum_pixels:
        raise ValueError(
            "Video exceeds MAX_VIDEO_PIXELS."
        )

    rate = None

    for name in (
        "avg_frame_rate",
        "r_frame_rate",
    ):
        try:
            candidate = Fraction(
                stream.get(
                    name,
                    "0/1",
                )
            )

            if 0 < float(candidate) <= 240:
                rate = candidate
                break

        except (
            ValueError,
            ZeroDivisionError,
        ):
            pass

    if is_gif:
        rate = Fraction(
            os.getenv(
                "GIF_FPS",
                "25",
            )
        )

    if (
        rate is None
        or not 0 < float(rate) <= 240
    ):
        raise ValueError(
            "Cannot determine a valid frame rate."
        )

    duration = float(
        info.get(
            "format",
            {},
        ).get(
            "duration",
            0,
        )
        or 0
    )

    maximum_duration = float(
        os.getenv(
            "MAX_DURATION_SECONDS",
            "600",
        )
    )

    if duration > maximum_duration:
        raise ValueError(
            "Media exceeds MAX_DURATION_SECONDS."
        )

    return (
        str(path),
        "gif" if is_gif else "video",
        {
            "width": width,
            "height": height,
            "fps": float(rate),
            "duration": duration,
        },
    )


def tail(path, limit=12000):
    with open(path, "rb") as handle:
        handle.seek(
            0,
            2,
        )

        handle.seek(
            max(
                0,
                handle.tell() - limit,
            )
        )

        return handle.read().decode(
            "utf-8",
            errors="replace",
        )


def run_bridge(config, directory, event):
    path = Path(directory) / "config.json"

    config["result_json"] = str(
        Path(directory) / "bridge-result.json"
    )

    path.write_text(
        json.dumps(config)
    )

    log_path = (
        Path(directory)
        / "processing.log"
    )

    limit = int(
        os.getenv(
            "JOB_TIMEOUT_SECONDS",
            "3600",
        )
    )

    started = time.monotonic()

    with open(log_path, "wb") as log:
        child = subprocess.Popen(
            [
                FF_PYTHON,
                "-u",
                "/app/ff_bridge.py",
                str(path),
            ],
            cwd=FF_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )

        previous = ""

        try:
            while child.poll() is None:
                time.sleep(3)

                if (
                    time.monotonic() - started
                    > limit
                ):
                    raise TimeoutError(
                        "Processing exceeded "
                        "JOB_TIMEOUT_SECONDS."
                    )

                lines = [
                    line[len("PM_PROGRESS:"):]
                    for line in tail(
                        log_path
                    ).splitlines()
                    if line.startswith(
                        "PM_PROGRESS:"
                    )
                ]

                if (
                    lines
                    and lines[-1] != previous
                ):
                    try:
                        update = json.loads(
                            lines[-1]
                        )

                    except json.JSONDecodeError:
                        continue

                    previous = lines[-1]

                    progress(
                        event,
                        **update,
                    )

            details = tail(
                log_path
            )

            print(
                details,
                flush=True,
            )

            if child.returncode:
                raise RuntimeError(
                    "FaceFusion processing failed:\n"
                    + details[-5000:]
                )

        finally:
            if child.poll() is None:
                os.killpg(
                    child.pid,
                    signal.SIGTERM,
                )

                try:
                    child.wait(
                        timeout=10
                    )

                except subprocess.TimeoutExpired:
                    os.killpg(
                        child.pid,
                        signal.SIGKILL,
                    )

                    child.wait()

    result_file = Path(
        config["result_json"]
    )

    if not result_file.is_file():
        raise RuntimeError(
            "Processor exited without a result manifest."
        )

    return json.loads(
        result_file.read_text()
    )


def handler(event):
    identifier = event.get(
        "id",
        "unknown",
    )

    print(
        f"JOB START {identifier} BUILD={BUILD_ID}",
        flush=True,
    )

    try:
        job = event.get(
            "input",
            {},
        )

        if not isinstance(job, dict):
            raise ValueError(
                "Input must be a JSON object."
            )

        action = job.get(
            "action",
            "swap",
        )

        print(
            "ACTION=" + repr(action),
            flush=True,
        )

        if action not in ACTIONS:
            raise ValueError(
                f"Unknown action {action!r}. "
                f"Supported: {', '.join(ACTIONS)}"
            )

        # Lightweight version/protocol check.
        if action == "capabilities":
            return {
                "status": "success",
                "build_id": BUILD_ID,
                "actions": list(ACTIONS),
                "chunk_bytes": CHUNK_BYTES,
            }

        # Chunked input upload routes.
        if action in (
            "begin_upload",
            "upload_chunk",
            "finish_upload",
        ):
            return upload_request(job)

        # Chunked output download route.
        if action == "fetch_chunk":
            return fetch_chunk(job)

        # Recover using a saved artifact ID.
        if action == "result_manifest":
            return read_manifest(
                job.get("artifact_id")
            )[1]

        # Recover using the original processing job ID.
        if action == "result_for_job":
            return result_for_job(
                job.get("job_id")
            )

        # Only detect/swap continue into media processing.
        if action == "swap":
            results_root()
            cleanup_expired()

        pipeline = job.get(
            "pipeline",
            "hs_a",
        )

        if pipeline not in PIPELINES:
            raise ValueError(
                "Unknown pipeline. "
                "Use the multi-person HTML."
            )

        model, restore = PIPELINES[
            pipeline
        ]

        factor = int(
            job.get(
                "expression_factor",
                60,
            )
        )

        if not 0 <= factor <= 100:
            raise ValueError(
                "Expression factor must be 0–100."
            )

        output_format = job.get(
            "output_format",
            "mp4",
        )

        if output_format not in (
            "mp4",
            "gif",
        ):
            raise ValueError(
                "Output format must be mp4 or gif."
            )

        with tempfile.TemporaryDirectory(
            prefix="pm_multi_"
        ) as directory:
            incoming = (
                Path(directory)
                / "input.bin"
            )

            progress(
                event,
                stage="downloading",
            )

            original_bytes = materialise(
                job.get("media"),
                incoming,
            )

            media_hash = sha256(
                incoming
            )

            if (
                action == "swap"
                and job.get("media_sha256")
                != media_hash
            ):
                raise ValueError(
                    "Target changed since scanning. "
                    "Scan this exact file/link again."
                )

            target, kind, probe = probe_media(
                incoming,
                directory,
            )

            config = {
                "action": action,
                "target": target,
                "media_type": kind,
                "probe": probe,
                "model": model,
                "restore": (
                    restore
                    if action == "swap"
                    else False
                ),
                "expression_factor": factor,
            }

            if action == "detect":
                result = run_bridge(
                    config,
                    directory,
                    event,
                )

                result.update(
                    media_sha256=media_hash,
                    input_bytes=original_bytes,
                    media_type=kind,
                    probe=probe,
                    build_id=BUILD_ID,
                )

                return result

            identities = job.get(
                "identities"
            )

            mappings = job.get(
                "mappings"
            )

            source_inputs = job.get(
                "sources"
            )

            if (
                not isinstance(identities, list)
                or not 1 <= len(identities) <= 40
            ):
                raise ValueError(
                    "Scan faces first."
                )

            if not all(
                isinstance(item, dict)
                and "character_id" in item
                for item in identities
            ):
                raise ValueError(
                    "Invalid identity records. Scan again."
                )

            if (
                not isinstance(mappings, dict)
                or not mappings
                or not isinstance(source_inputs, dict)
            ):
                raise ValueError(
                    "Assign at least one replacement photo."
                )

            ids = [
                str(item["character_id"])
                for item in identities
            ]

            if (
                len(set(ids)) != len(ids)
                or any(
                    str(key) not in ids
                    for key in mappings
                )
            ):
                raise ValueError(
                    "Invalid/duplicate identity mapping."
                )

            if any(
                not isinstance(key, str)
                or key not in source_inputs
                for key in mappings.values()
            ):
                raise ValueError(
                    "A mapped replacement photo is missing."
                )

            sources = {}

            unique_source_keys = dict.fromkeys(
                mappings.values()
            )

            for index, key in enumerate(
                unique_source_keys
            ):
                raw = (
                    Path(directory)
                    / f"source-{index}.bin"
                )

                materialise(
                    source_inputs[key],
                    raw,
                )

                destination = (
                    Path(directory)
                    / f"source-{index}.png"
                )

                image_to_png(
                    raw,
                    destination,
                )

                sources[key] = str(
                    destination
                )

            output = Path(directory) / (
                "result.png"
                if kind == "image"
                else "result.mp4"
            )

            config.update(
                sources=sources,
                identities=identities,
                mappings=mappings,
                identity_space=job.get(
                    "identity_space"
                ),
                output=str(output),
            )

            result = run_bridge(
                config,
                directory,
                event,
            )

            mime = (
                "image/png"
                if kind == "image"
                else "video/mp4"
            )

            output_kind = (
                "image"
                if kind == "image"
                else "video"
            )

            if (
                kind != "image"
                and output_format == "gif"
            ):
                progress(
                    event,
                    stage="encoding_gif",
                )

                gif = (
                    Path(directory)
                    / "result.gif"
                )

                completed = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(output),
                        "-filter_complex",
                        (
                            "split[s0][s1];"
                            "[s0]palettegen[p];"
                            "[s1][p]paletteuse"
                        ),
                        "-loop",
                        "0",
                        str(gif),
                    ],
                    capture_output=True,
                    timeout=int(
                        os.getenv(
                            "JOB_TIMEOUT_SECONDS",
                            "3600",
                        )
                    ),
                )

                if completed.returncode:
                    raise RuntimeError(
                        "GIF encoding failed: "
                        + completed.stderr.decode(
                            errors="replace"
                        )[-2000:]
                    )

                output = gif
                mime = "image/gif"
                output_kind = "gif"

            if (
                not output.is_file()
                or output.stat().st_size == 0
            ):
                raise RuntimeError(
                    "No output file was produced."
                )

            progress(
                event,
                stage="saving_result",
                output_bytes=output.stat().st_size,
            )

            result.update(
                pipeline=pipeline,
                model=model,
                input_bytes=original_bytes,
                expression_factor=(
                    factor
                    if restore
                    else 0
                ),
            )

            return store_result(
                output,
                output_kind,
                mime,
                result,
                identifier,
            )

    except Exception:
        traceback.print_exc()
        raise

    finally:
        print(
            "JOB END " + str(identifier),
            flush=True,
        )


def configure_assets():
    destination = os.getenv(
        "FF_ASSETS_DIR",
        "",
    ).strip()

    if not destination:
        return

    path = Path(
        destination
    )

    if not path.parent.is_dir():
        raise RuntimeError(
            "FF_ASSETS_DIR parent volume is not mounted."
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    link = (
        Path(FF_DIR)
        / ".assets"
    )

    if (
        link.is_symlink()
        and link.resolve() == path.resolve()
    ):
        return

    if (
        link.exists()
        or link.is_symlink()
    ):
        raise RuntimeError(
            "FaceFusion .assets already exists. "
            "Leave FF_ASSETS_DIR empty or migrate it first."
        )

    link.symlink_to(
        path,
        target_is_directory=True,
    )


if __name__ == "__main__":
    configure_assets()

    print(
        "READY "
        + BUILD_ID
        + "; complete chunk-transfer routes; "
        "no runtime source rewriting.",
        flush=True,
    )

    runpod.serverless.start(
        {"handler": handler}
    )
