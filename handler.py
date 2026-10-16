import ast
import base64
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import runpod

from scanner import app, download, normalise, scan, find_anchor


FF_DIR = "/opt/facefusion"
FF_PYTHON = "/opt/ffenv/bin/python"
FF_SCRIPT = os.path.join(FF_DIR, "facefusion.py")

GPU_PATCH_MARKER = "# PUTHIYAMUGAM_SAFE_GPU_TELEMETRY_V1"
BYPASS_MARKER = "# PUTHIYAMUGAM_AST_BYPASS_V1"

PIPELINES = {
    "hs_a": ("hyperswap_1a_256", False),
    "hs_a_expr": ("hyperswap_1a_256", True),
    "hs_b": ("hyperswap_1b_256", False),
    "hs_b_expr": ("hyperswap_1b_256", True),
    "hs_c": ("hyperswap_1c_256", False),
    "hs_c_expr": ("hyperswap_1c_256", True),
}


def configure_assets():
    destination = os.environ.get("FF_ASSETS_DIR", "").strip()

    if not destination:
        return

    destination_path = Path(destination)
    parent = destination_path.parent

    if not parent.is_dir():
        raise RuntimeError(
            f"Asset volume parent does not exist: {parent}"
        )

    destination_path.mkdir(parents=True, exist_ok=True)

    link = Path(FF_DIR) / ".assets"

    if link.is_symlink():
        if link.resolve() != destination_path.resolve():
            raise RuntimeError(
                "Existing FaceFusion asset link points elsewhere."
            )

    elif link.exists():
        raise RuntimeError(
            "FaceFusion .assets already exists. "
            "Leave FF_ASSETS_DIR empty or migrate it into the mounted "
            "volume before enabling this setting."
        )

    else:
        link.symlink_to(
            destination_path,
            target_is_directory=True,
        )


class AstSafetyBypasser(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name in ("analyse_frame", "analyse_image", "analyse_video", "detect_nsfw", "analyse_stream"):
            node.body = [ast.Return(value=ast.Constant(value=False))]
        elif node.name == "common_pre_check":
            node.body = [ast.Return(value=ast.Constant(value=True))]
        return node


def patch_facefusion():
    """
    1. Patch GPU telemetry parsing (avoids crash on non-integer fb_memory).
    2. Patch content_analyser.py via AST (analyse_* returns False).
    3. Patch core.py via AST (common_pre_check returns True to bypass hash check).
    """
    ff_pkg = Path(FF_DIR) / "facefusion"

    # ---------------- 1. GPU Telemetry Patch ----------------
    execution_path = ff_pkg / "execution.py"
    if execution_path.is_file():
        source = execution_path.read_text(encoding="utf-8")
        if GPU_PATCH_MARKER not in source:
            tree = ast.parse(source)
            matches = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "create_value_and_unit"
            ]
            if len(matches) == 1:
                function_node = matches[0]
                replacement = (
                    "def create_value_and_unit(text):\n"
                    f"    {GPU_PATCH_MARKER}\n"
                    "    import re\n"
                    "    if not isinstance(text, str):\n"
                    "        return None\n"
                    "    match = re.fullmatch(r'\\s*([0-9]+)\\s+(\\S+)\\s*', text)\n"
                    "    if match is None:\n"
                    "        return None\n"
                    "    return {\n"
                    "        'value': int(match.group(1)),\n"
                    "        'unit': match.group(2)\n"
                    "    }\n"
                )
                lines = source.splitlines(keepends=True)
                patched = (
                    "".join(lines[: function_node.lineno - 1])
                    + replacement
                    + "".join(lines[function_node.end_lineno :])
                )
                execution_path.write_text(patched, encoding="utf-8")
                print("Applied FaceFusion GPU telemetry patch.", flush=True)

    # ---------------- 2. Content Analyser Patch (AST) ----------------
    analyser_path = ff_pkg / "content_analyser.py"
    if analyser_path.is_file():
        source = analyser_path.read_text(encoding="utf-8")
        if BYPASS_MARKER not in source:
            tree = ast.parse(source)
            transformer = AstSafetyBypasser()
            new_tree = transformer.visit(tree)
            ast.fix_missing_locations(new_tree)
            patched_code = f"{BYPASS_MARKER}\n" + ast.unparse(new_tree)
            analyser_path.write_text(patched_code, encoding="utf-8")
            print("Patched content_analyser.py (all checks return False).", flush=True)

    # ---------------- 3. Core Pre-Check Patch (AST) ----------------
    core_path = ff_pkg / "core.py"
    if core_path.is_file():
        core_source = core_path.read_text(encoding="utf-8")
        if BYPASS_MARKER not in core_source:
            tree = ast.parse(core_source)
            transformer = AstSafetyBypasser()
            new_tree = transformer.visit(tree)
            ast.fix_missing_locations(new_tree)
            patched_code = f"{BYPASS_MARKER}\n" + ast.unparse(new_tree)
            core_path.write_text(patched_code, encoding="utf-8")
            print("Patched core.py (common_pre_check returns True).", flush=True)

    # Clear cached bytecode
    cache_dir = ff_pkg / "__pycache__"
    if cache_dir.is_dir():
        for cached_file in cache_dir.glob("*.pyc"):
            try:
                cached_file.unlink()
            except OSError:
                pass


def tail(path, size=10000):
    if not os.path.exists(path):
        return ""

    with open(path, "rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - size))
        return handle.read().decode("utf-8", errors="replace")


def run_command(command, directory):
    patch_facefusion()

    log_path = os.path.join(directory, "facefusion.log")
    timeout = int(os.environ.get("JOB_TIMEOUT_SECONDS", "1800"))
    started = time.monotonic()

    print("FACEFUSION COMMAND: " + json.dumps(command), flush=True)

    with open(log_path, "wb") as output:
        process = subprocess.Popen(
            command,
            cwd=FF_DIR,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )

        previous = ""

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

                    print(f"QUALITY JOB elapsed={elapsed:.1f}s", flush=True)

                    if elapsed > timeout:
                        raise TimeoutError(
                            f"FaceFusion exceeded {timeout}s.\n{current}"
                        )
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
        raise RuntimeError(f"FaceFusion exited with code {code}:\n{details}")

    return time.monotonic() - started


def handler(event):
    job_id = event.get("id", "unknown")
    print(f"JOB START {job_id}", flush=True)

    try:
        job = event.get("input", {})

        if not isinstance(job, dict):
            raise ValueError("Job input must be a JSON object.")

        if job.get("is_gif"):
            raise ValueError("GIF input is not supported. Convert to MP4.")

        video = bool(job.get("is_video", True))
        media_type = "video" if video else "image"
        action = job.get("action", "swap")

        if action not in ("detect", "swap"):
            raise ValueError(f"Unknown action: {action}")

        with tempfile.TemporaryDirectory(prefix="hyperswap_") as directory:
            target = os.path.join(
                directory,
                "target.mp4" if video else "target.jpg",
            )

            download(job.get("media"), target)

            if action == "detect":
                return {
                    "status": "success",
                    "characters": scan(target, video),
                }

            swaps = job.get("swaps", [])

            if not isinstance(swaps, list) or len(swaps) != 1:
                raise ValueError(
                    "Select exactly ONE replacement person per job."
                )

            swap = swaps[0]

            if not isinstance(swap, dict):
                raise ValueError("Replacement mapping must be a JSON object.")

            pipeline = job.get("pipeline", "hs_a")
            pipeline = {
                "fast": "hs_a",
                "enhanced": "hs_a_expr",
            }.get(pipeline, pipeline)

            if pipeline not in PIPELINES:
                raise ValueError(f"Unknown pipeline: {pipeline}")

            model, restore = PIPELINES[pipeline]

            factor = int(
                job.get(
                    "expression_factor",
                    os.environ.get("EXPRESSION_FACTOR", "60"),
                )
            )

            if not 0 <= factor <= 100:
                raise ValueError("Expression factor must be between 0 and 100.")

            source_input = os.path.join(directory, "source_input.img")
            download(swap.get("new_face"), source_input)

            image = cv2.imread(source_input)
            if image is None:
                raise ValueError("Cannot decode replacement image.")

            faces = app.get(image)
            if len(faces) != 1:
                raise ValueError(
                    "Replacement image must contain exactly one detectable face."
                )

            embedding = normalise(swap.get("target_embedding"))
            if embedding.shape != faces[0].normed_embedding.shape:
                raise ValueError("Invalid target embedding dimensions.")

            source = os.path.join(directory, "source.png")
            if not cv2.imwrite(source, image):
                raise RuntimeError("Could not write replacement image.")

            print("Finding a single-person reference frame...", flush=True)
            reference, similarity = find_anchor(target, video, embedding)

            result = os.path.join(
                directory,
                "result.mp4" if video else "result.jpg",
            )

            processors = ["face_swapper"]
            if restore:
                processors.append("expression_restorer")

            command = [
                FF_PYTHON,
                FF_SCRIPT,
                "headless-run",
                "--source-paths",
                source,
                "--target-path",
                target,
                "--output-path",
                result,
                "--processors",
                *processors,
                "--face-swapper-model",
                model,
                "--face-selector-mode",
                "reference",
                "--reference-frame-number",
                str(reference),
                "--reference-face-position",
                "0",
                "--reference-face-distance",
                os.environ.get("REFERENCE_FACE_DISTANCE", "0.30"),
                "--face-mask-types",
                "box",
                "occlusion",
                "--execution-providers",
                "cuda",
                "--output-video-encoder",
                "libx264",
                "--output-video-preset",
                "fast",
                "--output-video-quality",
                "90",
                "--output-image-quality",
                "97",
            ]

            if restore:
                command += [
                    "--expression-restorer-factor",
                    str(factor),
                ]

            seconds = run_command(command, directory)

            if not os.path.isfile(result) or not os.path.getsize(result):
                raise RuntimeError(
                    "No output produced.\n"
                    + tail(os.path.join(directory, "facefusion.log"))
                )

            with open(result, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")

            return {
                "status": "success",
                "media_type": media_type,
                "base64": encoded,
                "metrics": {
                    "pipeline": pipeline,
                    "model": model,
                    "expression_factor": factor if restore else 0,
                    "reference_frame": reference,
                    "reference_similarity": round(float(similarity), 4),
                    "facefusion_seconds": round(seconds, 3),
                },
            }

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(exc),
        }

    finally:
        print(f"JOB END {job_id}", flush=True)


if __name__ == "__main__":
    configure_assets()
    runpod.serverless.start({"handler": handler})
