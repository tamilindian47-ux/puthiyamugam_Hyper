# PuthiyaMugam HyperSwap quality-test worker

Separate RunPod Serverless deployment. Do not overwrite your working endpoint.

## Included pipelines

- hs_a: HyperSwap 1A
- hs_a_expr: HyperSwap 1A + expression restoration
- hs_b: HyperSwap 1B
- hs_b_expr: HyperSwap 1B + expression restoration
- hs_c: HyperSwap 1C
- hs_c_expr: HyperSwap 1C + expression restoration

These are test candidates, not a ranking or a guarantee of better pose/expression quality.
No GPEN enhancement is used. Expression restoration is a different operation.

## Deploy

1. Put Dockerfile, requirements.txt, scanner.py and handler.py in the new GitHub repository root.
2. Build an image using the Dockerfile, push it to a registry, and configure a NEW RunPod Serverless endpoint to use that image. Use your existing GitHub-to-image build process if already configured. Pushing GitHub alone does not deploy without that automation.
3. Set the RunPod job execution timeout high enough for your test (the worker timeout is 1800 seconds). A platform timeout can terminate it sooner.
4. Use one concurrent job per worker. The scanner is shared and inference resources are not isolated for concurrent calls.
5. Open PuthiyaMugam.html locally. Enter the NEW endpoint ID and API key; neither is persisted by this code.
6. Start with a 5-10 second constant-frame-rate MP4. Scan, select ONE person, provide a single-face source photo, run pipeline 1, then pipeline 2 on exactly the same clip.
7. Repeat with 1B and 1C and compare identity, mouth/eye expression, occlusion, temporal flicker and processing time.

## First execution / assets

FaceFusion models are NOT pre-downloaded in the image. Initial jobs can download extra model weights. This requires outbound internet and adds latency. First-job timings are not steady-state speed measurements. The FaceFusion subprocess starts and reloads models for each swap job; this quality-test worker does not keep swap models resident between jobs.

Optional: mount a RunPod network volume at /runpod-volume and set FF_ASSETS_DIR=/runpod-volume/facefusion-assets. The directory is used as FaceFusion's .assets directory via a symlink. Leave this variable empty if no volume is mounted. Test with a single worker first; simultaneous initial downloads from multiple workers are not coordinated here.

## Limitations

- One selected replacement identity per job. Other people may appear but matching is reference-based and is not infallible.
- Requires at least one sampled frame with only the selected person detectable. Group-only footage is rejected instead of guessing cross-library face ordering.
- The reference search is sampled (up to 120 frames), not exhaustive. It can miss a brief suitable frame.
- FaceFusion and InsightFace may detect different faces. A one-face InsightFace anchor reduces ambiguity but does not mathematically guarantee FaceFusion picks the same face. Inspect output with multiple people carefully.
- MP4/video and still photos only; GIF unsupported.
- Very large browser payloads/output can exceed RunPod request/response limits or phone memory. A direct input URL avoids embedding input bytes in JSON, but this prototype still returns base64 output. Production large-file handling needs object storage and output URLs.
- Base64 uploads are repeated for scan and swap; no media cache is implemented.
- No claim of improved quality, speed or production readiness: container build, GPU execution and visual comparisons still require testing on your endpoint.
- Build pins FaceFusion 3.3.0 and checks required CLI flags. It is not a full reproducible dependency lock. If upstream dependencies fail, inspect build logs instead of silently moving to another release.
- Check upstream FaceFusion and model licensing/terms before commercial use. Use only media and identities you have permission to process. Upstream content safeguards are retained.
- No NVENC. Output uses CPU libx264, FaceFusion quality=90 (not equivalent to CRF 17).
- GPU providers being registered does not establish that every individual graph node executes on GPU.

## Troubleshooting

RunPod worker logs include the FaceFusion command, elapsed time and the captured FaceFusion log tail. Returned errors include the actual subprocess failure. A log heartbeat is not a frame-progress measurement.

The browser submits once via /run and polls the returned ID. If polling fails, Resume polling uses the SAME ID without resubmitting. If submission itself times out before an ID is received, check RunPod before submitting again: the job could already exist.

Keep the Android browser open and awake during tests. Android browser/background behaviour and CORS access from a local file must be tested on the specific device; this code cannot guarantee those platform behaviours.

To preset the endpoint ID, edit the HTML input's value attribute. Do not put API keys in a public repository. There is no localStorage, sessionStorage, service worker or third-party script.
