# Body-camera face anonymizer

Streaming redaction for body-worn camera video. Faces the live detector finds are blurred, then an independent RetinaFace pass flags residual visible faces. Duplicate upload events reuse the first job.

Write-up: [`docs/Shivam_Pronto_Bodycam_Face_Anonymizer_Report.pdf`](docs/Shivam_Pronto_Bodycam_Face_Anonymizer_Report.pdf)

Weights and sample clips are **not** in this repo. `python -m face_anon download-models --sample` fetches them.

```text
upload event --> idempotent job store --> YuNet + ByteTrack + hybrid redact
                                                |
                                                v
                                      RetinaFace QC on the output
                                                |
                                   pass  or  needs_review (exit 2)
```

Default live path: YuNet at 640 px, privacy-tuned ByteTrack (start 0.28, keep 0.18), adaptive hybrid ellipse. Quality check uses RetinaFace-ResNet50 on the finished file. CUDA is used when present; the same code falls back to CPU.

## Install

```bash
pip install -e .
pip install onnxruntime-gpu==1.23.2   # optional, CUDA 12
python -m face_anon download-models --sample
```

## Run

```bash
# one video
PYTHONPATH=src python -m face_anon process \
  --input data/samples/airport_5min.mp4 \
  --output outputs/airport_5min_anon.mp4

# CPU-only, or SCRFD-10G on GPU
PYTHONPATH=src python -m face_anon --config configs/cpu_yunet.yaml process --input ... --output ...
PYTHONPATH=src python -m face_anon --config configs/gpu_scrfd.yaml process --input ... --output ...

# async upload (safe to send twice)
PYTHONPATH=src python -m face_anon submit --upload-id evt-001 --input data/samples/airport_5min.mp4
PYTHONPATH=src python -m face_anon worker --once
PYTHONPATH=src python -m face_anon status --job-id <id>

# proxy recall + residual faces
PYTHONPATH=src python -m face_anon eval \
  --input data/samples/airport_5min.mp4 \
  --output outputs/airport_5min_anon.mp4 \
  --max-frames 600

PYTHONPATH=src python -m pytest -q
```

`process` writes the mp4 and a sidecar JSON (throughput, faces painted, QC). Exit `0` is a pass. Exit `2` means a person should review.

## Layout

```text
configs/                  default, CPU, and SCRFD-GPU settings
src/face_anon/
  factory.py              detector / tracker / QC construction
  pipeline.py             encode loop + optional retry
  detect_yunet.py         live detector
  detect_scrfd.py         optional live detector
  detect_retinaface.py    independent QC detector
  track.py                IoU coast + ByteTrack
  anonymize.py            hybrid ellipse redact
  qc.py                   sampled output audit
  jobs.py                 async + idempotent store
  cli.py                  process | submit | worker | status | eval
scripts/                  ablation and body-cam fixture
tests/
docs/                     assignment report
```

## Evaluation

There is no public box-labelled body-camera set for this brief, so evaluation is layered:

1. Unit tests (geometry, pad, tracker, QC budget, job reuse).
2. Residual-visible faces on the output (the operational SLA).
3. Proxy recall vs RetinaFace on the input (`eval`).
4. Full-video ablation: `scripts/run_ablation.py` and `scripts/run_pr_sweep.py`.

On the 20-minute airport video, YuNet + privacy-tuned ByteTrack reached 0.990 recall on 3,690 stand-in faces. On a 5-minute end-to-end run the default path was 41.5 FPS (1.38x real time). Details are in `docs/Shivam_Pronto_Bodycam_Face_Anonymizer_Report.pdf`.
