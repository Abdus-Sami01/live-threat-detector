# Benchmarks

## Latency (Modal)

X3D-M action branch, 16-frame clip @ 224², FP32, batch 1, 100-clip average.

| GPU  | classify ms/window | detect ms (est) | fuse ms | end-to-end ms | est fps | real-time (>15) |
|------|--------------------|-----------------|---------|---------------|---------|-----------------|
| L40S | 28.61              | 8.0             | 0.2     | ~36.8         | 27.2    | ✓               |

Measured 2026-07-23 via `modal run -m threat_detector.bench.modal_app::bench`.
End-to-end excludes the N1 cascade skip — on real CCTV the action branch runs on a
fraction of windows, so effective throughput is higher than the per-window figure.

## Pending

- FP16 + `torch.compile` classify pass (expect notable ms drop).
- Detect branch measured (not estimated) once YOLO weights fine-tuned.
- UCF-Crime false-alarms/hour at the calibrated operating point.
- A40 / A100 comparison row.
