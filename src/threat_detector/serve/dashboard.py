"""Minimal upload dashboard: POST a video, get an alert timeline back.

    uvicorn threat_detector.serve.dashboard:app
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse

from ..action.x3d_branch import X3DConfig, X3DRecognizer
from ..core.frames import VideoFileSource
from ..core.pipeline import Pipeline
from ..detection.yolo_branch import YoloConfig, YoloDetector

app = FastAPI(title="Live Threat Detector")

_PAGE = """<!doctype html><title>Threat Detector</title>
<h2>Live Threat Detector</h2>
<form action="/analyze" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="video/*" required>
  <button type="submit">Analyze</button>
</form>
<p>Returns a JSON alert timeline.</p>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


@app.post("/analyze")
async def analyze(file: UploadFile):
    tmp = Path(tempfile.gettempdir()) / f"upload_{file.filename}"
    tmp.write_bytes(await file.read())
    source = VideoFileSource(str(tmp))
    pipe = Pipeline(YoloDetector(YoloConfig()), X3DRecognizer(X3DConfig()), fps=source.fps)
    events = pipe.run(source)
    tmp.unlink(missing_ok=True)
    return {
        "timeline": [
            {"time_s": round(e.time_s, 2), "threat": e.threat.value,
             "confidence": round(e.confidence, 3)}
            for e in events
        ],
        "alerts": len(events),
        "compute_saved": round(pipe.stats.compute_saved, 4),
    }
