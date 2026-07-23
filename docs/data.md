# Datasets

## Quick download (free sets)

RWF-2000 (fights) and D-Fire (fire/smoke) pull + reorganize automatically via
kagglehub (needs Kaggle creds + accepted dataset terms):

```bash
pip install -e ".[data]"
PYTHONPATH=src python scripts/download_data.py rwf     # -> data/action/{train,val}/{fight,normal}
PYTHONPATH=src python scripts/download_data.py dfire   # -> data/yolo + data.yaml
```

RWF-2000 is research-only (no commercial use / redistribution without SMIIP Lab
approval). Fall (URFD/le2i), loiter, weapon, and UCF-Crime sets are staged manually.

## Action clips (X3D head)

Layout: `data/action/<split>/<label>/*.mp4`, split ∈ {train,val,test},
label ∈ {fight,fall,loiter,normal}. Sources: RWF-2000 (fight), URFD/le2i (fall),
normal CCTV clips, loiter clips.

Build manifest:

```bash
PYTHONPATH=src python -c "from threat_detector.data.ingest import prepare_action_manifest as p; print(p('data/action','data/manifests/action.jsonl'))"
```

## Detection images (YOLO head)

Ultralytics layout under `data/yolo/images/{train,val}` + `labels/...`, classes
`person,weapon,fire,smoke`. Sources: weapon (pistol/knife sets), D-Fire/FireNet.

Write config:

```bash
PYTHONPATH=src python -c "from threat_detector.data.yolo_dataset import write_data_yaml as w; w('data/yolo','data/yolo/data.yaml')"
```

## UCF-Crime (eval only)

Layout: `ucf_crime/{normal,anomaly}/*.mp4`. Used solely for FA/hour calibration.

## Modal Volume upload

```bash
modal volume create threat-detector-data
modal volume put threat-detector-data data/manifests /manifests
modal volume put threat-detector-data data/action /action
modal volume put threat-detector-data data/yolo /yolo
modal volume put threat-detector-data ucf_crime /ucf_crime
```

## Run order

1. `modal run -m threat_detector.bench.modal_app::train_yolo`
2. `modal run -m threat_detector.bench.modal_app::train_x3d`
3. `modal run -m threat_detector.bench.modal_app::eval_ucf`
