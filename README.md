# parking-app

Smart parking demo for a *smart cities* hackathon.

## What it does
- Uses a **recorded parking lot video** (MP4) instead of a live stream
- Lets you assign **cameras to lots**
- Lets you draw **spots** (handicap, metered, etc.) and **zones**
- For zones, it generates **virtual parking spots** from a 4-corner zone rectangle/quad using perspective-aware interpolation (`grid_cols`)
- Runs YOLO vehicle detection and assigns detected vehicles to nearest virtual spots to estimate which spaces are occupied

## Quickstart

### 1) Put your video here
`data/videos/lot1.mp4`

(Optional) Put a reference frame here:
`data/frames/lot1_reference.jpg`

### 2) Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Initialize DB
```bash
python -m backend.seed
```

### 4) Start API
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open:
- Dashboard: http://localhost:8000/static/dashboard.html
- Admin: http://localhost:8000/static/admin.html

### 5) Define regions (Admin)
- Draw a **zone** for standard parking using exactly **4 corner points** (document-scan style quad):
  - set capacity (e.g., 60)
  - set grid columns (e.g., 10)
- Draw a few **spots** for handicap / metered

### 6) Start CV worker
In another terminal:
```bash
python cv/worker.py
```

By default spot occupancy now uses a **5-second** empty-state hold to avoid false clears from brief occlusions (e.g., a moving car passing in front). Tune it with:
```bash
SPOT_EMPTY_HOLD_SECONDS=5 python cv/worker.py
```

You can also enable asymmetric hysteresis in **seconds** to require less time to mark a spot busy and more time to mark it free:
```bash
HYSTERESIS_ENTER_SECONDS=0.5 HYSTERESIS_EXIT_SECONDS=0.75 python cv/worker.py
```


Optional: fuse a CNRPark+EXT slot classifier (from `wuyenlin/parking_lot_occupancy_detection`) with geometry occupancy:
```bash
SLOT_CLASSIFIER_MODEL_PATH=data/models/model.pth SLOT_CLASSIFIER_ARCH=mAlexNet SLOT_CLASSIFIER_INPUT_SIZE=224 SLOT_CLASSIFIER_WEIGHT=0.35 SLOT_CLASSIFIER_DECISION_THRESHOLD=0.5 python cv/worker.py
```

Where to store models:
- Recommended location: `data/models/` (e.g. `data/models/model.pth`).
- Best format: **unzipped `.pth` checkpoint**.
- Zipped archives are also supported when using `SLOT_CLASSIFIER_MODEL_URL`; worker auto-extracts and picks the first `.pth` found.

Auto-download options:
- Model only (worker-side):
```bash
SLOT_CLASSIFIER_MODEL_URL=https://.../model.zip SLOT_CLASSIFIER_ARCH=mAlexNet python cv/worker.py
```
- Datasets + splits (script):
```bash
bash scripts/fetch_cnrpark_assets.sh
# optional target dir:
bash scripts/fetch_cnrpark_assets.sh data/datasets
```

Notes:
- `SLOT_CLASSIFIER_MODEL_PATH` or `SLOT_CLASSIFIER_MODEL_URL` enables classifier fusion.
- Supported architectures are `mAlexNet` and `AlexNet` (compatible with the referenced repo).
- `SLOT_CLASSIFIER_WEIGHT` controls classifier influence (`0.0` = geometry only, `1.0` = classifier only).

Spot regions are evaluated using YOLO vehicle center points for stationary detections inside each spot polygon.


Zone occupancy uses YOLO center-point assignment plus box-overlap assignment against generated virtual spots.

Zone virtual spots now follow a perspective-aware warped grid derived from the 4 zone corners, which helps long zones where far-row geometry compresses visually.


Additional timing controls (seconds-based):
```bash
SAMPLE_EVERY_N_FRAMES=10 \
STATIONARY_MIN_SECONDS=1.0 \
STATIONARY_MAX_MISSING_SECONDS=2.0 \
python cv/worker.py
```

By default the worker uses the YOLO26 detector checkpoint (`yolo26m.pt`). The worker first checks that name directly and then `data/models/yolo26m.pt`. If neither exists, it automatically falls back to `yolov8s.pt`. You can still override it explicitly:
```bash
YOLO_MODEL_NAME=yolov8m.pt python cv/worker.py
```

You can also use a custom detector checkpoint URL (for example, an exported Roboflow model) and the worker will download it into `data/models/` automatically:
```bash
YOLO_MODEL_URL=https://example.com/parking-vehicles.pt python cv/worker.py
```

For custom parking datasets (e.g., classes `0:car`, `1:motorcycle`, `2:van`), set the class filter used by occupancy assignment and debug overlays:
```bash
YOLO_VEHICLE_CLASS_IDS=0,1,2 python cv/worker.py
```

The worker now defaults to a very permissive detector threshold (`YOLO_CONF=0.01`) so you can see everything first and then dial it up to remove junk.

To make detections easier to inspect, the debug overlay now paints detected vehicles with a semi-transparent **purple mask**. If the model outputs instance masks (segmentation), it draws the actual polygon; otherwise it falls back to bbox-based masking with the same purple style.

Recommended tuning command:
```bash
YOLO_CONF=0.20 YOLO_IOU=0.50 python cv/worker.py
```

The debug page also includes a temporary **Overlay min confidence** slider so you can visually tune filtering before changing worker env vars.

The dashboard will show open counts and filtering.

## Dataset note: CNRPark+EXT

CNRPark+EXT is a strong candidate **for training and benchmarking occupancy classifiers** in this project:

- ~150k labeled parking-space patches (`free` / `busy`)
- varied weather, viewpoints, shadows, and partial occlusions
- includes slot-level metadata via CSV
- licensed under **ODbL v1.0** (verify attribution/share-alike obligations for your deployment)

Practical caveats before adopting it in production:

- It is from a **single site** (CNR Pisa), so domain shift is likely on new lots/cameras.
- Captures are from 2015-2016; camera quality/layout differs from modern deployments.
- Patch-based labels are ideal for per-slot classifiers, but this app currently does end-to-end detection + geometry assignment.

Recommended use in this repo:

1. Use CNRPark+EXT to pretrain or validate a patch classifier for spot occupancy.
2. Evaluate transfer on your own camera footage before relying on it for live KPIs.
3. Keep this app's zone/spot geometry flow as the fallback when camera domain shift degrades classifier performance.

If you publish a model trained with this dataset, include dataset attribution and ODbL compliance notes in your model card/release docs.
