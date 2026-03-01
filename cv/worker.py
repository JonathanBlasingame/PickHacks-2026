import json
import os
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from backend.db import connect, fetch_all, init_db
from cv.occupancy import (
    RegionDef,
    Smoother,
    StationaryVehicleFilter,
    HysteresisFilter,
    centers_from_boxes,
    point_in_poly,
    generate_virtual_spots,
    assign_cars_to_spots,
)
from cv.slot_classifier import SlotClassifier, SlotClassifierConfig, crop_polygon_patch


SAMPLE_EVERY_N_FRAMES = int(os.getenv("SAMPLE_EVERY_N_FRAMES", "10"))
STATIONARY_MIN_SECONDS = float(os.getenv("STATIONARY_MIN_SECONDS", "1.0"))
STATIONARY_MAX_MOTION_PX = float(os.getenv("STATIONARY_MAX_MOTION_PX", "12.0"))
STATIONARY_MATCH_MAX_DIST_PX = float(os.getenv("STATIONARY_MATCH_MAX_DIST_PX", "80.0"))
STATIONARY_MAX_MISSING_SECONDS = float(os.getenv("STATIONARY_MAX_MISSING_SECONDS", "2.0"))
STATIONARY_PERSIST_SECONDS = float(os.getenv("STATIONARY_PERSIST_SECONDS", "0.6"))
SPOT_EMPTY_HOLD_SECONDS = float(os.getenv("SPOT_EMPTY_HOLD_SECONDS", "5.0"))
HYSTERESIS_ENTER_SECONDS = float(os.getenv("HYSTERESIS_ENTER_SECONDS", "0.5"))
HYSTERESIS_EXIT_SECONDS = float(os.getenv("HYSTERESIS_EXIT_SECONDS", "0.75"))
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolo26m.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.05"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.90"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "300"))
YOLO_AGNOSTIC_NMS = os.getenv("YOLO_AGNOSTIC_NMS", "true").strip().lower() in ("1", "true", "yes", "on")
SLOT_CLASSIFIER_MODEL_PATH = os.getenv("SLOT_CLASSIFIER_MODEL_PATH", "").strip()
SLOT_CLASSIFIER_ARCH = os.getenv("SLOT_CLASSIFIER_ARCH", "mAlexNet").strip()
SLOT_CLASSIFIER_BUSY_INDEX = int(os.getenv("SLOT_CLASSIFIER_BUSY_INDEX", "1"))
SLOT_CLASSIFIER_INPUT_SIZE = int(os.getenv("SLOT_CLASSIFIER_INPUT_SIZE", "224"))
SLOT_CLASSIFIER_DEVICE = os.getenv("SLOT_CLASSIFIER_DEVICE", "cpu").strip()
SLOT_CLASSIFIER_WEIGHT = float(os.getenv("SLOT_CLASSIFIER_WEIGHT", "0.35"))
SLOT_CLASSIFIER_DECISION_THRESHOLD = float(os.getenv("SLOT_CLASSIFIER_DECISION_THRESHOLD", "0.5"))
SLOT_CLASSIFIER_MODEL_URL = os.getenv("SLOT_CLASSIFIER_MODEL_URL", "").strip()
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL", "").strip()
YOLO_VEHICLE_CLASS_IDS = os.getenv("YOLO_VEHICLE_CLASS_IDS", "0,1,2").strip()
REGION_REFRESH_EVERY_FRAMES = int(os.getenv("REGION_REFRESH_EVERY_FRAMES", "60"))
REGION_REFRESH_MIN_SECONDS = float(os.getenv("REGION_REFRESH_MIN_SECONDS", "5.0"))

def parse_vehicle_class_ids(raw: str) -> Set[int]:
    out: Set[int] = set()
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            print(f"[WARN] Ignoring invalid class id in YOLO_VEHICLE_CLASS_IDS: {token}")
    if not out:
        out = {0, 1, 2}
        print("[WARN] No valid class IDs configured; defaulting to 0,1,2")
    return out


VEHICLE_CLASS_IDS = parse_vehicle_class_ids(YOLO_VEHICLE_CLASS_IDS)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_checkpoint(root: Path) -> Optional[Path]:
    candidates = sorted(root.rglob("*.pth"))
    return candidates[0] if candidates else None


def _download_and_prepare_model(url: str) -> Optional[str]:
    try:
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(url.split("?")[0]).name or "slot_model.pth"
        target = model_dir / filename
        if not target.exists():
            print(f"[INFO] Downloading slot classifier: {url}")
            urlretrieve(url, str(target))

        suffix = target.suffix.lower()
        if suffix == ".pth":
            return str(target)

        extract_root = model_dir / f"{target.stem}_extracted"
        if extract_root.exists():
            ckpt = _find_checkpoint(extract_root)
            if ckpt is not None:
                return str(ckpt)

        extract_root.mkdir(parents=True, exist_ok=True)
        if suffix == ".zip":
            with zipfile.ZipFile(target, "r") as zf:
                zf.extractall(extract_root)
        elif suffix in (".tgz", ".gz", ".tar"):
            with tarfile.open(target, "r:*") as tf:
                tf.extractall(extract_root)
        else:
            print(f"[WARN] Unsupported model archive format: {target.name}")
            return None

        ckpt = _find_checkpoint(extract_root)
        if ckpt is None:
            print(f"[WARN] No .pth checkpoint found in extracted archive: {target}")
            return None
        return str(ckpt)
    except Exception as exc:
        print(f"[WARN] Failed model download/extract: {exc}")
        return None


def _download_model_file(url: str, model_dir: Path) -> Optional[str]:
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(url.split("?")[0]).name
        if not filename:
            print(f"[WARN] Unable to infer model filename from URL: {url}")
            return None

        target = model_dir / filename
        if not target.exists():
            print(f"[INFO] Downloading detector model: {url}")
            urlretrieve(url, str(target))
        return str(target)
    except Exception as exc:
        print(f"[WARN] Failed detector model download: {exc}")
        return None


def resolve_yolo_model_name(model_name: str) -> str:
    if YOLO_MODEL_URL:
        downloaded = _download_model_file(YOLO_MODEL_URL, Path("data/models"))
        if downloaded:
            return downloaded

    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)

    if len(candidate.parts) == 1:
        in_models_dir = Path("data/models") / model_name
        if in_models_dir.exists():
            return str(in_models_dir)

    if model_name == "yolo26m.pt":
        fallback = "yolov8s.pt"
        print(f"[WARN] Default YOLO26 checkpoint not found ({model_name} or data/models/{model_name}); falling back to {fallback}")
        return fallback

    return model_name


def maybe_load_slot_classifier() -> Optional[SlotClassifier]:
    model_path = SLOT_CLASSIFIER_MODEL_PATH
    if not model_path and SLOT_CLASSIFIER_MODEL_URL:
        model_path = _download_and_prepare_model(SLOT_CLASSIFIER_MODEL_URL) or ""

    if not model_path:
        return None

    try:
        clf = SlotClassifier(
            SlotClassifierConfig(
                model_path=model_path,
                architecture=SLOT_CLASSIFIER_ARCH,
                input_size=SLOT_CLASSIFIER_INPUT_SIZE,
                busy_index=SLOT_CLASSIFIER_BUSY_INDEX,
                device=SLOT_CLASSIFIER_DEVICE,
            )
        )
        print(f"[INFO] Loaded slot classifier arch={SLOT_CLASSIFIER_ARCH} path={model_path}")
        return clf
    except Exception as exc:
        print(f"[WARN] Failed to load slot classifier: {exc}")
        return None


def fused_occupied(geometry_occupied: bool, classifier_busy_prob: Optional[float]) -> bool:
    if classifier_busy_prob is None:
        return bool(geometry_occupied)
    geom_score = 1.0 if geometry_occupied else 0.0
    cls_w = min(max(SLOT_CLASSIFIER_WEIGHT, 0.0), 1.0)
    combined = (1.0 - cls_w) * geom_score + cls_w * float(classifier_busy_prob)
    return combined >= SLOT_CLASSIFIER_DECISION_THRESHOLD


def load_regions_for_camera(camera_id: int) -> List[RegionDef]:
    rows = fetch_all(
        "SELECT id, kind, spot_type, capacity, grid_cols, angle_override_deg, points_json FROM regions WHERE camera_id = ? AND enabled = 1",
        (camera_id,),
    )
    out: List[RegionDef] = []
    for r in rows:
        pts = json.loads(r["points_json"])
        out.append(
            RegionDef(
                id=r["id"],
                kind=r["kind"],
                spot_type=r["spot_type"],
                capacity=int(r["capacity"]),
                grid_cols=r.get("grid_cols"),
                angle_override_deg=r.get("angle_override_deg"),
                points=[(float(x), float(y)) for x, y in pts],
            )
        )
    return out


def _region_signature(reg: RegionDef) -> Tuple[Any, ...]:
    points = tuple((round(float(x), 4), round(float(y), 4)) for (x, y) in reg.points)
    return (
        int(reg.id),
        reg.kind,
        reg.spot_type,
        int(reg.capacity),
        reg.grid_cols,
        reg.angle_override_deg,
        points,
    )


def _regions_changed(current: List[RegionDef], latest: List[RegionDef]) -> bool:
    current_sig = sorted(_region_signature(reg) for reg in current)
    latest_sig = sorted(_region_signature(reg) for reg in latest)
    return current_sig != latest_sig


def cleanup_deleted_region_state(removed_region_ids: List[int]) -> None:
    if not removed_region_ids:
        return
    placeholders = ",".join("?" for _ in removed_region_ids)
    with connect() as conn:
        conn.execute(f"DELETE FROM region_state WHERE region_id IN ({placeholders})", tuple(removed_region_ids))
        conn.execute(f"DELETE FROM zone_virtual_state WHERE region_id IN ({placeholders})", tuple(removed_region_ids))
        conn.commit()


def upsert_state_spot(region_id: int, occupied: bool, confidence: float) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO region_state (region_id, occupied, count, open_spots, confidence, last_updated)
            VALUES (?, ?, 0, 0, ?, ?)
            """,
            (region_id, 1 if occupied else 0, float(confidence), now_iso()),
        )
        conn.commit()


def upsert_state_zone(region_id: int, occupied_count: int, open_spots: int, confidence: float) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO region_state (region_id, occupied, count, open_spots, confidence, last_updated)
            VALUES (?, 0, ?, ?, ?, ?)
            """,
            (region_id, int(occupied_count), int(open_spots), float(confidence), now_iso()),
        )
        conn.commit()


def upsert_zone_virtual(region_id: int, spots_payload: List[Dict[str, Any]]) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO zone_virtual_state (region_id, spots_json, last_updated)
            VALUES (?, ?, ?)
            """,
            (region_id, json.dumps(spots_payload), now_iso()),
        )
        conn.commit()




def upsert_cv_debug(
    camera_id: int,
    vehicles: List[Tuple[float, float]],
    vehicle_boxes: List[Dict[str, Any]],
    detector_meta: Dict[str, Any],
    region_debug: List[Dict[str, Any]],
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cv_debug_state (
                camera_id,
                vehicles_json,
                vehicle_anchors_json,
                vehicle_tires_json,
                vehicle_tire_segments_json,
                vehicle_boxes_json,
                detector_meta_json,
                regions_json,
                last_updated
            )
            VALUES (?, ?, '[]', '[]', '[]', ?, ?, ?, ?)
            """,
            (
                camera_id,
                json.dumps([[float(x), float(y)] for (x, y) in vehicles]),
                json.dumps(vehicle_boxes),
                json.dumps(detector_meta),
                json.dumps(region_debug),
                now_iso(),
            ),
        )
        conn.commit()


@dataclass
class CameraRuntimeState:
    camera_id: int
    video_path: str
    regions: List[RegionDef]
    cap: cv2.VideoCapture
    frame_idx: int
    smoother: Smoother
    stationary_filter: StationaryVehicleFilter
    hysteresis: HysteresisFilter
    frame_interval_seconds: float
    processed_frames_since_refresh: int
    last_region_refresh_ts: float


def main(sample_every_n_frames: int = SAMPLE_EVERY_N_FRAMES, model_name: str = YOLO_MODEL_NAME) -> None:
    init_db()

    cams = fetch_all("SELECT * FROM cameras ORDER BY id ASC")
    if not cams:
        print("No cameras found. Run: python -m backend.seed")
        return

    resolved_model_name = resolve_yolo_model_name(model_name)
    model = YOLO(resolved_model_name)
    print(f"[INFO] YOLO model={resolved_model_name}")
    print(
        "[INFO] YOLO inference thresholds "
        f"conf={YOLO_CONF:.3f} iou={YOLO_IOU:.3f} agnostic_nms={YOLO_AGNOSTIC_NMS} max_det={YOLO_MAX_DET}"
    )
    slot_classifier = maybe_load_slot_classifier()
    states: List[CameraRuntimeState] = []

    for cam in cams:
        cam_id = cam["id"]
        video_path = cam["video_path"]
        if not Path(video_path).exists():
            print(f"[WARN] video not found for camera {cam_id}: {video_path}")
            continue

        regions = load_regions_for_camera(cam_id)
        if not regions:
            print(f"[INFO] No regions defined for camera {cam_id}. Waiting for /static/admin.html updates.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        effective_fps = (fps / sample_every_n_frames) if fps and fps > 0 else 1.0
        frame_interval_seconds = 1.0 / max(effective_fps, 0.001)

        print(f"[INFO] Processing camera {cam_id} video={video_path} regions={len(regions)} source_fps={fps:.2f} analyzed_fps={effective_fps:.2f}")
        states.append(
            CameraRuntimeState(
                camera_id=cam_id,
                video_path=video_path,
                regions=regions,
                cap=cap,
                frame_idx=0,
                smoother=Smoother(window=5, threshold=3, empty_hold_seconds=SPOT_EMPTY_HOLD_SECONDS),
                stationary_filter=StationaryVehicleFilter(
                    min_stationary_seconds=STATIONARY_MIN_SECONDS,
                    max_motion_px=STATIONARY_MAX_MOTION_PX,
                    match_max_dist_px=STATIONARY_MATCH_MAX_DIST_PX,
                    max_missing_seconds=STATIONARY_MAX_MISSING_SECONDS,
                    persist_seconds=STATIONARY_PERSIST_SECONDS,
                ),
                hysteresis=HysteresisFilter(
                    min_enter_seconds=HYSTERESIS_ENTER_SECONDS,
                    min_exit_seconds=HYSTERESIS_EXIT_SECONDS,
                ),
                frame_interval_seconds=frame_interval_seconds,
                processed_frames_since_refresh=0,
                last_region_refresh_ts=time.monotonic(),
            )
        )

    if not states:
        print("[WARN] No valid cameras available for processing.")
        return

    try:
        while True:
            for state in states:
                ok, frame = state.cap.read()
                if not ok:
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state.frame_idx = 0
                    continue

                state.frame_idx += 1
                if state.frame_idx % sample_every_n_frames != 0:
                    continue

                state.processed_frames_since_refresh += 1
                now_mono = time.monotonic()
                should_refresh_regions = (
                    state.processed_frames_since_refresh >= max(1, REGION_REFRESH_EVERY_FRAMES)
                    or (now_mono - state.last_region_refresh_ts) >= max(0.0, REGION_REFRESH_MIN_SECONDS)
                )
                if should_refresh_regions:
                    latest_regions = load_regions_for_camera(state.camera_id)
                    if _regions_changed(state.regions, latest_regions):
                        current_ids = {reg.id for reg in state.regions}
                        latest_ids = {reg.id for reg in latest_regions}
                        removed_region_ids = sorted(current_ids - latest_ids)
                        if removed_region_ids:
                            cleanup_deleted_region_state(removed_region_ids)
                        state.regions = latest_regions
                        print(
                            f"[INFO] cam={state.camera_id} regions refreshed old={len(current_ids)} new={len(latest_ids)} "
                            f"removed={len(removed_region_ids)}"
                        )
                    else:
                        print(f"[INFO] cam={state.camera_id} region refresh checked (no changes)")
                    state.processed_frames_since_refresh = 0
                    state.last_region_refresh_ts = now_mono

                results = model.predict(
                    frame,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    agnostic_nms=YOLO_AGNOSTIC_NMS,
                    max_det=YOLO_MAX_DET,
                    verbose=False,
                )
                r0 = results[0]

                centers: List[Tuple[float, float]] = []
                stationary_flags: List[bool] = []
                vehicle_boxes: List[np.ndarray] = []
                vehicle_classes: List[int] = []
                vehicle_confidences: List[float] = []
                vehicle_mask_polys: List[Optional[List[List[float]]]] = []
                names = getattr(r0, "names", {}) or {}
                mask_polys_all = getattr(r0.masks, "xy", None) if getattr(r0, "masks", None) is not None else None
                if r0.boxes is not None and len(r0.boxes) > 0:
                    boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
                    classes = r0.boxes.cls.cpu().numpy().astype(int).tolist()
                    confidences = r0.boxes.conf.cpu().numpy().astype(float).tolist()

                    vehicle_boxes = []
                    for det_idx, (b, cls, conf) in enumerate(zip(boxes_xyxy, classes, confidences)):
                        if cls in VEHICLE_CLASS_IDS:
                            vehicle_boxes.append(b)
                            vehicle_classes.append(int(cls))
                            vehicle_confidences.append(float(conf))
                            poly = None
                            if mask_polys_all is not None and det_idx < len(mask_polys_all):
                                pts = mask_polys_all[det_idx]
                                if pts is not None and len(pts) >= 3:
                                    poly = [[float(x), float(y)] for (x, y) in pts.tolist()]
                            vehicle_mask_polys.append(poly)

                    if vehicle_boxes:
                        vb = np.array(vehicle_boxes, dtype=np.float32)
                        centers = centers_from_boxes(vb)

                dt_seconds = state.frame_interval_seconds
                stationary_flags = state.stationary_filter.update(centers, boxes_xyxy=vehicle_boxes, dt_seconds=dt_seconds)
                vehicle_boxes_debug = []
                for idx, box in enumerate(vehicle_boxes):
                    x1, y1, x2, y2 = box.tolist()
                    class_id = int(vehicle_classes[idx]) if idx < len(vehicle_classes) else -1
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                    vehicle_boxes_debug.append(
                        {
                            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "mask_poly": vehicle_mask_polys[idx] if idx < len(vehicle_mask_polys) else None,
                            "class_id": class_id,
                            "class_name": str(class_name),
                            "confidence": float(vehicle_confidences[idx]) if idx < len(vehicle_confidences) else 0.0,
                            "stationary": bool(stationary_flags[idx]) if idx < len(stationary_flags) else False,
                        }
                    )

                stationary_boxes = [box for box, is_stationary in zip(vehicle_boxes, stationary_flags) if is_stationary]
                stationary_centers = [p for p, is_stationary in zip(centers, stationary_flags) if is_stationary]

                persisted_tracks = state.stationary_filter.get_persisted_stationary_observations()
                persisted_boxes: List[np.ndarray] = []
                persisted_centers: List[Tuple[float, float]] = []
                if persisted_tracks:
                    persisted_box_tuples = [tr["box_xyxy"] for tr in persisted_tracks]
                    persisted_vb = np.array(persisted_box_tuples, dtype=np.float32)
                    persisted_boxes = [np.array(b, dtype=np.float32) for b in persisted_vb]
                    persisted_centers = centers_from_boxes(persisted_vb)

                occupancy_boxes = stationary_boxes + persisted_boxes

                # Update each region
                region_debug: List[Dict[str, Any]] = []
                for reg in state.regions:
                    if reg.kind == "spot":
                        occupied_live = any(point_in_poly(cx, cy, reg.points) for (cx, cy) in stationary_centers)
                        occupied_held = any(point_in_poly(cx, cy, reg.points) for (cx, cy) in persisted_centers)
                        occupied_now = occupied_live or occupied_held
                        occupied_from_live = occupied_live
                        occupied_from_persisted = occupied_held
                        cls_prob = None
                        if slot_classifier is not None:
                            patch = crop_polygon_patch(frame, reg.points)
                            if patch is not None:
                                cls_prob = slot_classifier.predict_busy_prob(patch)

                        occupied_fused = fused_occupied(occupied_now, cls_prob)
                        occupied_smooth = state.smoother.update(reg.id, occupied_fused, dt_seconds=dt_seconds)
                        occupied_final = state.hysteresis.update(reg.id, occupied_smooth, dt_seconds=dt_seconds)
                        upsert_state_spot(reg.id, occupied_final, confidence=1.0)
                        region_debug.append(
                            {
                                "region_id": reg.id,
                                "kind": "spot",
                                "spot_type": reg.spot_type,
                                "points": [[float(x), float(y)] for (x, y) in reg.points],
                                "occupied": bool(occupied_final),
                                "occupied_raw": bool(occupied_now),
                                "occupied_center": bool(occupied_now),
                                "occupancy_from_live_detection": bool(occupied_from_live),
                                "occupancy_from_persisted_track": bool(occupied_from_persisted),
                                "occupied_fused": bool(occupied_fused),
                                "occupied_smooth": bool(occupied_smooth),
                                "classifier_busy_prob": cls_prob,
                                "detection_mode": "center_point_only",
                                "stationary_vehicles": len(stationary_boxes),
                                "persisted_stationary_tracks": len(persisted_tracks),
                            }
                        )

                    elif reg.kind == "zone":
                        grid_cols = reg.grid_cols or 10
                        spots = generate_virtual_spots(
                            zone_points=reg.points,
                            capacity=reg.capacity,
                            grid_cols=grid_cols,
                            angle_override_deg=reg.angle_override_deg,
                        )
                        occupied_by_center_live = assign_cars_to_spots(stationary_centers, spots, max_dist_px=80.0)
                        occupied_by_center_held = assign_cars_to_spots(persisted_centers, spots, max_dist_px=80.0)
                        occupied_flags_raw = [a or b for a, b in zip(occupied_by_center_live, occupied_by_center_held)]
                        occupied_flags_from_live = list(occupied_by_center_live)
                        occupied_flags_from_persisted = list(occupied_by_center_held)
                        occupied_flags_fused: List[bool] = []
                        occupied_probs: List[Optional[float]] = []
                        for i, spot in enumerate(spots):
                            cls_prob = None
                            if slot_classifier is not None:
                                patch = crop_polygon_patch(frame, spot["poly"])
                                if patch is not None:
                                    cls_prob = slot_classifier.predict_busy_prob(patch)
                            occupied_probs.append(cls_prob)
                            occupied_flags_fused.append(fused_occupied(occupied_flags_raw[i], cls_prob))
                        occupied_flags = [
                            state.hysteresis.update((reg.id, i), occ, dt_seconds=dt_seconds)
                            for i, occ in enumerate(occupied_flags_fused)
                        ]
                        occupied_count = sum(1 for v in occupied_flags if v)
                        open_spots = max(0, reg.capacity - occupied_count)

                        upsert_state_zone(reg.id, occupied_count=occupied_count, open_spots=open_spots, confidence=1.0)

                        payload = []
                        for s, occ in zip(spots, occupied_flags):
                            payload.append(
                                {
                                    "center": [s["center"][0], s["center"][1]],
                                    "poly": [[p[0], p[1]] for p in s["poly"]],
                                    "occupied": bool(occ),
                                    "occupied_raw": bool(occupied_flags_raw[len(payload)]),
                                    "occupied_center": bool(occupied_flags_raw[len(payload)]),
                                    "occupied_overlap": False,
                                    "occupancy_from_live_detection": bool(occupied_flags_from_live[len(payload)]),
                                    "occupancy_from_persisted_track": bool(occupied_flags_from_persisted[len(payload)]),
                                    "occupied_fused": bool(occupied_flags_fused[len(payload)]),
                                    "classifier_busy_prob": occupied_probs[len(payload)],
                                }
                            )
                        upsert_zone_virtual(reg.id, payload)
                        region_debug.append(
                            {
                                "region_id": reg.id,
                                "kind": "zone",
                                "spot_type": reg.spot_type,
                                "points": [[float(x), float(y)] for (x, y) in reg.points],
                                "capacity": reg.capacity,
                                "occupied_count": occupied_count,
                                "open_spots": open_spots,
                                "virtual_spots": payload,
                                "stationary_vehicles": len(stationary_centers),
                                "persisted_stationary_tracks": len(persisted_tracks),
                                "hysteresis_enter_seconds": float(HYSTERESIS_ENTER_SECONDS),
                                "hysteresis_exit_seconds": float(HYSTERESIS_EXIT_SECONDS),
                                "slot_classifier_enabled": bool(slot_classifier is not None),
                                "slot_classifier_weight": float(SLOT_CLASSIFIER_WEIGHT),
                            }
                        )

                detector_meta = {
                    "model": resolved_model_name,
                    "conf": float(YOLO_CONF),
                    "iou": float(YOLO_IOU),
                    "max_det": int(YOLO_MAX_DET),
                    "vehicle_class_ids": sorted(int(cid) for cid in VEHICLE_CLASS_IDS),
                }
                upsert_cv_debug(
                    state.camera_id,
                    centers,
                    vehicle_boxes_debug,
                    detector_meta,
                    region_debug,
                )
                print(
                    f"[{now_iso()}] cam={state.camera_id} vehicles={len(centers)} "
                    f"classes={sorted(VEHICLE_CLASS_IDS)} updated={len(state.regions)}"
                )
            time.sleep(0.1)
    finally:
        for state in states:
            state.cap.release()


if __name__ == "__main__":
    main(sample_every_n_frames=SAMPLE_EVERY_N_FRAMES, model_name=YOLO_MODEL_NAME)
