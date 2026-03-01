import json
from datetime import datetime, timezone
import re
from pathlib import Path

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.db import execute, fetch_all, fetch_one, init_db
from backend.models import BulkSpotCreate, Camera, CameraCreate, CameraUpdate, Lot, LotCreate, LotUpdate, Region, RegionCreate
import math

app = FastAPI(title="Smart Parking (Recorded Video + Virtual Spots)")

static_dir = Path("backend/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
def _startup() -> None:
    init_db()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenCV is required for frame endpoints but is unavailable in this runtime",
        ) from exc
    return cv2


def _order_quad_points(points: List[List[float]]) -> List[List[float]]:
    if len(points) != 4:
        return [[float(x), float(y)] for x, y in points]

    pts = [[float(x), float(y)] for x, y in points]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    # Sort points clockwise around centroid.
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # Start from the top-left-ish point (smallest x+y) for stable ordering.
    start = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    ordered = pts[start:] + pts[:start]

    # Ensure clockwise order maps to [tl, tr, br, bl].
    # If winding is counter-clockwise, reverse middle ordering.
    tl, p1, p2, p3 = ordered
    cross = (p1[0] - tl[0]) * (p2[1] - tl[1]) - (p1[1] - tl[1]) * (p2[0] - tl[0])
    if cross > 0:
        ordered = [ordered[0], ordered[3], ordered[2], ordered[1]]

    return ordered


def _lerp(a: List[float], b: List[float], t: float) -> List[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def _quad_point(tl: List[float], tr: List[float], br: List[float], bl: List[float], u: float, v: float) -> List[float]:
    top = _lerp(tl, tr, u)
    bottom = _lerp(bl, br, u)
    return _lerp(top, bottom, v)


def _generate_virtual_spots_from_quad(points: List[List[float]], capacity: int, grid_cols: int) -> List[Dict[str, Any]]:
    cols = max(1, int(grid_cols))
    rows = max(1, int(math.ceil(float(capacity) / float(cols))))
    tl, tr, br, bl = _order_quad_points(points)

    spots: List[Dict[str, Any]] = []
    for r in range(rows):
        v0 = r / rows
        v1 = (r + 1) / rows
        for c in range(cols):
            u0 = c / cols
            u1 = (c + 1) / cols
            p_tl = _quad_point(tl, tr, br, bl, u0, v0)
            p_tr = _quad_point(tl, tr, br, bl, u1, v0)
            p_br = _quad_point(tl, tr, br, bl, u1, v1)
            p_bl = _quad_point(tl, tr, br, bl, u0, v1)
            center = _quad_point(tl, tr, br, bl, (u0 + u1) / 2.0, (v0 + v1) / 2.0)
            spots.append({
                "center": [float(center[0]), float(center[1])],
                "poly": [
                    [float(p_tl[0]), float(p_tl[1])],
                    [float(p_tr[0]), float(p_tr[1])],
                    [float(p_br[0]), float(p_br[1])],
                    [float(p_bl[0]), float(p_bl[1])],
                ],
            })
    return spots[:capacity]


def _point_in_ring(lat: float, lon: float, ring: List[List[float]]) -> bool:
    """Return True if point (lat, lon) is inside a GeoJSON linear ring."""
    if not ring or len(ring) < 3:
        return False

    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_in_polygon_coords(lat: float, lon: float, coords: List[Any]) -> bool:
    if not coords:
        return False

    outer = coords[0]
    if not _point_in_ring(lat, lon, outer):
        return False

    for hole in coords[1:]:
        if _point_in_ring(lat, lon, hole):
            return False

    return True


def _geojson_contains_point(geojson: Dict[str, Any], lat: float, lon: float) -> bool:
    gtype = geojson.get("type")

    if gtype == "Polygon":
        return _point_in_polygon_coords(lat, lon, geojson.get("coordinates") or [])

    if gtype == "MultiPolygon":
        polygons = geojson.get("coordinates") or []
        return any(_point_in_polygon_coords(lat, lon, poly) for poly in polygons)

    if gtype == "Feature":
        geometry = geojson.get("geometry") or {}
        return _geojson_contains_point(geometry, lat, lon)

    if gtype == "FeatureCollection":
        for feature in geojson.get("features") or []:
            if _geojson_contains_point(feature, lat, lon):
                return True
        return False

    return False


def _find_lot_id_for_point(lat: Optional[float], lon: Optional[float]) -> Optional[int]:
    if lat is None or lon is None:
        return None

    lots = fetch_all("SELECT id, polygon_geojson FROM lots WHERE polygon_geojson IS NOT NULL AND polygon_geojson != '' ORDER BY id ASC")
    for lot in lots:
        polygon_geojson = lot.get("polygon_geojson")
        if not polygon_geojson:
            continue
        try:
            geo = json.loads(polygon_geojson)
        except json.JSONDecodeError:
            continue
        if _geojson_contains_point(geo, float(lat), float(lon)):
            return int(lot["id"])
    return None


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "ui": "/static/dashboard.html", "admin": "/static/admin.html", "debug": "/static/debug.html"}


# Lots
@app.post("/api/lots", response_model=Lot)
def create_lot(payload: LotCreate) -> Any:
    lot_id = execute(
        "INSERT INTO lots (name, lot_group, lat, lon, polygon_geojson) VALUES (?, ?, ?, ?, ?)",
        (payload.name, payload.lot_group, payload.lat, payload.lon, payload.polygon_geojson),
    )
    return fetch_one("SELECT * FROM lots WHERE id = ?", (lot_id,))


@app.get("/api/lots", response_model=List[Lot])
def list_lots() -> Any:
    return fetch_all("SELECT * FROM lots ORDER BY id ASC")



@app.put("/api/lots/{lot_id}", response_model=Lot)
def update_lot(lot_id: int, payload: LotUpdate) -> Any:
    lot = fetch_one("SELECT * FROM lots WHERE id = ?", (lot_id,))
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")

    fields = []
    params = []
    if payload.name is not None:
        fields.append("name = ?")
        params.append(payload.name)
    if payload.lot_group is not None:
        fields.append("lot_group = ?")
        params.append(payload.lot_group)
    if payload.lat is not None:
        fields.append("lat = ?")
        params.append(payload.lat)
    if payload.lon is not None:
        fields.append("lon = ?")
        params.append(payload.lon)
    if payload.polygon_geojson is not None:
        fields.append("polygon_geojson = ?")
        params.append(payload.polygon_geojson)

    if fields:
        params.append(lot_id)
        execute(f"UPDATE lots SET {', '.join(fields)} WHERE id = ?", tuple(params))

    return fetch_one("SELECT * FROM lots WHERE id = ?", (lot_id,))

# Cameras
@app.post("/api/cameras", response_model=Camera)
def create_camera(payload: CameraCreate) -> Any:
    lot = fetch_one("SELECT * FROM lots WHERE id = ?", (payload.lot_id,))
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")

    cam_id = execute(
        "INSERT INTO cameras (lot_id, name, video_path, reference_frame_path, lat, lon) VALUES (?, ?, ?, ?, ?, ?)",
        (payload.lot_id, payload.name, payload.video_path, payload.reference_frame_path, payload.lat, payload.lon),
    )
    return fetch_one("SELECT * FROM cameras WHERE id = ?", (cam_id,))


@app.get("/api/cameras", response_model=List[Camera])
def list_cameras(lot_id: Optional[int] = None) -> Any:
    if lot_id is None:
        return fetch_all("SELECT * FROM cameras ORDER BY id ASC")
    return fetch_all("SELECT * FROM cameras WHERE lot_id = ? ORDER BY id ASC", (lot_id,))


@app.put("/api/cameras/{camera_id}", response_model=Camera)
def update_camera(camera_id: int, payload: CameraUpdate) -> Any:
    cam = fetch_one("SELECT * FROM cameras WHERE id = ?", (camera_id,))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    fields = []
    params = []

    if payload.lot_id is not None:
        lot = fetch_one("SELECT * FROM lots WHERE id = ?", (payload.lot_id,))
        if not lot:
            raise HTTPException(status_code=404, detail="Lot not found")
        fields.append("lot_id = ?")
        params.append(payload.lot_id)
    else:
        next_lat = payload.lat if payload.lat is not None else cam.get("lat")
        next_lon = payload.lon if payload.lon is not None else cam.get("lon")
        inferred_lot_id = _find_lot_id_for_point(next_lat, next_lon)
        if inferred_lot_id is not None and inferred_lot_id != cam.get("lot_id"):
            fields.append("lot_id = ?")
            params.append(inferred_lot_id)
    if payload.name is not None:
        fields.append("name = ?")
        params.append(payload.name)
    if payload.video_path is not None:
        fields.append("video_path = ?")
        params.append(payload.video_path)
    if payload.reference_frame_path is not None:
        fields.append("reference_frame_path = ?")
        params.append(payload.reference_frame_path)
    if payload.lat is not None:
        fields.append("lat = ?")
        params.append(payload.lat)
    if payload.lon is not None:
        fields.append("lon = ?")
        params.append(payload.lon)

    if fields:
        params.append(camera_id)
        execute(f"UPDATE cameras SET {', '.join(fields)} WHERE id = ?", tuple(params))

    return fetch_one("SELECT * FROM cameras WHERE id = ?", (camera_id,))




@app.get("/api/cameras/{camera_id}/frame")
def get_camera_frame(camera_id: int, t: float = 0.0) -> Response:
    cam = fetch_one("SELECT * FROM cameras WHERE id = ?", (camera_id,))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    video_path = cam.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found for this camera")

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        raise HTTPException(status_code=500, detail="Video has no readable frames")

    if fps and fps > 0:
        target_frame = int(max(0.0, float(t)) * fps) % frame_count
    else:
        target_frame = int(max(0.0, float(t))) % frame_count

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Failed to read requested frame")

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame image")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/api/cameras/{camera_id}/reference_frame")
def get_reference_frame(camera_id: int) -> FileResponse:
    cam = fetch_one("SELECT * FROM cameras WHERE id = ?", (camera_id,))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    ref = cam.get("reference_frame_path")
    if ref and Path(ref).exists():
        return FileResponse(ref)

    # Auto-generate a reference frame from the recorded video if missing.
    video_path = cam.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found for this camera")

    frames_dir = Path("data/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = frames_dir / f"camera_{camera_id}_reference.jpg"

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video to extract reference frame")

    # Grab a frame ~1 second in (or first frame if unknown FPS)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    target_frame = int(fps) if fps and fps > 0 else 0
    if target_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Failed to read a frame from the video")

    # Write JPEG
    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write reference frame image")

    # Persist the new reference path so the admin UI can load it reliably next time.
    execute("UPDATE cameras SET reference_frame_path = ? WHERE id = ?", (str(out_path), camera_id))

    return FileResponse(out_path)



# Regions
def _max_named_spot_number(camera_id: int, prefix: str) -> int:
    rows = fetch_all(
        "SELECT name FROM regions WHERE camera_id = ? AND kind = 'spot' AND name LIKE ?",
        (camera_id, f"{prefix} %"),
    )
    pattern = re.compile(rf"^{re.escape(prefix)}\s+(\d+)$")
    max_num = 0
    for row in rows:
        name = (row.get("name") or "").strip()
        m = pattern.match(name)
        if not m:
            continue
        max_num = max(max_num, int(m.group(1)))
    return max_num


def _normalize_spot_name(camera_id: int, spot_type: str, raw_name: str) -> str:
    name = (raw_name or "").strip()
    if name:
        return name
    prefix = "Reserved" if spot_type == "reserved" else "Spot"
    next_num = _max_named_spot_number(camera_id, prefix) + 1
    return f"{prefix} {next_num}"


def _validate_bulk_spot_payload(payload: BulkSpotCreate) -> None:
    cam = fetch_one("SELECT * FROM cameras WHERE id = ?", (payload.camera_id,))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    if len(payload.points) != 4:
        raise HTTPException(status_code=422, detail="Bulk spot creation requires exactly 4 zone corner points")


def _validate_region_payload(payload: RegionCreate) -> None:
    cam = fetch_one("SELECT * FROM cameras WHERE id = ?", (payload.camera_id,))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if payload.kind == "zone":
        if payload.capacity <= 0:
            raise HTTPException(status_code=422, detail="Zone capacity must be >= 1")
        if payload.grid_cols is not None and payload.grid_cols <= 0:
            raise HTTPException(status_code=422, detail="Zone grid_cols must be >= 1")
        if len(payload.points) != 4:
            raise HTTPException(status_code=422, detail="Zone must be defined by exactly 4 corner points (rectangle/quad)")
    elif len(payload.points) < 3:
        raise HTTPException(status_code=422, detail="Spot regions require at least 3 points")


@app.post("/api/regions", response_model=Region)
def create_region(payload: RegionCreate) -> Any:
    _validate_region_payload(payload)

    points_json = json.dumps(payload.points)

    # For spots, force defaults
    capacity = int(payload.capacity) if payload.kind == "zone" else 1
    grid_cols = int(payload.grid_cols) if (payload.kind == "zone" and payload.grid_cols) else None
    angle_override_deg = float(payload.angle_override_deg) if (payload.kind == "zone" and payload.angle_override_deg is not None) else None

    region_name = payload.name
    if payload.kind == "spot":
        region_name = _normalize_spot_name(payload.camera_id, payload.spot_type, payload.name)

    region_id = execute(
        """
        INSERT INTO regions (camera_id, name, kind, spot_type, capacity, grid_cols, angle_override_deg, points_json, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.camera_id,
            region_name,
            payload.kind,
            payload.spot_type,
            capacity,
            grid_cols,
            angle_override_deg,
            points_json,
            1 if payload.enabled else 0,
        ),
    )

    execute(
        """
        INSERT OR REPLACE INTO region_state (region_id, occupied, count, open_spots, confidence, last_updated)
        VALUES (?, 0, 0, ?, 0.0, ?)
        """,
        (region_id, capacity, now_iso()),
    )

    return _region_by_id(region_id)


@app.get("/api/regions", response_model=List[Region])
def list_regions(camera_id: Optional[int] = None) -> Any:
    q = "SELECT * FROM regions"
    params = ()
    if camera_id is not None:
        q += " WHERE camera_id = ?"
        params = (camera_id,)
    q += " ORDER BY id ASC"
    rows = fetch_all(q, params)
    return [_region_row_to_model(r) for r in rows]




@app.put("/api/regions/{region_id}", response_model=Region)
def update_region(region_id: int, payload: RegionCreate) -> Any:
    row = fetch_one("SELECT * FROM regions WHERE id = ?", (region_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Region not found")

    _validate_region_payload(payload)

    points_json = json.dumps(payload.points)
    capacity = int(payload.capacity) if payload.kind == "zone" else 1
    grid_cols = int(payload.grid_cols) if (payload.kind == "zone" and payload.grid_cols) else None
    angle_override_deg = float(payload.angle_override_deg) if (payload.kind == "zone" and payload.angle_override_deg is not None) else None

    region_name = payload.name
    if payload.kind == "spot":
        region_name = _normalize_spot_name(payload.camera_id, payload.spot_type, payload.name)

    execute(
        """
        UPDATE regions
        SET camera_id = ?, name = ?, kind = ?, spot_type = ?, capacity = ?, grid_cols = ?,
            angle_override_deg = ?, points_json = ?, enabled = ?
        WHERE id = ?
        """,
        (
            payload.camera_id,
            region_name,
            payload.kind,
            payload.spot_type,
            capacity,
            grid_cols,
            angle_override_deg,
            points_json,
            1 if payload.enabled else 0,
            region_id,
        ),
    )

    execute(
        """
        INSERT OR REPLACE INTO region_state (region_id, occupied, count, open_spots, confidence, last_updated)
        VALUES (?, 0, 0, ?, 0.0, ?)
        """,
        (region_id, capacity, now_iso()),
    )

    if payload.kind != "zone":
        execute("DELETE FROM zone_virtual_state WHERE region_id = ?", (region_id,))

    return _region_by_id(region_id)
@app.post("/api/regions/bulk_spots")
def create_bulk_spots(payload: BulkSpotCreate) -> Dict[str, Any]:
    _validate_bulk_spot_payload(payload)

    grid_cols = int(payload.grid_cols) if payload.grid_cols else 10
    spots = _generate_virtual_spots_from_quad(payload.points, int(payload.capacity), grid_cols)

    prefix = "Reserved" if payload.spot_type == "reserved" else "Spot"
    start_num = _max_named_spot_number(payload.camera_id, prefix) + 1
    created: List[Dict[str, Any]] = []

    for idx, spot in enumerate(spots):
        name = f"{prefix} {start_num + idx}"
        points_json = json.dumps([[float(px), float(py)] for (px, py) in spot["poly"]])
        region_id = execute(
            """
            INSERT INTO regions (camera_id, name, kind, spot_type, capacity, grid_cols, angle_override_deg, points_json, enabled)
            VALUES (?, ?, 'spot', ?, 1, NULL, NULL, ?, ?)
            """,
            (
                payload.camera_id,
                name,
                payload.spot_type,
                points_json,
                1 if payload.enabled else 0,
            ),
        )
        execute(
            """
            INSERT OR REPLACE INTO region_state (region_id, occupied, count, open_spots, confidence, last_updated)
            VALUES (?, 0, 0, 1, 0.0, ?)
            """,
            (region_id, now_iso()),
        )
        created.append(_region_by_id(region_id))

    return {"created_count": len(created), "regions": created}


@app.delete("/api/regions/{region_id}")
def delete_region(region_id: int) -> Dict[str, str]:
    execute("DELETE FROM zone_virtual_state WHERE region_id = ?", (region_id,))
    execute("DELETE FROM region_state WHERE region_id = ?", (region_id,))
    execute("DELETE FROM regions WHERE id = ?", (region_id,))
    return {"status": "deleted"}


def _region_by_id(region_id: int) -> Dict[str, Any]:
    row = fetch_one("SELECT * FROM regions WHERE id = ?", (region_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Region not found")
    return _region_row_to_model(row)


def _region_row_to_model(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "name": row["name"],
        "kind": row["kind"],
        "spot_type": row["spot_type"],
        "capacity": row["capacity"],
        "grid_cols": row.get("grid_cols"),
        "angle_override_deg": row.get("angle_override_deg"),
        "points": json.loads(row["points_json"]),
        "enabled": bool(row["enabled"]),
    }


@app.get("/api/debug/cameras/{camera_id}")
def get_camera_debug(camera_id: int) -> Dict[str, Any]:
    cam = fetch_one("SELECT * FROM cameras WHERE id = ?", (camera_id,))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    debug = fetch_one("SELECT * FROM cv_debug_state WHERE camera_id = ?", (camera_id,))
    if not debug:
        return {
            "camera_id": camera_id,
            "camera_name": cam["name"],
            "video_path": cam["video_path"],
            "vehicles": [],
            "vehicle_boxes": [],
            "detector_meta": {},
            "regions": [],
            "last_updated": "",
        }

    return {
        "camera_id": camera_id,
        "camera_name": cam["name"],
        "video_path": cam["video_path"],
        "vehicles": json.loads(debug["vehicles_json"]),
        "vehicle_boxes": json.loads(debug.get("vehicle_boxes_json") or "[]"),
        "detector_meta": json.loads(debug.get("detector_meta_json") or "{}"),
        "regions": json.loads(debug["regions_json"]),
        "last_updated": debug["last_updated"],
    }


# Availability (counts only) - for filtering + summary
@app.get("/api/availability")
def get_availability(
    include_types: str = "standard,handicap,metered,standard_zone,EV,reserved",
    min_open: int = 0,
    ignore_metered: bool = False,
    ignore_handicap: bool = False,
) -> Dict[str, Any]:
    types = [t.strip() for t in include_types.split(",") if t.strip()]
    if ignore_metered and "metered" in types:
        types.remove("metered")
    if ignore_handicap and "handicap" in types:
        types.remove("handicap")

    lots = fetch_all("SELECT * FROM lots ORDER BY id ASC")
    out_lots: List[Dict[str, Any]] = []

    for lot in lots:
        lot_id = lot["id"]
        cameras = fetch_all("SELECT * FROM cameras WHERE lot_id = ?", (lot_id,))
        if not cameras:
            continue

        totals: Dict[str, int] = {"spots_total": 0, "spots_open": 0}
        open_by_type: Dict[str, int] = {}
        last_updated = None

        for cam in cameras:
            cam_id = cam["id"]
            rows = fetch_all(
                """
                SELECT r.id as region_id, r.kind, r.spot_type, r.capacity,
                       s.occupied, s.count, s.open_spots, s.last_updated
                FROM regions r
                JOIN region_state s ON s.region_id = r.id
                WHERE r.camera_id = ? AND r.enabled = 1
                """,
                (cam_id,),
            )

            for r in rows:
                if r["spot_type"] not in types:
                    continue

                lu = r["last_updated"]
                if last_updated is None or (lu and lu > last_updated):
                    last_updated = lu

                if r["kind"] == "spot":
                    totals["spots_total"] += 1
                    is_open = 1 if int(r["occupied"]) == 0 else 0
                    totals["spots_open"] += is_open
                    open_by_type[r["spot_type"]] = open_by_type.get(r["spot_type"], 0) + is_open
                elif r["kind"] == "zone":
                    cap = int(r["capacity"])
                    open_spots = int(r["open_spots"])
                    totals["spots_total"] += cap
                    totals["spots_open"] += open_spots
                    open_by_type[r["spot_type"]] = open_by_type.get(r["spot_type"], 0) + open_spots

        if totals["spots_open"] >= int(min_open):
            out_lots.append(
                {
                    "lot_id": lot_id,
                    "lot_name": lot["name"],
                    "lot_group": lot.get("lot_group") or "",
                    "lat": lot["lat"],
                    "lon": lot["lon"],
                    "polygon_geojson": lot.get("polygon_geojson") or "",
                    "last_updated": last_updated or "",
                    "totals": totals,
                    "open_by_type": open_by_type,
                }
            )

    return {"lots": out_lots, "filters": {"types": types, "min_open": int(min_open)}}


# Virtual spots for a zone (for overlay)
@app.get("/api/zones/{region_id}/virtual")
def get_zone_virtual(region_id: int) -> Dict[str, Any]:
    reg = fetch_one("SELECT * FROM regions WHERE id = ?", (region_id,))
    if not reg:
        raise HTTPException(status_code=404, detail="Region not found")
    if reg["kind"] != "zone":
        raise HTTPException(status_code=400, detail="Region is not a zone")

    vs = fetch_one("SELECT * FROM zone_virtual_state WHERE region_id = ?", (region_id,))
    if not vs:
        return {"region_id": region_id, "spots": [], "last_updated": ""}
    return {"region_id": region_id, "spots": json.loads(vs["spots_json"]), "last_updated": vs["last_updated"]}
