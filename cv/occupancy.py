from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Hashable

import cv2
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon


@dataclass
class RegionDef:
    id: int
    name: str
    kind: str          # "spot" or "zone"
    spot_type: str     # "handicap", "metered", "standard_zone", etc.
    capacity: int
    grid_cols: Optional[int]
    angle_override_deg: Optional[float]
    points: List[Tuple[float, float]]


class Smoother:
    """Majority-vote smoothing for spot occupancy with delayed clearing.

    `empty_hold_seconds` keeps a spot occupied for a short duration after the
    vote flips to empty, which helps avoid false clears during brief occlusions.
    """

    def __init__(self, window: int = 5, threshold: int = 3, empty_hold_seconds: float = 0.0):
        self.window = window
        self.threshold = threshold
        self.empty_hold_seconds = max(0.0, float(empty_hold_seconds))
        self.buffers: Dict[int, List[bool]] = {}
        self.latched_state: Dict[int, bool] = {}
        self.pending_empty_seconds: Dict[int, float] = {}

    def update(self, region_id: int, value: bool, dt_seconds: float = 0.0) -> bool:
        buf = self.buffers.setdefault(region_id, [])
        buf.append(value)
        if len(buf) > self.window:
            buf.pop(0)

        voted_occupied = sum(1 for v in buf if v) >= self.threshold
        latched_occupied = self.latched_state.get(region_id, False)

        if voted_occupied:
            self.latched_state[region_id] = True
            self.pending_empty_seconds[region_id] = 0.0
            return True

        if latched_occupied and self.empty_hold_seconds > 0:
            pending = self.pending_empty_seconds.get(region_id, 0.0) + max(0.0, float(dt_seconds))
            self.pending_empty_seconds[region_id] = pending
            if pending <= self.empty_hold_seconds:
                return True

        self.latched_state[region_id] = False
        self.pending_empty_seconds[region_id] = 0.0
        return False


class StationaryVehicleFilter:
    """Tracks per-vehicle motion and marks detections as stationary once stabilized."""

    def __init__(
        self,
        min_stationary_seconds: float = 1.0,
        max_motion_px: float = 12.0,
        match_max_dist_px: float = 80.0,
        max_missing_seconds: float = 1.5,
        persist_seconds: float = 0.75,
    ):
        self.min_stationary_seconds = max(0.05, float(min_stationary_seconds))
        self.max_motion_px = max(0.0, float(max_motion_px))
        self.match_max_dist_px = max(1.0, float(match_max_dist_px))
        self.max_missing_seconds = max(0.05, float(max_missing_seconds))
        self.persist_seconds = min(self.max_missing_seconds, max(0.0, float(persist_seconds)))
        self._next_track_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        centers: List[Tuple[float, float]],
        boxes_xyxy: Optional[List[np.ndarray]] = None,
        dt_seconds: float = 0.0,
    ) -> List[bool]:
        """Returns a stationary flag for each center in `centers` order."""
        delta = max(0.0, float(dt_seconds))
        stationary_flags = [False] * len(centers)
        boxes = boxes_xyxy or []
        unmatched_track_ids = set(self._tracks.keys())

        candidates: List[Tuple[float, int, int]] = []
        max_d2 = self.match_max_dist_px * self.match_max_dist_px
        for det_i, (cx, cy) in enumerate(centers):
            for track_id, tr in self._tracks.items():
                tx, ty = tr["center"]
                dx = cx - tx
                dy = cy - ty
                d2 = dx * dx + dy * dy
                if d2 <= max_d2:
                    candidates.append((d2, det_i, track_id))

        det_to_track: Dict[int, int] = {}
        used_dets = set()
        used_tracks = set()
        for _, det_i, track_id in sorted(candidates, key=lambda x: x[0]):
            if det_i in used_dets or track_id in used_tracks:
                continue
            used_dets.add(det_i)
            used_tracks.add(track_id)
            det_to_track[det_i] = track_id
            unmatched_track_ids.discard(track_id)

        for det_i, (cx, cy) in enumerate(centers):
            if det_i in det_to_track:
                track_id = det_to_track[det_i]
                tr = self._tracks[track_id]
                px, py = tr["center"]
                motion_px = math.hypot(cx - px, cy - py)
                if motion_px <= self.max_motion_px:
                    tr["still_seconds"] += delta
                else:
                    tr["still_seconds"] = 0.0
                tr["center"] = (cx, cy)
                if det_i < len(boxes):
                    tr["box_xyxy"] = tuple(float(v) for v in boxes[det_i].tolist())
                tr["missing_seconds"] = 0.0
                tr["is_stationary"] = tr["still_seconds"] >= self.min_stationary_seconds
                stationary_flags[det_i] = tr["is_stationary"]
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = {
                    "center": (cx, cy),
                    "box_xyxy": tuple(float(v) for v in boxes[det_i].tolist()) if det_i < len(boxes) else None,
                    "still_seconds": 0.0,
                    "missing_seconds": 0.0,
                    "is_stationary": False,
                }

        for track_id in unmatched_track_ids:
            tr = self._tracks[track_id]
            tr["missing_seconds"] += delta

        expired = [tid for tid, tr in self._tracks.items() if tr["missing_seconds"] > self.max_missing_seconds]
        for tid in expired:
            del self._tracks[tid]

        return stationary_flags

    def get_persisted_stationary_observations(self) -> List[Dict[str, Any]]:
        """Returns stationary tracks currently held through the grace window."""
        held: List[Dict[str, Any]] = []
        if self.persist_seconds <= 0.0:
            return held
        for track_id, tr in self._tracks.items():
            missing_seconds = float(tr.get("missing_seconds", 0.0))
            if missing_seconds <= 0.0 or missing_seconds > self.persist_seconds:
                continue
            if not bool(tr.get("is_stationary", False)):
                continue
            box_xyxy = tr.get("box_xyxy")
            if box_xyxy is None:
                continue
            held.append(
                {
                    "track_id": int(track_id),
                    "center": tuple(float(v) for v in tr["center"]),
                    "box_xyxy": tuple(float(v) for v in box_xyxy),
                    "missing_seconds": missing_seconds,
                }
            )
        return held


class HysteresisFilter:
    """State filter with asymmetric enter/exit confirmation windows.

    Useful for suppressing occupancy flicker from brief detector dropouts and
    transient false positives.
    """

    def __init__(self, min_enter_seconds: float = 0.0, min_exit_seconds: float = 0.0):
        self.min_enter_seconds = max(0.0, float(min_enter_seconds))
        self.min_exit_seconds = max(0.0, float(min_exit_seconds))
        self._state: Dict[Hashable, bool] = {}
        self._enter_seconds: Dict[Hashable, float] = {}
        self._exit_seconds: Dict[Hashable, float] = {}

    def update(self, key: Hashable, observed_occupied: bool, dt_seconds: float = 0.0) -> bool:
        delta = max(0.0, float(dt_seconds))
        if observed_occupied:
            self._exit_seconds[key] = 0.0
            self._enter_seconds[key] = self._enter_seconds.get(key, 0.0) + delta
            if self._enter_seconds[key] >= self.min_enter_seconds:
                self._state[key] = True
        else:
            self._enter_seconds[key] = 0.0
            self._exit_seconds[key] = self._exit_seconds.get(key, 0.0) + delta
            if self._exit_seconds[key] >= self.min_exit_seconds:
                self._state[key] = False

        return self._state.get(key, False)


def point_in_poly(x: float, y: float, poly_pts: List[Tuple[float, float]]) -> bool:
    poly = Polygon(poly_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return False
    return bool(poly.covers(Point(x, y)))


def centers_from_boxes(boxes_xyxy: np.ndarray) -> List[Tuple[float, float]]:
    centers = []
    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append((float(cx), float(cy)))
    return centers


def anchor_points_from_boxes(boxes_xyxy: np.ndarray, y_bias: float = 0.9) -> List[Tuple[float, float]]:
    """Return per-box anchor points near each vehicle's tire contact patch.

    The y-bias defaults near the bottom edge of the detection to reduce adjacent-spot bleed.
    """
    anchors = []
    clamped_bias = min(max(float(y_bias), 0.0), 1.0)
    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        cx = (x1 + x2) / 2.0
        ay = y1 + (y2 - y1) * clamped_bias
        anchors.append((float(cx), float(ay)))
    return anchors


def box_xyxy_to_polygon(box_xyxy: np.ndarray) -> Polygon:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def spot_box_overlap_ratio(
    spot_points: List[Tuple[float, float]],
    box_xyxy: np.ndarray,
    mode: str = "intersection_over_spot",
) -> float:
    """Returns overlap score between a spot polygon and a detected vehicle box polygon."""
    spot_poly = Polygon(spot_points)
    if not spot_poly.is_valid:
        spot_poly = spot_poly.buffer(0)
    if spot_poly.is_empty or spot_poly.area <= 0:
        return 0.0

    box_poly = box_xyxy_to_polygon(box_xyxy)
    if not box_poly.is_valid:
        box_poly = box_poly.buffer(0)
    if box_poly.is_empty or box_poly.area <= 0:
        return 0.0

    inter_area = spot_poly.intersection(box_poly).area
    if inter_area <= 0:
        return 0.0

    if mode == "iou":
        union_area = spot_poly.union(box_poly).area
        return float(inter_area / union_area) if union_area > 0 else 0.0

    return float(inter_area / spot_poly.area)


def infer_min_area_rect(zone_points: List[Tuple[float, float]]) -> Tuple[float, float, float, float, float]:
    pts = np.array(zone_points, dtype=np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h), angle)
    (cx, cy), (w, h), angle = rect
    return float(cx), float(cy), float(w), float(h), float(angle)




def _order_quad_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(points, dtype=np.float32)
    if pts.shape[0] != 4:
        return [(float(x), float(y)) for x, y in pts.tolist()]

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(d))]
    bl = pts[int(np.argmax(d))]
    return [
        (float(tl[0]), float(tl[1])),
        (float(tr[0]), float(tr[1])),
        (float(br[0]), float(br[1])),
        (float(bl[0]), float(bl[1])),
    ]


def _lerp(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _quad_point(
    tl: Tuple[float, float],
    tr: Tuple[float, float],
    br: Tuple[float, float],
    bl: Tuple[float, float],
    u: float,
    v: float,
) -> Tuple[float, float]:
    """Bilinear interpolation inside a quadrilateral."""
    top = _lerp(tl, tr, u)
    bottom = _lerp(bl, br, u)
    return _lerp(top, bottom, v)


def generate_virtual_spots(
    zone_points: List[Tuple[float, float]],
    capacity: int,
    grid_cols: int,
    angle_override_deg: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Generates 'capacity' virtual parking spots inside a zone.

    Preferred path: when zone has exactly 4 points, treat it as a perspective-aware
    quadrilateral and generate a warped grid via bilinear interpolation.

    Backward-compatible path: for legacy non-4-point zones, fall back to min-area-rect
    generation.
    """
    cols = max(1, int(grid_cols))
    rows = max(1, math.ceil(capacity / cols))

    if len(zone_points) == 4:
        tl, tr, br, bl = _order_quad_points(zone_points)
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

                spots.append(
                    {
                        "center": (float(center[0]), float(center[1])),
                        "poly": [
                            (float(p_tl[0]), float(p_tl[1])),
                            (float(p_tr[0]), float(p_tr[1])),
                            (float(p_br[0]), float(p_br[1])),
                            (float(p_bl[0]), float(p_bl[1])),
                        ],
                    }
                )
        return spots[:capacity]

    # Legacy fallback for older polygon zones
    poly = Polygon(zone_points)
    cx, cy, w, h, angle = infer_min_area_rect(zone_points)

    long_side = max(w, h)
    short_side = min(w, h)

    base_angle_deg = angle
    if w < h:
        base_angle_deg = angle + 90.0

    if angle_override_deg is not None:
        base_angle_deg = angle_override_deg

    theta = math.radians(base_angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)

    cell_w = long_side / cols
    cell_h = short_side / rows

    candidates: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            ou = (-long_side / 2.0) + (c + 0.5) * cell_w
            ov = (-short_side / 2.0) + (r + 0.5) * cell_h

            x = cx + ou * ux + ov * vx
            y = cy + ou * uy + ov * vy

            half_w = cell_w / 2.0
            half_h = cell_h / 2.0
            corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            world = []
            for du, dv in corners:
                wx = x + du * ux + dv * vx
                wy = y + du * uy + dv * vy
                world.append((float(wx), float(wy)))

            candidates.append({"center": (float(x), float(y)), "poly": world})

    inside = [c for c in candidates if poly.contains(Point(*c["center"]))]
    if len(inside) >= capacity:
        return inside[:capacity]

    buffered = poly.buffer(6.0)
    inside2 = [c for c in candidates if buffered.contains(Point(*c["center"]))]
    return inside2[:capacity]


def assign_cars_to_spots(
    car_centers: List[Tuple[float, float]],
    spots: List[Dict[str, Any]],
    max_dist_px: float = 80.0,
) -> List[bool]:
    """Greedy car→spot assignment using center-in-spot first.

    Each car can occupy at most one spot and each spot can be occupied by at most one car.
    The primary rule is polygon intersection (car center inside a virtual spot polygon).
    If a spot has no polygon data, distance-to-center is used as fallback.
    """
    occupied = [False] * len(spots)
    if not spots or not car_centers:
        return occupied

    max_d2 = max_dist_px * max_dist_px

    for (cx, cy) in car_centers:
        candidates: List[Tuple[float, int]] = []
        for i, spot in enumerate(spots):
            if occupied[i]:
                continue

            poly = spot.get("poly")
            if poly and point_in_poly(cx, cy, poly):
                sx, sy = spot["center"]
                dx = cx - sx
                dy = cy - sy
                candidates.append((dx * dx + dy * dy, i))
                continue

            if not poly:
                sx, sy = spot["center"]
                dx = cx - sx
                dy = cy - sy
                d2 = dx * dx + dy * dy
                if d2 <= max_d2:
                    candidates.append((d2, i))

        if candidates:
            _, best_i = min(candidates, key=lambda x: x[0])
            occupied[best_i] = True

    return occupied


def assign_boxes_to_spots_by_overlap(
    car_boxes: List[np.ndarray],
    spots: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.08,
) -> List[bool]:
    """Assign car boxes to virtual spots using maximum polygon overlap.

    Uses greedy one-to-one matching by descending overlap so partially visible or
    perspective-skewed vehicles can still mark a spot occupied when anchor-point
    distance would miss.
    """
    occupied = [False] * len(spots)
    if not car_boxes or not spots:
        return occupied

    candidates: List[Tuple[float, int, int]] = []
    for spot_i, spot in enumerate(spots):
        spot_poly = spot.get("poly")
        if not spot_poly:
            continue
        for box_i, box in enumerate(car_boxes):
            overlap = spot_box_overlap_ratio(spot_poly, box, mode="intersection_over_spot")
            if overlap >= min_overlap_ratio:
                candidates.append((float(overlap), spot_i, box_i))

    used_spots = set()
    used_boxes = set()
    for _, spot_i, box_i in sorted(candidates, key=lambda x: x[0], reverse=True):
        if spot_i in used_spots or box_i in used_boxes:
            continue
        used_spots.add(spot_i)
        used_boxes.add(box_i)
        occupied[spot_i] = True

    return occupied


def segment_intersects_spot(
    spot_points: List[Tuple[float, float]],
    segment: Tuple[Tuple[float, float], Tuple[float, float]],
) -> bool:
    """Returns True when a tire-line segment crosses or lies within a spot polygon."""
    spot_poly = Polygon(spot_points)
    if not spot_poly.is_valid:
        spot_poly = spot_poly.buffer(0)
    if spot_poly.is_empty:
        return False

    line = LineString([segment[0], segment[1]])
    if not line.is_valid or line.length <= 0:
        return False

    return bool(spot_poly.intersects(line))


def assign_segments_to_spots_by_intersection(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    spots: List[Dict[str, Any]],
) -> List[bool]:
    """Assign occupancy when tire-line segments intersect spot polygons."""
    occupied = [False] * len(spots)
    if not segments or not spots:
        return occupied

    for i, spot in enumerate(spots):
        spot_poly = spot.get("poly")
        if not spot_poly:
            continue
        occupied[i] = any(segment_intersects_spot(spot_poly, seg) for seg in segments)
    return occupied
