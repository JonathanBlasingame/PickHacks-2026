from typing import List, Tuple

import math
import numpy as np


def footprint_points_from_boxes(boxes_xyxy: np.ndarray, split_ratio: float = 0.24, y_bias: float = 0.96) -> List[Tuple[float, float]]:
    """Estimate a stable ground-contact point per vehicle box.

    Uses a point near the bottom of the box and slightly offset from center
    toward the side with visible tire contact. This improves spot assignment in
    perspective-heavy scenes compared with pure center anchors.
    """
    points: List[Tuple[float, float]] = []
    clamped_split = min(max(float(split_ratio), 0.0), 0.49)
    clamped_y = min(max(float(y_bias), 0.0), 1.0)

    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = float((x1 + x2) / 2.0)
        # heuristic: nearer camera rows tend to have stronger right-side visibility
        side_offset = w * clamped_split
        fx = cx + side_offset
        fy = float(y1 + h * clamped_y)
        points.append((fx, fy))

    return points


def tire_points_and_segments_from_boxes(
    boxes_xyxy: np.ndarray,
    y_bias: float = 0.76,
    x_margin_ratio: float = 0.16,
    frame_width: float | None = None,
    side_x_offset_ratio: float = 0.15,
    perspective_tilt_max_deg: float = 18.0,
    segment_axis: str = "length",
    min_aspect_ratio: float = 1.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Estimate a visible tire-pair segment per vehicle box.

    Segment direction follows the vehicle's dominant image axis and includes a
    small perspective-driven tilt so results are not constrained to exact
    horizontal/vertical lines.
    """
    tire_points: List[Tuple[float, float]] = []
    tire_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    clamped_y = min(max(float(y_bias), 0.05), 0.95)
    clamped_margin = min(max(float(x_margin_ratio), 0.05), 0.40)
    clamped_side_offset = min(max(float(side_x_offset_ratio), 0.0), 0.3)
    clamped_tilt_deg = min(max(float(perspective_tilt_max_deg), 0.0), 35.0)
    clamped_min_aspect = max(1.0, float(min_aspect_ratio))
    normalized_axis = str(segment_axis).strip().lower()
    if normalized_axis not in ("width", "length"):
        normalized_axis = "length"

    use_frame = frame_width is not None and float(frame_width) > 1.0
    frame_mid_x = (float(frame_width) / 2.0) if use_frame else 0.0

    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        x_min = float(x1 + w * clamped_margin)
        x_max = float(x2 - w * clamped_margin)
        y_min = float(y1 + h * clamped_margin)
        y_max = float(y2 - h * clamped_margin)

        side_sign = 0.0
        if use_frame:
            side_sign = (cx - frame_mid_x) / frame_mid_x
            side_sign = min(max(side_sign, -1.0), 1.0)

        long_axis_is_horizontal = w >= h
        use_horizontal_axis = long_axis_is_horizontal if normalized_axis == "length" else not long_axis_is_horizontal

        seg_center_x = float(cx - side_sign * w * clamped_side_offset)
        seg_center_y = float(y1 + h * clamped_y)
        base_angle_deg = 0.0 if use_horizontal_axis else 90.0
        seg_len = max(2.0, (x_max - x_min) if use_horizontal_axis else (y_max - y_min))

        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < clamped_min_aspect:
            # Near-square boxes have unstable major-axis flips frame to frame.
            # Keep a short segment to avoid bleeding into adjacent empty spots.
            seg_len = max(2.0, min(w, h) * 0.40)
            clamped_tilt_local = 0.0
        else:
            clamped_tilt_local = clamped_tilt_deg

        theta = math.radians(base_angle_deg + side_sign * clamped_tilt_local)
        dx = math.cos(theta) * (seg_len * 0.5)
        dy = math.sin(theta) * (seg_len * 0.5)

        p1x = min(max(seg_center_x - dx, x_min), x_max)
        p1y = min(max(seg_center_y - dy, y_min), y_max)
        p2x = min(max(seg_center_x + dx, x_min), x_max)
        p2y = min(max(seg_center_y + dy, y_min), y_max)

        p1 = (float(p1x), float(p1y))
        p2 = (float(p2x), float(p2y))
        tire_points.extend([p1, p2])
        tire_segments.append((p1, p2))

    return tire_points, tire_segments
