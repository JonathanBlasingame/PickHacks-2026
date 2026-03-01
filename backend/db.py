import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path("parking.db")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite schema. Safe to run multiple times."""
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                lot_group TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                polygon_geojson TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                video_path TEXT NOT NULL,
                reference_frame_path TEXT,
                lat REAL,
                lon REAL,
                FOREIGN KEY(lot_id) REFERENCES lots(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL, -- "spot" or "zone"
                spot_type TEXT NOT NULL, -- standard, handicap, metered, EV, reserved, standard_zone, etc
                capacity INTEGER NOT NULL DEFAULT 1,
                grid_cols INTEGER,
                angle_override_deg REAL,
                points_json TEXT NOT NULL, -- JSON list of [x,y]
                enabled INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(camera_id) REFERENCES cameras(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS region_state (
                region_id INTEGER PRIMARY KEY,
                occupied INTEGER NOT NULL DEFAULT 0,  -- for spots
                count INTEGER NOT NULL DEFAULT 0,     -- for zones (occupied_count for virtual-spot assignment)
                open_spots INTEGER NOT NULL DEFAULT 0,-- for zones: capacity - count
                confidence REAL NOT NULL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                FOREIGN KEY(region_id) REFERENCES regions(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS zone_virtual_state (
                region_id INTEGER PRIMARY KEY,
                spots_json TEXT NOT NULL, -- list of {center:[x,y], poly:[[x,y]...], occupied:bool}
                last_updated TEXT NOT NULL,
                FOREIGN KEY(region_id) REFERENCES regions(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cv_debug_state (
                camera_id INTEGER PRIMARY KEY,
                vehicles_json TEXT NOT NULL, -- list of [x,y]
                vehicle_anchors_json TEXT NOT NULL DEFAULT "[]", -- list of [x,y] anchor points
                vehicle_tires_json TEXT NOT NULL DEFAULT "[]", -- list of [x,y] estimated tire points
                vehicle_tire_segments_json TEXT NOT NULL DEFAULT "[]", -- list of [[x1,y1],[x2,y2]] tire lines
                vehicle_boxes_json TEXT NOT NULL DEFAULT "[]", -- list of {xyxy,class_id,class_name,confidence,stationary}
                detector_meta_json TEXT NOT NULL DEFAULT "{}", -- detector config metadata used for debug rendering
                regions_json TEXT NOT NULL, -- list of per-region debug summaries
                last_updated TEXT NOT NULL,
                FOREIGN KEY(camera_id) REFERENCES cameras(id)
            )
            """
        )

        # Lightweight migrations for existing databases.
        lot_cols = {row["name"] for row in conn.execute("PRAGMA table_info(lots)").fetchall()}
        if "lot_group" not in lot_cols:
            conn.execute("ALTER TABLE lots ADD COLUMN lot_group TEXT")

        cols = {row["name"] for row in conn.execute("PRAGMA table_info(cv_debug_state)").fetchall()}
        if "vehicle_anchors_json" not in cols:
            conn.execute(
                "ALTER TABLE cv_debug_state ADD COLUMN vehicle_anchors_json TEXT NOT NULL DEFAULT '[]'"
            )

        if "vehicle_tires_json" not in cols:
            conn.execute(
                "ALTER TABLE cv_debug_state ADD COLUMN vehicle_tires_json TEXT NOT NULL DEFAULT '[]'"
            )

        if "vehicle_tire_segments_json" not in cols:
            conn.execute(
                "ALTER TABLE cv_debug_state ADD COLUMN vehicle_tire_segments_json TEXT NOT NULL DEFAULT '[]'"
            )

        if "vehicle_boxes_json" not in cols:
            conn.execute(
                "ALTER TABLE cv_debug_state ADD COLUMN vehicle_boxes_json TEXT NOT NULL DEFAULT '[]'"
            )

        if "detector_meta_json" not in cols:
            conn.execute(
                "ALTER TABLE cv_debug_state ADD COLUMN detector_meta_json TEXT NOT NULL DEFAULT '{}'"
            )

        conn.commit()


def fetch_all(query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def fetch_one(query: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def execute(query: str, params: Tuple[Any, ...] = ()) -> int:
    with connect() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid
