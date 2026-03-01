from backend.db import init_db, execute, fetch_one


def seed() -> None:
    init_db()

    # Demo lot location (roughly Rolla, MO)
    lot = fetch_one("SELECT * FROM lots WHERE name = ?", ("Demo Lot",))
    if not lot:
        lot_id = execute(
            "INSERT INTO lots (name, lat, lon, polygon_geojson) VALUES (?, ?, ?, ?)",
            ("Demo Lot", 37.9514, -91.7713, None),
        )
    else:
        lot_id = lot["id"]

    cam = fetch_one("SELECT * FROM cameras WHERE name = ?", ("Lot Cam 1",))
    if not cam:
        execute(
            "INSERT INTO cameras (lot_id, name, video_path, reference_frame_path, lat, lon) VALUES (?, ?, ?, ?, ?, ?)",
            (
                lot_id,
                "Lot Cam 1",
                "data/videos/lot1.mp4",
                "data/frames/lot1_reference.jpg",
                37.9514,
                -91.7713,
            ),
        )


if __name__ == "__main__":
    seed()
    print("Seed complete.")
