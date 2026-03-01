import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ApiAndClassifierTests(unittest.TestCase):
    def _fresh_app(self, tmp_path: Path):
        import backend.db as db

        db.DB_PATH = tmp_path / "test.db"

        import backend.app as app_module

        importlib.reload(app_module)
        app_module.init_db()
        return app_module

    def test_create_region_rejects_invalid_zone_values(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI test client unavailable: {exc}")

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            app_module = self._fresh_app(Path(td))
            client = TestClient(app_module.app)

            lot = client.post(
                "/api/lots",
                json={"name": "Lot", "lat": 1.0, "lon": 2.0, "polygon_geojson": ""},
            ).json()
            cam = client.post(
                "/api/cameras",
                json={"lot_id": lot["id"], "name": "Cam", "video_path": "dummy.mp4"},
            ).json()

            bad_payloads = [
                {"capacity": 0, "grid_cols": 1},
                {"capacity": -1, "grid_cols": 1},
                {"capacity": 10, "grid_cols": 0},
                {"capacity": 10, "grid_cols": -2},
            ]

            for bad in bad_payloads:
                payload = {
                    "camera_id": cam["id"],
                    "name": "Zone",
                    "kind": "zone",
                    "spot_type": "standard_zone",
                    "capacity": bad["capacity"],
                    "grid_cols": bad["grid_cols"],
                    "points": [[0, 0], [100, 0], [100, 100]],
                    "enabled": True,
                }
                resp = client.post("/api/regions", json=payload)
                self.assertEqual(resp.status_code, 422)


    def test_create_zone_requires_exactly_four_corner_points(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI test client unavailable: {exc}")

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            app_module = self._fresh_app(Path(td))
            client = TestClient(app_module.app)

            lot = client.post(
                "/api/lots",
                json={"name": "Lot", "lat": 1.0, "lon": 2.0, "polygon_geojson": ""},
            ).json()
            cam = client.post(
                "/api/cameras",
                json={"lot_id": lot["id"], "name": "Cam", "video_path": "dummy.mp4"},
            ).json()

            payload = {
                "camera_id": cam["id"],
                "name": "Zone",
                "kind": "zone",
                "spot_type": "standard_zone",
                "capacity": 10,
                "grid_cols": 2,
                "points": [[0, 0], [100, 0], [100, 100]],
                "enabled": True,
            }
            resp = client.post("/api/regions", json=payload)
            self.assertEqual(resp.status_code, 422)

    def test_get_camera_frame_returns_jpeg_bytes_without_disk_cache_file(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI test client unavailable: {exc}")

        try:
            cv2 = importlib.import_module("cv2")
        except Exception as exc:
            self.skipTest(f"OpenCV unavailable in environment: {exc}")
        import numpy as np
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            app_module = self._fresh_app(Path(td))
            client = TestClient(app_module.app)

            video_path = Path(td) / "tiny.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 2.0, (32, 24))
            if not writer.isOpened():
                self.skipTest("OpenCV video writer unavailable in environment")
            for _ in range(4):
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            lot = client.post(
                "/api/lots",
                json={"name": "Lot", "lat": 1.0, "lon": 2.0, "polygon_geojson": ""},
            ).json()
            cam = client.post(
                "/api/cameras",
                json={"lot_id": lot["id"], "name": "Cam", "video_path": str(video_path)},
            ).json()

            resp = client.get(f"/api/cameras/{cam['id']}/frame?t=0")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.headers["content-type"].startswith("image/jpeg"))
            self.assertTrue(resp.headers.get("cache-control"))
            self.assertEqual(resp.content[:2], b"\xff\xd8")

    def test_malexnet_outputs_logits_not_softmax_distribution(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        try:
            from cv.slot_classifier import _mAlexNet
        except Exception as exc:
            self.skipTest(f"slot classifier dependencies unavailable: {exc}")

        model = _mAlexNet(num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 2))
        probs_sum = float(out.sum(dim=1).item())
        self.assertGreater(abs(probs_sum - 1.0), 1e-3)


class TireSegmentGeometryTests(unittest.TestCase):
    def test_tire_segments_can_follow_configured_width_axis(self):
        try:
            import numpy as np
            from cv.footprint import tire_points_and_segments_from_boxes
        except Exception as exc:
            self.skipTest(f"footprint dependencies unavailable: {exc}")

        boxes = np.array([[0.0, 0.0, 100.0, 40.0]], dtype=np.float32)
        _, seg_length = tire_points_and_segments_from_boxes(boxes, segment_axis="length", perspective_tilt_max_deg=0.0)
        _, seg_width = tire_points_and_segments_from_boxes(boxes, segment_axis="width", perspective_tilt_max_deg=0.0)

        (lx1, ly1), (lx2, ly2) = seg_length[0]
        (wx1, wy1), (wx2, wy2) = seg_width[0]
        self.assertGreater(abs(lx2 - lx1), abs(ly2 - ly1))
        self.assertGreater(abs(wy2 - wy1), abs(wx2 - wx1))

    def test_tire_segments_shorten_for_near_square_boxes(self):
        try:
            import numpy as np
            from cv.footprint import tire_points_and_segments_from_boxes
        except Exception as exc:
            self.skipTest(f"footprint dependencies unavailable: {exc}")

        boxes = np.array([[10.0, 10.0, 62.0, 56.0]], dtype=np.float32)
        _, seg_default = tire_points_and_segments_from_boxes(
            boxes,
            segment_axis="length",
            min_aspect_ratio=1.0,
            perspective_tilt_max_deg=0.0,
        )
        _, seg_guarded = tire_points_and_segments_from_boxes(
            boxes,
            segment_axis="length",
            min_aspect_ratio=1.4,
            perspective_tilt_max_deg=0.0,
        )

        (d1x, d1y), (d2x, d2y) = seg_default[0]
        (g1x, g1y), (g2x, g2y) = seg_guarded[0]
        default_len = ((d2x - d1x) ** 2 + (d2y - d1y) ** 2) ** 0.5
        guarded_len = ((g2x - g1x) ** 2 + (g2y - g1y) ** 2) ** 0.5
        self.assertLess(guarded_len, default_len)



if __name__ == "__main__":
    unittest.main()
