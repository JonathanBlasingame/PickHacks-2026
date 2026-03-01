from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Point = List[float]  # [x, y]


class LotCreate(BaseModel):
    name: str
    lot_group: Optional[str] = None
    lat: float
    lon: float
    polygon_geojson: Optional[str] = None


class Lot(LotCreate):
    id: int


class CameraCreate(BaseModel):
    lot_id: int
    name: str
    video_path: str
    reference_frame_path: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class Camera(CameraCreate):
    id: int


class RegionCreate(BaseModel):
    camera_id: int
    name: str
    kind: Literal["spot", "zone"]
    spot_type: str = Field(..., description="e.g., standard, handicap, metered, standard_zone")
    capacity: int = Field(default=1, ge=1)
    grid_cols: Optional[int] = Field(
        default=None,
        ge=1,
        description="Zones only: number of virtual spots across the short axis.",
    )
    angle_override_deg: Optional[float] = Field(default=None, description="Zones only: override inferred angle (degrees).")
    points: List[Point]
    enabled: bool = True


class Region(RegionCreate):
    id: int


class LotUpdate(BaseModel):
    name: Optional[str] = None
    lot_group: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    polygon_geojson: Optional[str] = None


class CameraUpdate(BaseModel):
    lot_id: Optional[int] = None
    name: Optional[str] = None
    video_path: Optional[str] = None
    reference_frame_path: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class BulkSpotCreate(BaseModel):
    camera_id: int
    spot_type: str = Field(default="standard", description="spot type for generated spots")
    capacity: int = Field(default=1, ge=1)
    grid_cols: Optional[int] = Field(default=None, ge=1)
    angle_override_deg: Optional[float] = Field(default=None)
    points: List[Point]
    enabled: bool = True
