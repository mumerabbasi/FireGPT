# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional

class BoundingBox(BaseModel):
    """
    Represents the bounding box for the area of interest.
    """
    top_left_lat: float = Field(..., description="Latitude of the top-left corner.")
    top_left_lon: float = Field(..., description="Longitude of the top-left corner.")
    bottom_right_lat: float = Field(..., description="Latitude of the bottom-right corner.")
    bottom_right_lon: float = Field(..., description="Longitude of the bottom-right corner.")

class FireDangerRequest(BaseModel):
    """
    Request model for the fire danger assessment API.
    """
    bbox: BoundingBox
    subgrid_size_m: int = Field(100, ge=10, description="Size of each subgrid cell in meters (e.g., 100 for 100x100m). Minimum 10m.")
    forecast_hours: int = Field(3, ge=0, description="Number of hours into the future for the GFS forecast. Minimum 0 hours.")
    # --- NEW PARAMETER ---
    poi_search_buffer_m: int = Field(5000, ge=0, description="Buffer distance in meters outside the main bounding box to search for Points of Interest. A value of 0 means search only within the bbox.")

class SubgridProperties(BaseModel):
    """
    Properties extracted for each subgrid.
    """
    jrc_forest_cover_presence_10m: Optional[int] = None
    hansen_treecover2000_perc: Optional[float] = None
    corine_land_cover_code: Optional[int] = None
    corine_land_cover_description: Optional[str] = None
    hansen_lossyear: Optional[int] = None
    u_wind_ms: Optional[float] = None
    v_wind_ms: Optional[float] = None
    temperature_k: Optional[float] = None
    altitude_m: Optional[float] = None

class FireDangerResult(BaseModel):
    """
    Fire danger assessment for a subgrid.
    """
    score: int = Field(..., ge=0, le=100)
    factors: List[str]

class SubgridData(BaseModel):
    """
    Data for a single subgrid including its location and assessment.
    """
    row: int
    col: int
    center_lat: float
    center_lon: float
    properties: SubgridProperties
    fire_danger: FireDangerResult

# --- POI MODELS (No changes here, just for context) ---
class ClosestSubgridInfo(BaseModel):
    """
    Information about the closest subgrid to a POI.
    """
    subgrid_id: str
    distance_m: float

class POIData(BaseModel):
    """
    Details for a single Point of Interest.
    """
    osm_id: int
    name: Optional[str] = None
    type: str # e.g., 'hospital', 'school', 'shelter'
    subtype: Optional[str] = None # e.g., 'community_centre', 'fire_station'
    lat: float
    lon: float
    is_within_bbox: bool
    closest_subgrid: Optional[ClosestSubgridInfo] = None # Closest subgrid from the assessment
    distance_from_bbox_center_m: Optional[float] = None # General distance from the center of the main query bbox

class FireDangerResponse(BaseModel):
    """
    Response model for the fire danger assessment API.
    Includes fire danger for subgrids and relevant POIs.
    """
    total_subgrids_with_data: int
    total_pois: int
    total_pois_within_bbox: int
    total_pois_within_search_buffer: int
    subgrids: Dict[str, SubgridData]
    key_pois: List[POIData] # New key for POIs
    message: str