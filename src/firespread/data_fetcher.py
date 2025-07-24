import ee
import math
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, List, Optional
import os
import json
from google.oauth2 import service_account # <-- This is the NEW and CRUCIAL import

# Constants from your original script
# Note: You might consider making these configurable or passing them if they change often.
TILE_SCALE = 16

def initialize_earth_engine():
    """
    Initializes the Earth Engine API using a service account key file.
    This version includes the necessary Earth Engine specific OAuth scope.
    """
    try:
        service_account_key_path = os.getenv('EE_SERVICE_ACCOUNT_KEY')

        if service_account_key_path and os.path.exists(service_account_key_path):
            print(f"Attempting Earth Engine initialization using service account key file: {service_account_key_path}")

            # --- CRITICAL CHANGE: Define the required Earth Engine scope ---
            scopes = ['https://www.googleapis.com/auth/earthengine']

            # Use google.oauth2.service_account.Credentials.from_service_account_file
            # Pass the scopes explicitly
            credentials = service_account.Credentials.from_service_account_file(
                service_account_key_path,
                scopes=scopes # <--- This is the new argument
            )

            # Initialize Earth Engine with the created credentials object
            ee.Initialize(credentials)
            print("Earth Engine initialized successfully with service account credentials and correct scope.")
        else:
            print("EE_SERVICE_ACCOUNT_KEY environment variable not set or service account key file not found at specified path.")
            print("Attempting default Earth Engine initialization (may require prior 'earthengine authenticate' or Application Default Credentials).")
            # Note: Default initialization also requires correct permissions/scopes for the default method.
            ee.Initialize()
            print("Earth Engine initialized successfully with default method.")

    except ee.EEException as e:
        print(f"Earth Engine initialization failed: {e}")
        raise # Re-raise the exception for FastAPI
    except FileNotFoundError as e:
        print(f"Service account key file not found at the specified path: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during Earth Engine initialization: {e}")
        raise


def _calculate_grid_points(top_left: Tuple[float, float], bottom_right: Tuple[float, float], subgrid_size_m: int) -> ee.FeatureCollection:
    """
    Calculates and returns Earth Engine FeatureCollection of grid points.
    """
    lat_diff = top_left[0] - bottom_right[0]
    lon_diff = bottom_right[1] - top_left[1]

    center_lat = (top_left[0] + bottom_right[0]) / 2
    meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(center_lat)) + 1.175 * math.cos(4 * math.radians(center_lat))
    meters_per_deg_lon = 111412.84 * math.cos(math.radians(center_lat)) - 93.5 * math.cos(3 * math.radians(center_lat)) + 0.118 * math.cos(5 * math.radians(center_lat))

    num_lat_cells = int(abs(lat_diff * meters_per_deg_lat) / subgrid_size_m)
    num_lon_cells = int(abs(lon_diff * meters_per_deg_lon) / subgrid_size_m)

    if num_lat_cells == 0: num_lat_cells = 1
    if num_lon_cells == 0: num_lon_cells = 1

    print(f"Calculated grid: {num_lat_cells} rows, {num_lon_cells} columns.")

    lat_step = lat_diff / num_lat_cells
    lon_step = lon_diff / num_lon_cells

    grid_points = []
    for r in range(num_lat_cells):
        for c in range(num_lon_cells):
            lat = top_left[0] - (r + 0.5) * lat_step
            lon = top_left[1] + (c + 0.5) * lon_step
            point = ee.Feature(ee.Geometry.Point([lon, lat]), {
                'row': r,
                'col': c,
                'original_lat': lat,
                'original_lon': lon
            })
            grid_points.append(point)

    return ee.FeatureCollection(grid_points)

def _load_ee_datasets(region: ee.Geometry, forecast_hours: int, scale: int) -> ee.Image:
    """
    Loads and combines all necessary Earth Engine datasets into a single image.
    """
    print("Loading Earth Engine datasets...")

    # Global Forecast System (GFS) - 0.25 degree resolution (~25km)
    now = datetime.now(timezone.utc)
    target_time = now + timedelta(hours=forecast_hours)

    gfs_collection = ee.ImageCollection('NOAA/GFS0P25') \
        .filterDate(target_time - timedelta(days=1), target_time + timedelta(days=1)) \
        .filterBounds(region) \
        .sort('system:time_start', False)
    gfs_image = gfs_collection.first()
    if gfs_image is None:
        print("WARNING: No GFS image found for the specified time range. Attempting to use a broader range.")
        gfs_image = ee.ImageCollection('NOAA/GFS0P25') \
            .filterDate(now - timedelta(days=2), now) \
            .filterBounds(region) \
            .sort('system:time_start', False) \
            .first()
        if gfs_image is None:
            raise ValueError("No GFS data available for the region and time. Please check GEE availability or broaden time filter.")

    # IMPORTANT: The band names for GFS were incorrect in your previous script.
    # They should be 'u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground', 'temperature_2m_above_ground'.
    gfs_bands = gfs_image.select(
        ['u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground', 'temperature_2m_above_ground'],
        ['u_wind', 'v_wind', 'temperature']
    )
    print("GFS data loaded.")

    # EC JRC global map of forest cover 2020, V2 (10m resolution, binary forest/non-forest)
    print("Loading EC JRC Global Forest Cover 2020, V2 (10m)...")
    jrc_forest_cover_collection = ee.ImageCollection('JRC/GFC2020/V2')
    jrc_forest_cover = jrc_forest_cover_collection.mosaic().select('Map').rename('jrc_forest_cover_presence')
    print("JRC Global Forest Cover loaded.")

    # UMD/hansen/global_forest_change_2024_v1_12 (for treecover2000 and lossyear) - ~30m resolution
    print("Loading UMD/hansen/global_forest_change_2024_v1_12 for treecover2000 and lossyear...")
    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
    hansen_treecover = hansen.select('treecover2000').rename('hansen_treecover2000')
    lossyear = hansen.select('lossyear').rename('lossyear')
    print("Hansen treecover and lossyear loaded.")

    # Copernicus CORINE Land Cover (CLC) - 100m resolution (2018 is a common reference)
    print("Loading Copernicus CORINE Land Cover 2018 (100m)...")
    corine_lc = ee.Image('COPERNICUS/CORINE/V20/100m/2018').select('landcover').rename('corine_land_cover')
    print("Copernicus CORINE Land Cover loaded.")

    # SRTM Digital Elevation Model - ~30m resolution
    print("Loading SRTM Digital Elevation Model...")
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('altitude')
    print("SRTM DEM loaded.")

    # Combine all bands into a single image for sampling
    combined_image = ee.Image.cat([
        jrc_forest_cover.reproject(crs='EPSG:4326', scale=scale),
        hansen_treecover.reproject(crs='EPSG:4326', scale=scale),
        corine_lc.reproject(crs='EPSG:4326', scale=scale),
        lossyear.reproject(crs='EPSG:4326', scale=scale),
        gfs_bands.reproject(crs='EPSG:4326', scale=scale),
        srtm.reproject(crs='EPSG:4326', scale=scale)
    ])
    print("All datasets combined into a single image.")
    return combined_image

def get_subgrid_data_from_ee(
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float],
    subgrid_size_m: int,
    forecast_hours: int
) -> List[Dict]:
    """
    Fetches raw Earth Engine data for a grid of sub-regions.

    Args:
        top_left (tuple): (latitude, longitude) of the top-left corner of the bounding box.
        bottom_right (tuple): (latitude, longitude) of the bottom-right corner of the bounding box.
        subgrid_size_m (int): The desired size of each subgrid in meters.
        forecast_hours (int): The number of hours into the future for the GFS forecast.

    Returns:
        list: A list of dictionaries, where each dictionary represents a sampled feature.
    """
    # 1. Define the bounding box
    bbox_coords = [
        [top_left[1], top_left[0]],        # Top-left (lon, lat)
        [bottom_right[1], top_left[0]],     # Top-right (lon, lat)
        [bottom_right[1], bottom_right[0]], # Bottom-right (lon, lat)
        [top_left[1], bottom_right[0]],     # Bottom-left (lon, lat)
        [top_left[1], top_left[0]]          # Close the polygon
    ]
    region = ee.Geometry.Polygon(bbox_coords)

    # 2. Calculate the grid points
    grid_fc = _calculate_grid_points(top_left, bottom_right, subgrid_size_m)

    # 3. Load and combine Earth Engine datasets
    combined_image = _load_ee_datasets(region, forecast_hours, subgrid_size_m)

    # Sample the combined image at each grid point
    print(f"Sampling combined image at {subgrid_size_m}m resolution...")
    sampled_data = combined_image.sampleRegions(
        collection=grid_fc,
        properties=['row', 'col', 'original_lat', 'original_lon'],
        scale=subgrid_size_m,
        tileScale=TILE_SCALE,
        geometries=True
    )
    print("Sampling complete. Fetching info from Earth Engine (this may take a moment)...")

    return sampled_data.getInfo()['features']