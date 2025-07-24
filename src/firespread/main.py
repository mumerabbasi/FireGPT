# main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Tuple
import ee # Keep ee import in main for startup initialization

# Import the refactored modules and new POI models
from . import models
from .data_fetcher import initialize_earth_engine, get_subgrid_data_from_ee
from .fire_model import calculate_fire_danger, CORINE_LC_MAPPING # Keep CORINE_LC_MAPPING here
from .poi_fetcher import fetch_pois_from_overpass, calculate_poi_relationships

app = FastAPI(
    title="Forest Fire Danger Assessment API",
    description="An API to assess forest fire danger based on geospatial data and weather forecasts.",
    version="1.0.0"
)

# Initialize Earth Engine when the FastAPI app starts up
@app.on_event("startup")
async def startup_event():
    initialize_earth_engine()

@app.post(
    "/assess-fire-danger",
    response_model=models.FireDangerResponse,
    summary="Assess forest fire danger and identify key POIs for a given bounding box",
    response_description="A detailed report of fire danger scores for each subgrid and relevant POIs within/near the specified area."
)
async def assess_fire_danger(request: models.FireDangerRequest):
    """
    Receives a bounding box and parameters, fetches relevant geospatial data from Earth Engine,
    calculates a simplified fire danger score for each subgrid, and identifies key Points of Interest (POIs)
    within and near the bounding box, providing proximity information.

    **Input:**
    - `bbox`: Defines the area of interest using `top_left_lat`, `top_left_lon`,
      `bottom_right_lat`, and `bottom_right_lon`.
    - `subgrid_size_m`: The desired size (in meters) for each square subgrid cell
      (e.g., 100 for 100x100m). Minimum 10m.
    - `forecast_hours`: The number of hours into the future for the GFS weather forecast
      (e.g., 3 for 3 hours from now). Minimum 0 hours.
    - `poi_search_buffer_m`: Buffer distance in meters outside the main bounding box to search for
      Points of Interest. A value of 0 means search only within the bbox.

    **Output:**
    Returns a JSON object containing:
    - Fire danger scores and contributing factors for each valid subgrid.
    - A list of key POIs, including their location, type, whether they are within the
      main bounding box, and their distance to the closest assessed subgrid.
    """
    # Reorder request bbox to (south, west, north, east) for Overpass API
    overpass_bbox_formatted = (
        request.bbox.bottom_right_lat, # South
        request.bbox.top_left_lon,    # West
        request.bbox.top_left_lat,    # North
        request.bbox.bottom_right_lon # East
    )

    # top_left for GEE remains (top_left_lat, top_left_lon)
    top_left = (request.bbox.top_left_lat, request.bbox.top_left_lon)
    bottom_right = (request.bbox.bottom_right_lat, request.bbox.bottom_right_lon)
    subgrid_size_m = request.subgrid_size_m
    forecast_hours = request.forecast_hours
    poi_search_buffer_m = request.poi_search_buffer_m # Get the new parameter

    try:
        # 1. Fetch raw data from Earth Engine
        data_features = get_subgrid_data_from_ee(
            top_left, bottom_right, subgrid_size_m, forecast_hours
        )

        output_subgrids: Dict[str, models.SubgridData] = {}
        raw_subgrid_data_for_poi_calc: Dict[str, Dict] = {} # Store raw data for POI calculations
        subgrid_index = 0

        # Process each sampled feature from Earth Engine
        for feature in data_features:
            properties = feature.get('properties', {})
            geom = feature.get('geometry')

            jrc_forest_cover_presence_val = properties.get('jrc_forest_cover_presence')
            hansen_treecover_val = properties.get('hansen_treecover2000')
            lossyear_val = properties.get('lossyear')
            corine_land_cover_code = properties.get('corine_land_cover')
            u_wind_val = properties.get('u_wind')
            v_wind_val = properties.get('v_wind')
            temperature_val_k = properties.get('temperature')
            altitude_val = properties.get('altitude')

            row_idx = properties.get('row')
            col_idx = properties.get('col')
            original_lat = properties.get('original_lat')
            original_lon = properties.get('original_lon')

            center_lat, center_lon = None, None
            if geom and 'coordinates' in geom and len(geom['coordinates']) == 2:
                center_lon = geom['coordinates'][0]
                center_lat = geom['coordinates'][1]
            else:
                center_lat = original_lat
                center_lon = original_lon

            # Check for critical data presence for fire danger calculation
            if all(val is not None for val in [
                hansen_treecover_val, corine_land_cover_code, lossyear_val,
                u_wind_val, v_wind_val, temperature_val_k, altitude_val,
                center_lat, center_lon
            ]):
                fire_danger_results = calculate_fire_danger(
                    hansen_treecover2000=hansen_treecover_val,
                    corine_land_cover=corine_land_cover_code,
                    lossyear=lossyear_val,
                    u_wind=u_wind_val,
                    v_wind=v_wind_val,
                    temperature_k=temperature_val_k,
                    altitude=altitude_val,
                    forecast_hours=forecast_hours
                )

                subgrid_id_key = f'subgrid_{subgrid_index}'
                output_subgrids[subgrid_id_key] = models.SubgridData(
                    row=row_idx,
                    col=col_idx,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    properties=models.SubgridProperties(
                        jrc_forest_cover_presence_10m=jrc_forest_cover_presence_val,
                        hansen_treecover2000_perc=hansen_treecover_val,
                        corine_land_cover_code=corine_land_cover_code,
                        corine_land_cover_description=CORINE_LC_MAPPING.get(corine_land_cover_code, "Unknown"),
                        hansen_lossyear=lossyear_val,
                        u_wind_ms=u_wind_val,
                        v_wind_ms=v_wind_val,
                        temperature_k=temperature_val_k,
                        altitude_m=altitude_val
                    ),
                    fire_danger=models.FireDangerResult(
                        score=fire_danger_results['fire_danger_score'],
                        factors=fire_danger_results['contributing_factors']
                    )
                )
                
                # Store necessary info for POI linkage
                raw_subgrid_data_for_poi_calc[subgrid_id_key] = {
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'fire_danger_score': fire_danger_results['fire_danger_score']
                }
                subgrid_index += 1
        
        if not output_subgrids:
            return JSONResponse(
                status_code=status.HTTP_204_NO_CONTENT,
                content={
                    "total_subgrids_with_data": 0,
                    "subgrids": {},
                    "key_pois": [],
                    "message": "No valid subgrids found within the specified bounding box for assessment or all data was missing."
                }
            )

        # 2. Fetch and process POIs using the correctly formatted bbox and the new buffer parameter
        raw_pois = fetch_pois_from_overpass(overpass_bbox_formatted, buffer_m=poi_search_buffer_m)

        # Filter subgrids to include only those with a high fire danger score for POI proximity check
        high_danger_subgrids = {
            sg_id: sg_data for sg_id, sg_data in raw_subgrid_data_for_poi_calc.items()
            if sg_data['fire_danger_score'] >= 50 # Example threshold for "high danger"
        }
        
        # If no high danger subgrids, use all subgrids for general proximity
        subgrids_for_poi_calc = high_danger_subgrids if high_danger_subgrids else raw_subgrid_data_for_poi_calc

        processed_pois = calculate_poi_relationships(
            raw_pois,
            subgrids_for_poi_calc,
            overpass_bbox_formatted # Pass the bbox in (south, west, north, east) order for 'is_within_bbox' check
        )

        return models.FireDangerResponse(
            total_subgrids_with_data=len(output_subgrids),
            total_pois=len(raw_pois),
            total_pois_within_bbox=len([poi for poi in processed_pois if poi.is_within_bbox]),
            total_pois_within_search_buffer=len([poi for poi in processed_pois if poi.distance_from_bbox_center_m is not None]),
            subgrids=output_subgrids,
            key_pois=processed_pois,
            message="Fire danger assessment and POI identification completed successfully."
        )

    except ee.EEException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Earth Engine error: {e}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input data error: {e}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Forest Fire Danger Assessment API is running. Use /assess-fire-danger endpoint for assessment."}