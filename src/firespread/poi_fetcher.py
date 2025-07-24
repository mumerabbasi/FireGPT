# poi_fetcher.py
import requests
import json
import math # Ensure math is imported for trigonometry functions
from typing import List, Dict, Tuple, Optional
from geopy.distance import geodesic
from geopy.point import Point
from . import models

# Define common Overpass API endpoint
OVERPASS_API_URL = "http://overpass-api.de/api/interpreter"

# Define OSM tags for key Places of Interest
# This list can be expanded based on what's critical for evacuation/alerts
OSM_POI_TAGS = {
    "amenity": ["hospital", "clinic", "school", "kindergarten", "community_centre", "fire_station", "police", "pharmacy", "shelter"],
    "building": ["church", "mosque", "synagogue", "temple", "civic", "public", "apartments", "dormitory", "detached", "house", "residential"], # Added more residential types
    "leisure": ["park", "pitch", "stadium"], # Large open areas
    "tourism": ["hotel", "motel", "hostel", "camp_site", "caravan_site"], # Tourist areas
    "shop": ["supermarket", "mall", "department_store"], # Commercial centers
    "railway": ["station"], # Transport hubs
    "highway": ["rest_area"], # Rest areas on highways
    "place": ["village", "town", "city"], # Settlements (will often be very large, might need filtering)
}

def _build_overpass_query(bbox_overpass_format: Tuple[float, float, float, float], buffer_m: int = 2000) -> str:
    """
    Builds an Overpass QL query string for fetching POIs within a buffered bounding box.

    Args:
        bbox_overpass_format (tuple): (south, west, north, east) of the original query area.
        buffer_m (int): Buffer distance in meters to expand the bbox for nearby POIs.

    Returns:
        str: Overpass QL query string.
    """
    south_orig, west_orig, north_orig, east_orig = bbox_overpass_format

    # Calculate buffered bounding box for querying
    # This is a rough approximation for buffering in degrees.
    # More precise buffering would involve projecting to a local UTM zone or similar.
    center_lat = (south_orig + north_orig) / 2
    lat_buffer_deg = buffer_m / 111000.0 # ~111 km per degree of latitude
    
    # Avoid division by zero if at poles, though unlikely for typical maps
    lon_buffer_deg = buffer_m / (111000.0 * math.cos(math.radians(center_lat))) if abs(math.cos(math.radians(center_lat))) > 1e-6 else lat_buffer_deg

    buffered_south = south_orig - lat_buffer_deg
    buffered_west = west_orig - lon_buffer_deg
    buffered_north = north_orig + lat_buffer_deg
    buffered_east = east_orig + lon_buffer_deg

    # Construct the OR conditions for all desired tags within a union block
    query_elements = []
    for tag_key, tag_values in OSM_POI_TAGS.items():
        for value in tag_values:
            # Query for nodes, ways, and relations. Use 'out center;' to get a lat/lon for all.
            query_elements.append(f'  node["{tag_key}"="{value}"]({buffered_south},{buffered_west},{buffered_north},{buffered_east});')
            query_elements.append(f'  way["{tag_key}"="{value}"]({buffered_south},{buffered_west},{buffered_north},{buffered_east});')
            query_elements.append(f'  rel["{tag_key}"="{value}"]({buffered_south},{buffered_west},{buffered_north},{buffered_east});')

    query_body = "\n".join(query_elements)

    # Final Overpass QL query: use a union block and then output elements
    return f"""
        [out:json][timeout:60];
        (
          {query_body}
        );
        out center;
    """

def fetch_pois_from_overpass(bbox_overpass_format: Tuple[float, float, float, float], buffer_m: int = 2000) -> List[Dict]:
    """
    Fetches key Points of Interest (POIs) from OpenStreetMap via Overpass API.

    Args:
        bbox_overpass_format (tuple): (south, west, north, east) of the original query area.
        buffer_m (int): Buffer distance in meters to expand the bbox for nearby POIs.

    Returns:
        List[Dict]: A list of dictionaries, each representing a POI with 'osm_id',
                    'name', 'type', 'subtype', 'lat', 'lon'.
    """
    query = _build_overpass_query(bbox_overpass_format, buffer_m)
    headers = {'User-Agent': 'ForestFireDangerAPI/1.0'} # Good practice to identify your client

    try:
        response = requests.post(OVERPASS_API_URL, data=query, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        pois = []
        for element in data.get('elements', []):
            # 'center' key exists for ways and relations when 'out center;' is used.
            # Nodes have 'lat' and 'lon' directly.
            lat = element.get('lat') or element.get('center', {}).get('lat')
            lon = element.get('lon') or element.get('center', {}).get('lon')

            if lat is None or lon is None:
                continue # Skip elements without valid coordinates

            tags = element.get('tags', {})
            poi_type = "Other" # Default
            poi_subtype = None

            # Find the primary POI type based on our defined tags
            for tag_key, tag_values in OSM_POI_TAGS.items():
                if tag_key in tags and tags[tag_key] in tag_values:
                    poi_type = tag_key # e.g., 'amenity', 'building'
                    poi_subtype = tags[tag_key] # e.g., 'hospital', 'residential'
                    break # Found a match, break from inner loop
            
            # Fallback to general type if no specific match, but tags exist
            if poi_type == "Other" and tags:
                poi_type = list(tags.keys())[0] # Take first key as general type
                poi_subtype = tags[poi_type] # Take its value as subtype

            pois.append({
                'osm_id': element['id'],
                'name': tags.get('name'),
                'type': poi_type,
                'subtype': poi_subtype,
                'lat': lat,
                'lon': lon
            })
        print(f"Fetched {len(pois)} POIs from Overpass API.")
        return pois

    except requests.exceptions.RequestException as e:
        print(f"Error fetching POIs from Overpass API: {e}")
        # Return an empty list on error so the main function doesn't crash
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding Overpass API response: {e}")
        return []


def calculate_poi_relationships(
    pois: List[Dict],
    subgrids: Dict[str, Dict], # Expecting dictionary with 'center_lat', 'center_lon'
    main_bbox_overpass_format: Tuple[float, float, float, float] # (south, west, north, east)
) -> List['models.POIData']: # Forward reference for POIData
    """
    Calculates relationships (closest subgrid, distance) for each POI.

    Args:
        pois (List[Dict]): List of raw POI dictionaries from fetch_pois_from_overpass.
        subgrids (Dict[str, Dict]): Dictionary of processed subgrid data.
                                    Keys are 'subgrid_X', values are dictionaries of data
                                    containing 'center_lat' and 'center_lon'.
        main_bbox_overpass_format (Tuple): The main query bounding box in (south, west, north, east) format.

    Returns:
        List[models.POIData]: List of POIData Pydantic models with calculated relationships.
    """
    # Import models locally to avoid circular dependency issues if needed, or rely on top-level import
    from . import models 

    processed_pois: List['models.POIData'] = []
    
    # Unpack the main bbox for 'within' check
    main_south, main_west, main_north, main_east = main_bbox_overpass_format

    for poi in pois:
        poi_lat, poi_lon = poi['lat'], poi['lon']
        poi_point = Point(poi_lat, poi_lon)

        # Determine if POI is within the main query bounding box
        is_within_bbox = (main_south <= poi_lat <= main_north and
                          main_west <= poi_lon <= main_east)

        closest_subgrid_info = None
        min_distance_to_subgrid = float('inf')

        # Calculate distance to all subgrids to find the closest one
        if subgrids: # Ensure there are subgrids to compare against
            for subgrid_id, subgrid_data in subgrids.items():
                subgrid_lat = subgrid_data['center_lat']
                subgrid_lon = subgrid_data['center_lon']
                
                subgrid_point = Point(subgrid_lat, subgrid_lon)
                
                distance = geodesic(poi_point, subgrid_point).m # Distance in meters
                
                if distance < min_distance_to_subgrid:
                    min_distance_to_subgrid = distance
                    closest_subgrid_info = models.ClosestSubgridInfo(
                        subgrid_id=subgrid_id,
                        distance_m=round(min_distance_to_subgrid, 2)
                    )

        # Calculate distance to the center of the main bounding box
        bbox_center_lat = (main_south + main_north) / 2
        bbox_center_lon = (main_west + main_east) / 2
        distance_from_bbox_center_m = round(geodesic(poi_point, Point(bbox_center_lat, bbox_center_lon)).m, 2)


        processed_pois.append(models.POIData(
            osm_id=poi['osm_id'],
            name=poi['name'],
            type=poi['type'],
            subtype=poi['subtype'],
            lat=poi['lat'],
            lon=poi['lon'],
            is_within_bbox=is_within_bbox,
            closest_subgrid=closest_subgrid_info,
            distance_from_bbox_center_m=distance_from_bbox_center_m
        ))
    
    return processed_pois