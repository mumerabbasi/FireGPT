import math
from datetime import datetime

# --- Constants ---
# Define CORINE Land Cover mapping for interpretation (moved here)
CORINE_LC_MAPPING = {
    111: "Continuous urban fabric", 112: "Discontinuous urban fabric",
    121: "Industrial or commercial units", 122: "Road and rail networks",
    123: "Port areas", 124: "Airports",
    131: "Mineral extraction sites", 132: "Dump sites", 133: "Construction sites",
    141: "Green urban areas", 142: "Sport and leisure facilities",
    211: "Non-irrigated arable land", 212: "Permanently irrigated land", 213: "Rice fields",
    221: "Vineyards", 222: "Fruit trees and berry plantations", 223: "Olive groves",
    231: "Pastures",
    241: "Annual crops associated with permanent crops", 242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture, with significant areas of natural vegetation",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland-shrub",
    331: "Beaches, dunes, sands", 332: "Bare rocks", 333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes", 412: "Peat bogs",
    421: "Salt marshes", 422: "Salines", 423: "Intertidal flats",
    511: "Water courses", 512: "Water bodies",
    521: "Coastal lagoons", 522: "Estuaries", 523: "Sea and ocean"
}

def calculate_fire_danger(
    hansen_treecover2000: float,
    corine_land_cover: int,
    lossyear: int,
    u_wind: float,
    v_wind: float,
    temperature_k: float,
    altitude: float,
    forecast_hours: int
) -> dict:
    """
    Calculates a simplified fire danger score based on provided environmental variables.
    This is a conceptual model; a real fire danger model would be much more complex.

    Args:
        hansen_treecover2000 (float): Tree cover percentage (0-100%) from Hansen 2000.
        corine_land_cover (int): CORINE Land Cover class ID. Used for primary fuel type and inferring leaf type.
        lossyear (int): Year of forest loss (0 if no loss, otherwise year - 2000).
                        Recent loss (e.g., within 5 years) might indicate cleared land
                        (lower fuel initially) or increased dead fuel (higher risk).
        u_wind (float): Eastward wind component (m/s).
        v_wind (float): Northward wind component (m/s).
        temperature_k (float): Air temperature in Kelvin. Higher temperature means higher risk.
        altitude (float): Altitude in meters. Can influence local wind and temperature.
        forecast_hours (int): Number of hours into the forecast (for context, not directly used).

    Returns:
        dict: A dictionary containing:
            - 'fire_danger_score': A numerical score (0-100, higher = more danger).
            - 'contributing_factors': A list of factors influencing the score.
    """

    score = 0
    factors = []

    # Convert temperature to Celsius for easier interpretation
    temperature_c = temperature_k - 273.15 if temperature_k is not None else None

    # Calculate wind speed from U and V components
    wind_speed = math.sqrt(u_wind**2 + v_wind**2) if u_wind is not None and v_wind is not None else None

    # Determine dominant leaf type based on CORINE, if available
    inferred_leaf_type = None
    if corine_land_cover is not None:
        if corine_land_cover == 311: # Broad-leaved forest
            inferred_leaf_type = 'Broadleaf'
        elif corine_land_cover == 312: # Coniferous forest
            inferred_leaf_type = 'Coniferous'
        elif corine_land_cover == 313: # Mixed forest
            inferred_leaf_type = 'Mixed'

    # 1. Fuel Availability (Hansen Tree Cover, Inferred Leaf Type, CORINE Land Cover)
    if hansen_treecover2000 is not None:
        if hansen_treecover2000 > 50:
            score += 30 # Significant tree cover
            factors.append("High Tree Cover (Hansen)")
            if inferred_leaf_type == 'Coniferous':
                score += 15 # Coniferous forests are often more flammable
                factors.append("Coniferous Forest (inferred from CORINE)")
            elif inferred_leaf_type == 'Broadleaf':
                score += 5 # Broadleaf is generally less flammable
                factors.append("Broadleaf Forest (inferred from CORINE)")
            elif inferred_leaf_type == 'Mixed':
                score += 10 # Mixed forest
                factors.append("Mixed Forest (inferred from CORINE)")
        elif hansen_treecover2000 > 10:
            score += 10 # Moderate tree cover/scattered trees
            factors.append("Moderate Tree Cover (Hansen)")
        elif hansen_treecover2000 > 0:
            factors.append(f"Low Tree Cover ({hansen_treecover2000:.0f}%) (Hansen)")


    # CORINE Land Cover - for general fuel type beyond just trees
    if corine_land_cover is not None:
        first_digit = corine_land_cover // 100 # Get the hundreds digit

        if first_digit == 1: # Artificial Surfaces
            factors.append(f"Land Cover: {CORINE_LC_MAPPING.get(corine_land_cover, 'Artificial Surface')}")
            score -= 5
        elif first_digit == 2: # Agricultural Areas
            factors.append(f"Land Cover: {CORINE_LC_MAPPING.get(corine_land_cover, 'Agricultural Area')}")
            score -= 5
        elif first_digit == 4: # Wetlands
            factors.append(f"Land Cover: {CORINE_LC_MAPPING.get(corine_land_cover, 'Wetland')}")
            score -= 10 # More significant reduction for wetlands
        elif first_digit == 5: # Water Bodies
            factors.append(f"Land Cover: {CORINE_LC_MAPPING.get(corine_land_cover, 'Water Body')}")
            score -= 15 # Even more significant reduction for water
        elif corine_land_cover in [321, 322, 323]: # Natural grasslands, Moors and heathland, Sclerophyllous vegetation
            score += 8 # Grasslands/Shrublands can burn easily
            factors.append("Grassland/Heathland (CORINE)")
        elif corine_land_cover == 324: # Transitional woodland-shrub
            score += 10 # Shrublands are high fire risk
            factors.append("Transitional Woodland-Shrub (CORINE) - High Fuel")
        elif corine_land_cover == 334: # Burnt areas
            score -= 10 # Assuming lack of immediate fuel, or could be monitored differently
            factors.append("Burnt Area (CORINE - less immediate fuel)")

    # 2. Recent Forest Loss (as a potential indicator of dead fuel or cleared land)
    if lossyear is not None and lossyear > 0: # If there was a loss event
        current_year = datetime.now().year
        loss_actual_year = 2000 + lossyear # Convert GEE lossyear to actual year
        years_since_loss = current_year - loss_actual_year

        if 0 < years_since_loss <= 3: # Very recent loss (may mean more dead/drying fuel)
            score += 10
            factors.append(f"Very Recent Forest Loss ({years_since_loss} years ago) - Potential Dead Fuel")
        elif 3 < years_since_loss <= 10: # Recent loss (less immediate fuel if cleared, but can be regrowth)
            score += 5
            factors.append(f"Recent Forest Loss ({years_since_loss} years ago)")
        # For older loss, impact less significant on current risk

    # 3. Weather Conditions (Wind and Temperature)
    if wind_speed is not None:
        if wind_speed > 10: # Strong wind (>10 m/s ~ 36 km/h)
            score += 20
            factors.append(f"Strong Wind ({wind_speed:.1f} m/s)")
        elif wind_speed > 5: # Moderate wind (5-10 m/s)
            score += 10
            factors.append(f"Moderate Wind ({wind_speed:.1f} m/s)")
        else:
            factors.append(f"Low Wind ({wind_speed:.1f} m/s)")

    if temperature_c is not None:
        if temperature_c > 30: # Hot (>30°C)
            score += 15
            factors.append(f"High Temperature ({temperature_c:.1f}°C)")
        elif temperature_c > 20: # Warm (20-30°C)
            score += 5
            factors.append(f"Warm Temperature ({temperature_c:.1f}°C)")
        else:
            factors.append(f"Moderate Temperature ({temperature_c:.1f}°C)")

    # 4. Topography (Altitude - can influence microclimate and spread)
    if altitude is not None:
        if altitude > 1500: # High altitude, can be windier, different vegetation
            score += 5
            factors.append(f"High Altitude ({altitude:.0f}m)")
        elif altitude < 500: # Low altitude, could be valleys, less wind
             score -= 2 # Slight reduction if very low and not marshy
             factors.append(f"Low Altitude ({altitude:.0f}m)")

    # Clamp score to 0-100
    final_score = max(0, min(100, int(score)))

    return {
        'fire_danger_score': final_score,
        'contributing_factors': factors
    }