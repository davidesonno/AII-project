import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # Convert to radians
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in km

def adaptive_haversine(lat1, lon1, lat2, lon2):
    # Compute local Earth radius based on latitude
    a = 6378.137  # Equatorial radius (km)
    b = 6356.752  # Polar radius (km)
    lat_avg = np.radians((lat1 + lat2) / 2)
    
    R = np.sqrt(((a**2 * np.cos(lat_avg))**2 + (b**2 * np.sin(lat_avg))**2) /
                ((a * np.cos(lat_avg))**2 + (b * np.sin(lat_avg))**2))
    
    # Standard Haversine calculation with adaptive R
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in km

def ecef_distance(lat1, lon1, lat2, lon2):
    # WGS-84 ellipsoid constants
    a = 6378.137  # Semi-major axis (km)
    f = 1 / 298.257223563  # Flattening
    e2 = 2*f - f**2  # Square of eccentricity

    def latlon_to_ecef(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = N * np.cos(lat) * np.cos(lon)
        y = N * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2)) * np.sin(lat)
        return np.array([x, y, z])

    # Convert to NumPy arrays for broadcasting
    p1 = np.asarray(latlon_to_ecef(lat1, lon1))[:, np.newaxis]  # Make it (3,1)
    p2 = np.asarray(latlon_to_ecef(lat2, lon2))  # (3, N)

    return np.linalg.norm(p1 - p2, axis=0)  # Element-wise norm

def search_close_readings(df, center, radius, method=haversine):
    center_lat, center_lon = map(float, center.split(','))
    
    # Extract lat/lon values from 'geopoint' column
    lat_lon = np.array([list(map(float, gp.split(','))) for gp in df['geopoint']])
    
    # Compute all distances using Haversine formula (vectorized)
    distances = method(center_lat, center_lon, lat_lon[:, 0], lat_lon[:, 1])
    
    # Return filtered DataFrame
    return df[distances <= radius]

def divide_df_by_location(df, geopoint, radius, name=None, v=1):
    if v>0: print(f'Location{" "+name if name else ""}: {geopoint}')
    if v>0: print(f' > Filtering close traffic data...')
    close_df = search_close_readings(df, geopoint, radius)
    close_df=close_df.drop(columns=['geopoint', 'codice spira'])
    if v>0: print(' > Summing up hour data...')
    df_melted = close_df.reset_index().melt(id_vars=["data"], var_name="Hour", value_name="Traffic_value")
    df_melted['Hour'] = df_melted['Hour'].apply(lambda x: x.split('-')[0])
    df_melted['data'] = pd.to_datetime(df_melted['data'] + ' ' + df_melted['Hour'])
    
    df_melted = df_melted.rename(columns={'data': 'Date'}
                                ).drop(columns=['Hour']
                                ).groupby('Date', as_index=False)['Traffic_value'].sum(
                                ).resample('1h', on='Date'
                                ).mean(
                                ).reset_index(
                                ).fillna(0)
    df_melted = df_melted.set_index('Date')
    return df_melted