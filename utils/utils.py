from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def merge_csv_to_dataframe(input_folder, **kwargs):
    
    """Merge all CSV files in a folder into a single pandas DataFrame."""
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    dataframes = [pd.read_csv(os.path.join(input_folder, file), **kwargs) for file in csv_files]
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Merged {len(csv_files)} CSV files")
    
    return merged_df


def plot_time_series(dfs, value_column, date_column, legends, start_date=None, end_date=None, max_rows=5000, downsample_factor=8, title=''):
    plt.figure(figsize=(40, 6))

    for df, legend in zip(dfs, legends):
        if start_date and end_date:
            df = df[(df[date_column] >= start_date) & (df[date_column] < end_date)]
        
        df = df.sort_values(date_column)

        if len(df) > max_rows:
            df = df.iloc[::downsample_factor]  

        plt.plot(df[date_column], df[value_column], label=legend, linewidth=1)

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Time Series of {title}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_time_series_per_station(df, value_column, date_column, station_column, start_date=None, end_date=None, max_rows=5000, downsample_factor=8, title='Air Pollution Time Series'):
    plt.figure(figsize=(25, 6))

    # Apply date filtering if needed
    if start_date and end_date:
        df = df[(df[date_column] >= start_date) & (df[date_column] < end_date)]
    
    df = df.sort_values(date_column)

    # Loop through unique station codes and plot each
    for station in df[station_column].unique():
        station_df = df[df[station_column] == station]
        
        if len(station_df) > max_rows:
            station_df = station_df.iloc[::downsample_factor]  # Downsampling

        plt.plot(station_df[date_column], station_df[value_column], label=f'Station {station}', linewidth=1, alpha=1)

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend(title='Stations', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def map_date_to_idx(df, column_name, start_date:datetime):
    '''
	convert the date to an index, starting from `start_date` and increasing by 1 each hour
    TODO: not sure if it works correctly
    '''
    # Ensure the column is in datetime format
    df[column_name] = pd.to_datetime(df[column_name])

    # Compute the index based on the difference in hours
    df['date_index'] = ((df[column_name] - start_date).dt.total_seconds() // 3600).astype(int)

    return df

def convert_percentage_to_number(df):
    for col in df.columns:
        first_value = df[col].dropna().astype(str).iloc[0]  # Get the first non-null value as a string
        if first_value.endswith('%'):  # Check if it looks like a percentage
            df[col] = df[col].str.rstrip('%').astype(float) / 100
    return df


def apply_accuracy_df(readings_df, accuracies_df, add_verified_col=False, max_multiplier=100, half_multiplier=2):

    def accuracy_coeff(accuracies:pd.Series, max, half):
        return accuracies.where(accuracies == 0, 1 / accuracies)

    merged_df = readings_df.merge(accuracies_df, on=['data', 'codice spira'], 
                                  suffixes=('_reading', '_accuracy'), how='left').fillna(1)

    if add_verified_col:
        merged_df['accurate'] = merged_df.iloc[:, 2 + len(readings_df.columns[2:]) :].notna().all(axis=1)

    # Multiply only where accuracy is available, keeping original value if missing
    for col in readings_df.columns[2:]:  # Skip 'data' and 'codice spira'
        merged_df[col + '_reading'] = (merged_df[col + '_reading']  * accuracy_coeff(merged_df[col + '_accuracy'], max_multiplier, half_multiplier)).astype(int)

    cols = ['data', 'codice spira', 'accurate'] if add_verified_col else ['data', 'codice spira']
    final_df = merged_df[cols + [col + '_reading' for col in readings_df.columns[2:]]]
    final_df.columns = cols + list(readings_df.columns[2:])  # Rename columns back

    return final_df

def resample_df_on_column(df, agents_dict:dict, column='Date', v=1 ):
    s=df['Station'].iloc[0]
    if v>0: print(f'Resampling for station "{s}..."')
    resampled_dfs = []
    df_agents = np.unique(df['Agent'])
    for agent in df_agents:
        if v > 1: print(f'- Resampling {agent} ({agents_dict[agent]})')
        mask = df['Agent'] == agent
        resampled = (
            df[mask]
            .resample(agents_dict[agent], on=column)
            .max()  
            .reset_index()
        )
        resampled['Agent'] = agent 
        resampled_dfs.append(resampled)
    res = pd.concat(resampled_dfs)
    res['Station'] = res['Station'].fillna(df.iloc[0]['Station'])
    res['Unit'] = res['Unit'].fillna(df.iloc[0]['Unit'])
    return res


def fill_missing_dates(df: pd.DataFrame, column=None, method: str = 'ffill'):
    '''
	fills all the nans using the method passed. If using mean filling,
    `column` specifies the column to use for the average.
    '''
    if method == 'mfill':
        return df.fillna(df[column].mean())
    
    elif hasattr(df, method):  
        return getattr(df, method)()
    
    else:
        raise ValueError(f"Invalid method: {method}")
    
def fill_missing_dates_on_column_value(df, column, column_to_fill, mode, v=1):
    '''
    Specify the `method` to use to fill `Nan` values.

    Apply filling indipendently on the unique values of `column`.

    Usually `method` is one of ['ffill', 'bfill', 'interpolate', 'mfill']
    '''
    s=df['Station'].iloc[0]
    if v>0: print(f'Filling station "{s}..."')
    for val in np.unique(df[column]):
        if v>1: print(f'- Filling {val} values')
        mask = df[column] == val
        df[mask] = fill_missing_dates(df[mask], column_to_fill, mode)
    return df

def df_to_agents_dict(df, column='Agent', v=1):
    s=df['Station'].iloc[0]
    if v>0: print(f'Splitting station "{s}"...')
    agents_dict = {}
    for agent in np.unique(df[column]):
        agents_dict[agent] = df[df[column] == agent].sort_values(by='Date')
    
    return agents_dict


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

def divide_df_by_location(df, geopoint, radius,v=1):
    if v>0: print(f'Location: {geopoint}')
    if v>0: print(f'> Filtering close traffic data...')
    close_df = search_close_readings(df, geopoint, radius)
    close_df=close_df.drop(columns=['geopoint', 'codice spira'])
    if v>0: print('> Summing up hour data...')
    df_melted = close_df.melt(id_vars=["data"], var_name="Hour", value_name="Value")
    df_melted['Hour'] = df_melted['Hour'].apply(lambda x: x.split('-')[0])
    df_melted['data'] = pd.to_datetime(df_melted['data'] + ' ' + df_melted['Hour'])
    
    df_melted = df_melted.rename(columns={'data': 'Date'}
                                ).drop(columns=['Hour']
                                ).groupby('Date', as_index=False)['Value'].sum(
                                ).resample('1h', on='Date'
                                ).mean(
                                ).reset_index(
                                ).fillna(0)
    return df_melted