from utils import *
from .distances import *
from .date_encoding import *
from .missing_values import *

import pandas as pd
import numpy as np
import os
from datetime import timedelta


# === UTILITIES ===
def mean_wind_direction(vectors:pd.Series):
    x = np.cos(np.radians(vectors))
    y = np.sin(np.radians(vectors))
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_direction = np.degrees(np.arctan2(mean_y, mean_x))
    mean_direction = (mean_direction + 360) % 360  # Ensure [0, 360] range
        
    return mean_direction


def transform_weather_to_daily_df(df:pd.DataFrame):
    # 1. Averaging (Mean)
    # For variables that fluctuate throughout the day and are best represented by an average:
    # TAVG (Average Temperature) → Daily mean
    # RHAVG (Average Relative Humidity) → Daily mean
    # W_SCAL_INT (Scalar Wind Intensity) → Daily mean
    # RAD (Solar Radiation) → Daily mean (though total may also be used)
    agg = {col:'mean' for col in df.columns} # by default set the aggregation to sum. The columns might be named differently

    # For specific columns use different aggregations
    # 2. Summation (Total)
    # For variables that accumulate over the day:
    # PREC (Precipitation) → Daily total (sum of hourly values)
    # RAD (Solar Radiation) → Daily total (sum of hourly values)
    # ET0 (Evapotranspiration) → Daily total
    if 'PREC' in agg:
        agg['PREC'] = 'sum'
    if 'RAD' in agg: # NOTE: this is overriding RAD:sum
        agg['RAD'] = 'sum'
    if 'ET0' in agg: # dropped
        agg['ET0'] = 'sum'

    # 3. Vector-Based Aggregation (Wind)
    # For wind direction, a simple average can be misleading, so it’s computed using a vector average:
    # W_VEC_DIR (Vector Wind Direction) → Compute daily resultant vector and then find the mean direction
    if 'W_VEC_DIR' in agg:
        agg['W_VEC_DIR'] = mean_wind_direction

    # 4. Maximum / Minimum
    # For variables where the extreme values matter:
    # TAVG → Compute TMAX and TMIN separately if needed
    # RHAVG → Compute RHMAX and RHMIN if necessary
    max_min_agg = {}
    original_cols = []
    for col in ['TAVG', 'RHAVG']:
        if col in df.columns: # NOTE: this is overriding TAVG:sum and RHAVG:sum
            original_cols.append(col)
            max_min_agg[f"{col}_MAX"] = (col, 'max')
            max_min_agg[f"{col}_MIN"] = (col, 'min')

    # 5. Duration-Based Aggregation
    # For variables like leaf wetness, sum the number of hours it was wet:
    # LEAFW (Leaf Wetness) → Total hours per day where leaf wetness was detected
    if 'LEAFW' in agg:
        agg['LEAFW'] = lambda x: (x > 0).sum() # it gets applied to the pd.Series representing the day -> num. hours with LEAFW>0


    daily_df = df.copy().resample('D').agg(agg)

    if max_min_agg:
        # print(max_min_agg)
        # if you aggregate like df.agg(y=(col,aggfunction)) you transform the df
        # to only have the new column 'y' with that aggregation.
        # We have max_min_agg = {
        # 'TAVG_MAX': ('TAVG', 'max'),
        # 'TAVG_MIN': ('TAVG', 'min'),
        # 'RHAVG_MAX': ('RHAVG', 'max'),
        # 'RHAVG_MIN': ('RHAVG', 'min')}
        # In our case, we do df.agg(TAVG_MAX=('TAVG', 'max'),TAVG_MIN=...)
        # to have the 4 new columns, and then we join the aggregated dataframes
        df_max_min = df.copy().resample('D').agg(**max_min_agg)
        # display(df_max_min)
        # print(daily_df.columns,df_max_min.columns)
        daily_df = daily_df.join(df_max_min)
        daily_df = daily_df.drop(columns=original_cols)

    return daily_df

# === PREPROCESSING ===
def read_and_preprocess_dataset(datasets_folder, dataset, resample=False, fill_method='mfill', v=1):
    '''
	`v=0` stops any output prints.
	'''
    match dataset:
        case 'pollution':
            return preprocess_pollution_dataset(os.path.join(datasets_folder,'pollution/pollution.csv'), resample=resample, fill_method=fill_method, v=v)
            
        case 'traffic':
            return preprocess_traffic_dataset(os.path.join(datasets_folder,'traffic'), v=v)

        case 'weather':
            return preprocess_weather_dataset(os.path.join(datasets_folder,'weather'), v=v)
        
        case _:
            raise ValueError(f"Unknown dataset '{dataset}'. Only 'pollution', 'traffic' or 'weather' available")

def prepare_station_data_for_training(
        station_pollution_dict:dict,
        station_traffic_df:pd.DataFrame,
        weather_df:pd.DataFrame,
        encoding_method = 'full-sin-cos'
):
    '''
	Returns a dict mapping agents to their dataframe.\\
    The dataframe is normalized and encoded with the given `encoding_method`
    '''
    merged_dict = {}
    for agent,agent_pollution_df in station_pollution_dict.items():
        # if the agent is daily (PM), resample traffic and weather df before merging.
        if agent in ('PM2.5','PM10'):
            station_traffic_df = station_traffic_df.copy().resample('D').mean()
            daily_weather_df = transform_weather_to_daily_df(weather_df.copy())

        merged_dict[agent] = join_datasets(
            agent_pollution_df,
            station_traffic_df,
            daily_weather_df if agent in ('PM2.5','PM10') else weather_df,
            dropna=True
        )

    normalized_dict = {}
    for agent,merged_agent_df in merged_dict.items():
        normalized_dict[agent] = normalize_columns(
            merged_agent_df,
            skip=['Agent_value']
        )

    encoded_dict = {}
    for agent, normalized_agent_dict in normalized_dict.items():
        encoded_dict[agent] = encode_date_index(normalized_agent_dict, method=encoding_method)

    return encoded_dict

# === POLLUTION ===
def preprocess_pollution_dataset(csv_path, fill_method, resample=False, v=1):
    '''
	returns a list of dict, one for each station.

    each dict has a dataframe for each agent key value.
    '''
    df = pd.read_csv(csv_path, sep=';')
    df.rename(columns={
        'COD_STAZ': 'Station', 
        'AGENTE': 'Agent', 
        'DATA_INIZIO': 'Date', 
        'DATA_FINE': 'Ending_date', 
        'VALORE': 'Agent_value', 
        'UM': 'Unit'
    }, inplace=True)
    df.drop(columns=['Ending_date','Unit'], inplace=True)
    df = df[~df['Agent'].isin(['NO','NOX'])]
    df=df.sort_values(by='Date')
    df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: ' '.join(x.split('T')).split('+')[0]))
    df['Date'] = df['Date'].apply(lambda x: x - timedelta(minutes=x.minute))
    df = df.set_index('Date')
    df['Agent'] = df['Agent'].apply(lambda x: x.split(' ')[0])
    df = df[~df['Agent'].isin(['NO','NOX'])] # drop unwanted agents
    # split by station
    stations = np.unique(df['Station'])
    if v>0: print('Stations found: ', end='')
    if v>0: print(*stations, sep=', ')
    stations_dfs = {station:df[df['Station'] == station] for station in stations}

    # resampling
    if resample:
        agents_frequencies = {'PM10': '24h','PM2.5': '24h','CO': '1h','O3': '1h','NO2': '1h','C6H6': '1h'}
        resampled_dfs = {station:resample_df_on_column(station_df, agents_frequencies, v=v) for station,station_df in stations_dfs.items()}
        filled_dfs = {station:fill_missing_dates_on_column_value(resampled_df, column='Agent', column_to_fill='Agent_value', mode=fill_method, v=v) for station,resampled_df in resampled_dfs.times()}
        return {station:df_to_agents_dict(filled_df, drop_stations=True, drop_agents=True, v=v) for station,filled_df in filled_dfs.items()}

    return {station:df_to_agents_dict(stations_df, drop_stations=True, drop_agents=True, v=v) for station,stations_df in stations_dfs.items()}

# === TRAFFIC ===

def preprocess_traffic_dataset(traffic_folder, locations = None, radius=1, v=1):
    '''
	returns a list of dataframe with data sourronding each location.
    '''
    def map_values(x):
        if x == -0.01:
            return 0
        return x
    
    if not locations:
        locations = [
            ('GIARDINI MARGHERITA','44.482671138769533,11.35406170088398'),
            ('PORTA SAN FELICE','44.499059983334519,11.327526717440112'), 
            ('VIA CHIARINI','44.499134335170289,11.285089594971216') 
        ]

    if v>0: print('Merging readings files...')
    df = merge_csv_to_dataframe(os.path.join(traffic_folder, 'readings'), v=v, sep=';')
    if v>0: print('Merging accuracies files...')
    accuracies_df = merge_csv_to_dataframe(os.path.join(traffic_folder, 'accuracies'), v=v, sep=';')

    df = df.drop(columns=['id_uni','Livello','tipologia','codice','codice arco','codice via','Nome via','Nodo da','Nodo a','ordinanza','stato','codimpsem','direzione','angolo','longitudine','latitudine','ID_univoco_stazione_spira','Giorno della settimana','giorno settimana'])
    df = df.dropna()
    accuracies_df = convert_percentage_to_number(accuracies_df)
    accuracies_df = accuracies_df.map(map_values)
    common_cols = df.columns.intersection(accuracies_df.columns).tolist()

    accurate_traffic_df = apply_accuracy_df(df[common_cols],accuracies_df[common_cols],max_multiplier=15, half_multiplier=2).reset_index(drop=True)
    df = df.reset_index(drop=True)
    for col in list(set(df.columns) - set(common_cols)): # add back readings columns
        accurate_traffic_df[col] = df[col]
    accurate_traffic_df = accurate_traffic_df.set_index('data')

    dfs={name:divide_df_by_location(accurate_traffic_df, geopoint=center, radius=radius, name=name, v=v) for name,center in locations}
    
    return dfs

# === WEATHER ===
def preprocess_weather_dataset(weather_folder, v=1):
    '''
	returns a single df.
    '''
    if v>0: print('Merging weather files...')
    df = merge_csv_to_dataframe(weather_folder, v=v).rename(columns={'PragaTime':'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df= df.drop(columns=['W_VEC_INT','ET0'])
    df = df.set_index('Date')
    return df