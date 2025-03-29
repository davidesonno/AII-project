from utils import *
from .distances import *
from .date_encoding import *
from .missing_values import *

import pandas as pd
import numpy as np
import os
from datetime import timedelta


# === UTILITIES ===
def mean_vector_direction(vectors:pd.Series):
    '''
	Performs the average direction of a given `pd.Series` of angles, in degree.

    Returns the angle in degrees.
    '''
    x = np.cos(np.radians(vectors))
    y = np.sin(np.radians(vectors))
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_direction = np.degrees(np.arctan2(mean_y, mean_x))
    mean_direction = (mean_direction + 360) % 360  # Ensure [0, 360] range
        
    return mean_direction

DEFAULT_AGGREGATIONS = {
    # 'W_SCAL_INT': 'mean', # 'mean' by default
    'PREC': 'sum',
    'RAD': 'sum',
    'W_VEC_DIR': mean_vector_direction,
    'LEAFW': lambda x: (x > 0).sum()
}
DEFAULT_MAX_MIN_COLUMNS = ['TAVG', 'RHAVG']

def transform_weather_to_daily_df(df:pd.DataFrame, aggregations={}, max_min_columns=[], keep_splitted_columns=True, **kwargs):
    '''
	The `aggregations` dictionary is passed to `df.agg()` and should therefore have the proper syntax.
    '''
    agg = {col:'mean' for col in df.columns} # by default set the aggregation to sum. The columns might be named differently

    for feature, aggfunction in aggregations.items():
        if feature in agg: # if the feature exists in the df
            agg[feature] = aggfunction

    daily_df = df.copy().resample('D').agg(agg)

    max_min_agg = {}
    for col in max_min_columns:
        if col in df.columns: 
            max_min_agg[f"{col}_MAX"] = (col, 'max')
            max_min_agg[f"{col}_MIN"] = (col, 'min')

    if max_min_agg:
        # if you aggregate like df.agg(y=(col,aggfunction)) you transform the df to only have the new column 'y' with that aggregation.
        # We have max_min_agg = {
        # 'TAVG_MAX': ('TAVG', 'max'),
        # 'TAVG_MIN': ('TAVG', 'min'),
        # 'RHAVG_MAX': ('RHAVG', 'max'),
        # 'RHAVG_MIN': ('RHAVG', 'min')}
        # In our case, we do df.agg(TAVG_MAX=('TAVG', 'max'),TAVG_MIN=...) to have the 4 new columns, and then we join the aggregated dataframes
        df_max_min = df.copy().resample('D').agg(**max_min_agg)
        daily_df = daily_df.join(df_max_min)

        if not keep_splitted_columns:
            daily_df = daily_df.drop(columns=[col for col in max_min_columns if col in df.columns])

    return daily_df

def transform_traffic_to_daily_df(df:pd.DataFrame, bin_size=0, offset=0, dropna=True, **kwargs):
    '''
	`bin_size` specifies how many hours to group and add as features.
    If the value is 0, the whole day is averaged (same result if >= 24).
    '''
    # TODO verificare se i nomi delle colonne nuovi danno fastidio da qualche parte
    # non so se tipo ci sta qualcosa che accede specificatamente 'Traffic_value'~
    if 24%bin_size != 0:
        raise ValueError('bin_size should be a divisor of 24.')
    if bin_size == 0 or bin_size >= 24:
        return df.copy().resample('D').mean().rename({'Value':'Traffic_value'})
    
    df_resampled = df.copy()
    df_resampled['bin'] = df.index.hour
    df_resampled['bin'] = df_resampled['bin'].apply(lambda x: ((x-offset)%24)//bin_size)
    df_resampled['aux'] = (df.index-timedelta(hours=offset)).date

    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.drop(columns=['Date']).rename(columns={'aux':"Date"})
    df_resampled = df_resampled.set_index('Date')

    df_resampled = df_resampled.groupby(by=[df_resampled.index,'bin']).sum()
    df_resampled = df_resampled.unstack(level="bin")

    df_resampled.columns = [f"Traffic_{(i*bin_size+offset)%24}-{((i+1)*bin_size+offset-1)%23}" for i in range(24//bin_size)]

    if dropna:
        return df_resampled.dropna()
    else:
        return df_resampled

# === PREPROCESSING ===
def read_and_preprocess_dataset(datasets_folder, dataset, resample=False, fill_method='mfill', radius=1, v=1):
    '''
	`v=0` stops any output prints.
	'''
    match dataset:
        case 'pollution':
            return preprocess_pollution_dataset(os.path.join(datasets_folder,'pollution/pollution.csv'), resample=resample, fill_method=fill_method, v=v)
            
        case 'traffic':
            return preprocess_traffic_dataset(os.path.join(datasets_folder,'traffic'), radius=radius, v=v)

        case 'weather':
            return preprocess_weather_dataset(os.path.join(datasets_folder,'weather'), v=v)
        
        case _:
            raise ValueError(f"Unknown dataset '{dataset}'. Only 'pollution', 'traffic' or 'weather' available")

def prepare_station_data_for_training(
        station_pollution_dict:dict,
        station_traffic_df:pd.DataFrame,
        weather_df:pd.DataFrame,
        encoding_method = 'full-sin-cos',
        **kwargs,
):
    '''
	Returns a dict mapping agents to their dataframe.\\
    The dataframe is normalized and encoded with the given `encoding_method`
    '''
    merged_dict = {}
    for agent,agent_pollution_df in station_pollution_dict.items():
        # if the agent is daily (PM), resample traffic and weather df before merging.
        if agent in ('PM2.5','PM10'):
            daily_traffic_df = transform_traffic_to_daily_df(station_traffic_df.copy(), **kwargs)
            daily_weather_df = transform_weather_to_daily_df(weather_df.copy(), **kwargs)

        merged_dict[agent] = join_datasets(
            agent_pollution_df,
            daily_traffic_df if agent in ('PM2.5','PM10') else station_traffic_df, # else it got copied (i think)
            daily_weather_df if agent in ('PM2.5','PM10') else weather_df, # else it got copied (i think)
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