from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === FILE READING ==
def merge_csv_to_dataframe(input_folder, v=1, **kwargs):
    """Merge all CSV files in a folder into a single pandas DataFrame."""
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    dataframes = [pd.read_csv(os.path.join(input_folder, file), **kwargs) for file in csv_files]
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    if v>0:print(f"Merged {len(csv_files)} CSV files")
    
    return merged_df


# === UTILITIES ===
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
    # res['Unit'] = res['Unit'].fillna(df.iloc[0]['Unit']) # I dropped it earlier because it is useless
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

def df_to_agents_dict(df, column='Agent', drop_stations=False, drop_agents=True, v=1):
    s=df['Station'].iloc[0]
    if v>0: print(f'Splitting station "{s}"...')
    agents_dict = {}
    for agent in np.unique(df[column]):
        agent_dict = df[df[column] == agent].sort_values(by='Date')
        if drop_stations:
            agent_dict = agent_dict.drop(columns='Station')
        if drop_agents:
            agent_dict = agent_dict.drop(columns='Agent')
        
        agents_dict[agent] = agent_dict
    
    return agents_dict


def merge_datasets(*args, on, dropna=True):
    """
    Merges multiple DataFrames on the specified 'on' column.
    """
    merged_df = args[0]
    
    for df in args[1:]:
        merged_df = merged_df.merge(df, on=on, how='left')
    
    if dropna:
        merged_df = merged_df.dropna()
    return merged_df

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

def normalize_columns(df:pd.DataFrame, columns:list=[], skip:list=[]):
    '''
	Appies `MinMaxScaler` to the specified columns, skipping `skip` columns.
    
    If no columns are specified, all the columns are attempted to be scaled.
    '''
    scaler = MinMaxScaler()
    if not columns:
        columns = df.columns
    
    columns_to_normalize = [col for col in columns if col not in skip]
    aux = df.copy()
    aux[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    return aux

# === PLOTS ===
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




