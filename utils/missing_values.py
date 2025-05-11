from utils import *

import pandas as pd
import numpy as np


# === RESAMPLING ===
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
    if 'Unit' in df.columns:
        res['Unit'] = res['Unit'].fillna(df.iloc[0]['Unit'])
    return res


# === MISSING VALUES FILLING ===
def fill_missing_dates_with_model(df, settings:dict):
    pass

def fill_missing_dates(df: pd.DataFrame, method: str = 'ffill', column=None, model_settings:dict=None):
    '''
	fills all the nans using the method passed. If using mean filling,
    `column` specifies the column to use for the average.
    '''
    if method == 'mfill':
        return df.fillna(df[column].mean())
    
    elif method == 'model':
        return fill_missing_dates_with_model(df, model_settings)
    
    elif hasattr(df, method):  
        return getattr(df, method)()
    
    else:
        raise ValueError(f"Invalid method: {method}")
    
def fill_missing_dates_on_column_value(df, column, column_to_fill, mode, model_settings:dict=None, v=1):
    '''
    Specify the `method` to use to fill `Nan` values.

    Apply filling indipendently on the unique values of `column`.

    Usually `method` is one of ['ffill', 'bfill', 'interpolate', 'mfill']
    '''
    s=df['Station'].iloc[0]
    aux=df.copy()
    if v>0: print(f'Filling station "{s}..."')
    for val in np.unique(aux[column]):
        if v>1: print(f'- Filling {val} values')
        mask = aux[column] == val
        aux[mask] = fill_missing_dates(aux[mask], method=mode, column=column_to_fill)
    return aux
