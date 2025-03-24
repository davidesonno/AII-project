import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_AQI(df:pd.DataFrame, agent, limit, period, value_column, date_column:str='Date'):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    aqi = df.copy()
    if period == 'hour':
        if agent in ('PM2.5','PM10'):
            # aqi['AQI'] = aqi[value_column]/limit*100
            aqi['AQI'] = 0
            aqi[date_column] = pd.to_datetime(df[date_column])
            aqi = aqi.set_index(date_column)
            aqi = aqi.resample('1h').ffill().bfill()
            aqi = aqi.drop(columns=[value_column])

        else:    
            aqi['AQI'] = aqi[value_column]/limit*100
            aqi[date_column] = pd.to_datetime(df[date_column])
            aqi = aqi.set_index(date_column)
            aqi = aqi.drop(columns=[value_column])

    if period == 'day':
        if agent in ('PM2.5','PM10'):
            aqi['AQI'] = aqi[value_column]/limit*100
            aqi = aqi.drop(columns=[value_column])
            # aqi[date_column] = pd.to_datetime(aqi[date_column])
            aqi = aqi.set_index(date_column)

        # average agents
        if agent in ('O3','NO2','C6H6'):
            aqi = aqi.resample('D',on=date_column).mean()
            aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])
            # aqi[date_column] = pd.to_datetime(aqi[date_column])

        # MaSsImA dElLe MeDiE mObIlI sU 8 oRe
        if agent == 'CO':
            aqi = aqi.set_index(date_column).resample('1h').max()

            def rolling_average(day):
                day['AQI'] = day[value_column].rolling(window=8, min_periods=8).mean()
                return day['AQI'].max()

            aqi = aqi.groupby(aqi.index.date).apply(rolling_average).reset_index(name='AQI')
            aqi['AQI'] = aqi['AQI'] / limit * 100
            aqi = aqi.rename(columns={'index':date_column})
            aqi[date_column] = pd.to_datetime(aqi[date_column])
            aqi = aqi.set_index(date_column)

    return aqi 


def plot_AQI(station_dict, period, figsize=(20,5)):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    hour = period == 'hour'
    freq = '1h' if hour else 'D'
    
    agents = list(station_dict.keys())
    aqi_df = station_dict[agents[0]]
    aqi_df['agent'] = agents[0]
    for key in agents[1:]:
        aux = station_dict[key].copy()
        aux['agent'] = key
        aqi_df = pd.concat([aqi_df, aux])

    # aqi_df = aqi_df.groupby(aqi_df.index).max()
    aqi_df = aqi_df.fillna(-np.inf)
    aqi_df = aqi_df.sort_values(['AQI'])
    aqi_df = aqi_df.reset_index()
    aqi_df = aqi_df.drop_duplicates(subset='Date', keep='last')
    aqi_df = aqi_df.set_index('Date')
    aqi_df = aqi_df.resample(freq).max()
    aqi_df['agent'] = aqi_df['agent'].fillna('missing')
    # aqi_df['AQI'] = aqi_df['AQI'].fillna(150)

    agents = aqi_df['agent'].unique()
    # display(aqi_df)
    plt.figure(figsize=figsize)
    for _, segment in aqi_df.groupby('agent'):
        plt.bar(segment.index, segment['AQI'], label=f'{segment["agent"].iloc[0]}',width=0.03 if hour else 0.9)

    plt.legend(loc='upper left')
    plt.show()