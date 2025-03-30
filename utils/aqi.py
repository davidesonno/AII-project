import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    #https://forum.airnowtech.org/t/the-aqi-equation/169
CONC = { 
    'PM2.5':{
        'LO':[0, 9.1, 35.5, 55.5, 125.5, 255.5],
        'HI':[9.0, 35.4, 55.4, 125.4, 225.4, 325.4]
        },
    'PM10':{
        'LO':[0, 55, 155, 255, 355, 425],
        'HI':[54, 154, 254, 354, 424, 604]
        },
    'O3':{
        'LO':[0, 125, 165, 205, 405], # I add to add a lower bound
        'HI':[124, 164, 204, 404, 604] # I also added a starting value here
        }
}
AQI ={
    'PM2.5':{
        'LO': [0, 51, 101, 151, 201, 301],
        'HI': [50, 100, 150, 200, 300, 500]
        },
    'PM10':{
        'LO': [0, 51, 101, 151, 201, 301],
        'HI': [50, 100, 150, 200, 300, 500]
        },
    'O3':{
        'LO': [0, 101, 151, 201, 301],
        'HI': [100, 150, 200, 300, 500]
        }
}

limits = {
    'PM2.5': 25,
    'PM10': 50, 
    'CO': 10, 
    'O3': 180, 
    'NO2': 200, 
    'C6H6': 5
}


def compute_PM(value, agent):
    if value > 0:
        try:
            conc_lo = max([x for x in CONC[agent]['LO'] if x <= value])
            conc_hi = min([x for x in CONC[agent]['HI'] if x >= value])
            aqi_lo = AQI[agent]['LO'][CONC[agent]['LO'].index(conc_lo)]
            aqi_hi = AQI[agent]['HI'][CONC[agent]['HI'].index(conc_hi)]
            aqi = (aqi_hi - aqi_lo) / (conc_hi - conc_lo) * (value - conc_lo) + aqi_lo
            # print(value)
            # print(conc_lo, conc_hi, aqi_lo, aqi_hi, aqi)
            return aqi
        except ValueError as e:
            print(f"Error with value: {value}, agent: {agent}")
            raise ValueError(e)
    else:
        # does this ever happen? if it is 0 why nan, should we keep 0?
        return np.nan

import pandas as pd

def get_AQI(df: pd.DataFrame, agent, period, value_column, limit=None, breakpoints=True, include_hourly_pm=True):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    if limit is None:
        limit = limits[agent]
    
    aqi = df.copy()
    
    if period == 'hour':
        if agent in ('PM2.5', 'PM10'):
            if include_hourly_pm:
                if breakpoints:
                    aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
                else: 
                    aqi['AQI'] = aqi[value_column] / limit * 100
                
                aqi = aqi.set_index(aqi.index)  # Ensure the index is used
                aqi = aqi.resample('1h').ffill().bfill()
                aqi = aqi.drop(columns=[value_column])
            else:
                aqi = pd.DataFrame()
        elif agent == 'O3' and breakpoints:
            aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
            aqi = aqi.drop(columns=[value_column])
        else:
            aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])

    if period == 'day':
        if agent in ('PM2.5', 'PM10'):
            if include_hourly_pm:
                if breakpoints:
                    aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
                else: 
                    aqi['AQI'] = aqi[value_column] / limit * 100
                
            aqi = aqi.drop(columns=[value_column])

        # Average agents
        if agent in ('O3', 'NO2', 'C6H6'):
            aqi = aqi.resample('D').mean()
            if agent == 'O3' and breakpoints:
                aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
            else:
                aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])

        # MaSsImA dElLe MeDiE mObIlI sU 8 oRe
        if agent == 'CO':
            aqi = aqi.resample('1h').max()

            def rolling_average(day):
                day['AQI'] = day[value_column].rolling(window=8, min_periods=8).mean()
                return day['AQI'].max()

            aqi = aqi.groupby(aqi.index.date).apply(rolling_average).reset_index(name='AQI')
            aqi['AQI'] = aqi['AQI'] / limit * 100
            aqi = aqi.rename(columns={'index': 'Date'})
            aqi['Date'] = pd.to_datetime(aqi['Date'])
            aqi = aqi.set_index('Date')

    return aqi


def plot_AQI(station_AQI, title ='',figsize=(20,5), s=None, e=None, ylims=None, cmap=None):   
    if isinstance(station_AQI,dict):
        station_AQI = [station_AQI]
    if isinstance(title,dict):
        title = [title]

    hour = (station_AQI[0].index[1] - station_AQI[0].index[0]) < pd.Timedelta(hours=23)

    for aqi,titl in zip(station_AQI,title):
        if s and e:
            aqi_to_plot = aqi.copy()[(aqi.index>=s)&(aqi.index<=e)]
        elif s:
            aqi_to_plot = aqi.copy()[(aqi.index>=s)]
        elif e:
            aqi_to_plot = aqi.copy()[(aqi.index<=e)]

        plt.figure(figsize=figsize)
        is_diff_AQI = np.sum(aqi_to_plot['AQI'] < 0) # if there are negative values is probably a difference aqi
                
        cmap = plt.get_cmap('tab20b' if is_diff_AQI else 'Dark2')
        colors = cmap(np.linspace(0, 1, len(aqi_to_plot['agent'].unique())))

        for (i, (_, segment)) in enumerate(aqi_to_plot.groupby('agent')):
            plt.bar(segment.index, segment['AQI'], label=f'{segment["agent"].iloc[0]}', color=colors[i], width=0.03 if hour else 0.9)
        if ylims and not (is_diff_AQI):
            plt.ylim(ylims)
        plt.title(titl)
        plt.legend(loc='upper left')
        plt.show()

