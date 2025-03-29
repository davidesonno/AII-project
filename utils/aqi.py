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


def plot_AQI(station_dict, period, title ='',figsize=(20,5), s=None, e=None, ylims=None, plotly=False):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    hour = period == 'hour'
    freq = '1h' if hour else 'D'
    
    agents = list(station_dict.keys())
    aqi_df = station_dict[agents[0]].copy()
    aqi_df.loc[:, 'agent'] = agents[0]
    for key in agents[1:]:
        aux = station_dict[key].copy()
        if(len(aux) > 0): # if the aqi is hourly but pm is excluded the pm df is empty
            aux.loc[:, 'agent'] = key
            aqi_df = pd.concat([aqi_df, aux])
    if s and e:
        aqi_df = aqi_df[(aqi_df.index>=s)&(aqi_df.index<=e)]
    elif s:
        aqi_df = aqi_df[(aqi_df.index>=s)]
    elif e:
        aqi_df = aqi_df[(aqi_df.index<=e)]

    if(len(aqi_df) == 0):
        print('Empty dataframe... check the date range!')
        return

    aqi_df = aqi_df.fillna(-np.inf)
    aqi_df = aqi_df.sort_values(['AQI']) # why?
    aqi_df = aqi_df.reset_index()
    aqi_df = aqi_df.drop_duplicates(subset='Date', keep='last')
    aqi_df = aqi_df.set_index('Date')
    aqi_df = aqi_df.resample(freq).max()
    aqi_df.loc[:, 'agent'] = aqi_df['agent'].fillna('missing')
    agents = aqi_df['agent'].unique()
    
    if plotly:
        import plotly.express as px
        if ylims: pass #idk how to
        fig = px.bar(aqi_df.reset_index(), x='Date', y='AQI', color='agent', title=title)
        fig.update_layout(height=400)
        fig.show()
    else:
        plt.figure(figsize=figsize)
        for _, segment in aqi_df.groupby('agent'):
            plt.bar(segment.index, segment['AQI'], label=f'{segment["agent"].iloc[0]}',width=0.03 if hour else 0.9)
        if ylims:
            plt.ylim(ylims)
        plt.title(title)
        plt.legend(loc='upper left')
        plt.show()