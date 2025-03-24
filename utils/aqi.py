import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#https://forum.airnowtech.org/t/the-aqi-equation/169
CONC = { "PM25":{
            'LO':[0, 12.1, 35.5, 55.5, 150.5, 250.5],
            'HI':[12.0, 35.4, 55.4, 150.4, 250.4, 500.4]},
        'PM10':{
            'LO':[0, 55, 155, 255, 355, 425],
            'HI':[54, 154, 254, 354, 424, 604]}
}
AQI ={'LO': [0, 51, 101, 151, 201, 301],
      'HI': [50, 100, 150, 200, 300, 500]}

def compute_PM(value, agent):
    if value >0:
        conc_lo = max([x for x in CONC[agent]['LO'] if x <= value])
        conc_hi = min([x for x in CONC[agent]['HI'] if x >= value])
        aqi_lo = AQI['LO'][CONC[agent]['LO'].index(conc_lo)]
        aqi_hi = AQI['HI'][CONC[agent]['HI'].index(conc_hi)]
        aqi = (aqi_hi - aqi_lo) / (conc_hi - conc_lo) * (value - conc_lo) + aqi_lo
        # print(value)
        # print(conc_lo, conc_hi, aqi_lo, aqi_hi, aqi)
        return aqi
    else:
        return np.nan

def get_AQI(df:pd.DataFrame, agent, limit, period, value_column, date_column:str='Date'):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    aqi = df.copy()
    if period == 'hour':
        if agent in ('PM2.5','PM10'):
            agent = 'PM25' if agent == 'PM2.5' else agent
            aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
            # aqi['AQI'] = 0
            aqi = aqi.set_index(date_column)
            aqi = aqi.resample('1h').ffill().bfill()
            aqi = aqi.drop(columns=[value_column])
        else:    
            aqi['AQI'] = aqi[value_column]/limit*100
            aqi = aqi.drop(columns=[value_column])
            aqi = aqi.set_index(date_column)

    if period == 'day':
        if agent in ('PM2.5','PM10'):
            agent = 'PM25' if agent == 'PM2.5' else agent
            aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
            aqi = aqi.set_index(date_column)
            aqi = aqi.drop(columns=[value_column])

        # average agents
        if agent in ('O3','NO2','C6H6'):
            aqi = aqi.resample('D',on=date_column).mean()
            aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])

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

def plot_AQI(station_dict, period, title ='',figsize=(20,5), plotly=False):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    hour = period == 'hour'
    freq = '1h' if hour else 'D'
    
    agents = list(station_dict.keys())
    aqi_df = station_dict[agents[0]].copy()
    aqi_df.loc[:, 'agent'] = agents[0]
    for key in agents[1:]:
        aux = station_dict[key].copy()
        aux.loc[:, 'agent'] = key
        aqi_df = pd.concat([aqi_df, aux])

    aqi_df = aqi_df.fillna(-np.inf)
    aqi_df = aqi_df.sort_values(['AQI'])
    aqi_df = aqi_df.reset_index()
    aqi_df = aqi_df.drop_duplicates(subset='Date', keep='last')
    aqi_df = aqi_df.set_index('Date')
    aqi_df = aqi_df.resample(freq).max()
    aqi_df.loc[:, 'agent'] = aqi_df['agent'].fillna('missing')

    agents = aqi_df['agent'].unique()
    
    if plotly:
        import plotly.express as px
        fig = px.bar(aqi_df.reset_index(), x='Date', y='AQI', color='agent', title=title)
        fig.update_layout(height=400)
        fig.show()
    else:
        plt.figure(figsize=figsize)
        for _, segment in aqi_df.groupby('agent'):
            plt.bar(segment.index, segment['AQI'], label=f'{segment["agent"].iloc[0]}',width=0.03 if hour else 0.9)
        plt.title(title)
        plt.legend(loc='upper left')
        plt.show()