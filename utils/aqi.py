from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .training import display_metric_scores

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
        'LO':[0, 125, 165, 205, 405], # I had to add a lower bound
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
    'PM2.5': 40,
    'PM10': 50, 
    'CO': 10, 
    'O3': 120, 
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

def get_AQI(df: pd.DataFrame, agent, period, value_column, limit=None, breakpoints=False, include_hourly_pm=True):
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
        if agent in ('NO2', 'C6H6'):
            aqi = aqi.resample('D').mean()
            if agent == 'O3' and breakpoints: # TODO eventually move it boelow
                aqi['AQI'] = aqi[value_column].apply(lambda x: compute_PM(x, agent))
            else:
                aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])

        # MaSsImA dElLe MeDiE mObIlI sU 8 oRe
        if agent in ('O3','CO'):
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


def merge_AQIs(AQI_dict, period):
    if period not in ('day', 'hour'): 
        return ValueError(f'Period can only be `day` or `hour`. Got {period} instead')
    hour = period == 'hour'
    freq = '1h' if hour else 'D'

    station_AQIs = {}
    for station in AQI_dict.keys():
        agents = list(AQI_dict[station].keys())
        aqi_df = AQI_dict[station][agents[0]].copy()
        
        aqi_df.loc[:, 'agent'] = agents[0]
        for key in agents[1:]:
            aux = AQI_dict[station][key].copy()
            if(len(aux) > 0): # if the aqi is hourly but pm is excluded the pm df is empty
                aux.loc[:, 'agent'] = key
                aqi_df = pd.concat([aqi_df, aux])

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

        station_AQIs[station] = aqi_df
        
    return station_AQIs

def AQI_difference(pred, true):
    diff = pd.DataFrame(pred['AQI'] - true['AQI'])
    diff['agent'] = (pred['agent'] == true['agent']
                        ).astype(int).apply(lambda x: 'Same agent' if x else 'Different agents')
    
    return diff

def print_AQI_difference_metrics(AQI_diff):
    positive_avg = AQI_diff[AQI_diff['AQI'] > 0]['AQI'].mean()
    negative_avg = AQI_diff[AQI_diff['AQI'] < 0]['AQI'].mean()
    absolute_mean = AQI_diff['AQI'].abs().mean()
    agent_percentage = AQI_diff['agent'].value_counts(normalize=True) * 100
    positive_percentage = (AQI_diff['AQI'] > 0).mean() * 100
    negative_percentage = (AQI_diff['AQI'] < 0).mean() * 100

    print(f"> Absolute Mean AQI Difference: {absolute_mean:.2f}")
    print(f"> Positive Average AQI Difference: {positive_avg:.2f}")
    print(f"> Negative Average AQI Difference: {negative_avg:.2f}")
    print(f"> Percentage of Positive Differences: {positive_percentage:.2f}%")
    print(f"> Percentage of Negative Differences: {negative_percentage:.2f}%")
    print("> Percentage of Agent Values:")
    display_metric_scores(agent_percentage.to_dict(), start='   ', round=2)


categories = {
    'Good': 0,
    'Moderate': 50,
    'Poor': 100,
    'Very Poor': 150,
    'Severe': 200,
}

def map_category(value, categories):
    for category, up_bound in reversed(list(categories.items())):
        if value > up_bound:
            return category

def print_AQI_category_comparison(pred_AQI, true_AQI, categories):
    pred_categories = pred_AQI['AQI'].copy().dropna().apply(map_category, categories=categories)
    true_categories = true_AQI['AQI'].copy().dropna().apply(map_category, categories=categories)
    
    cm = confusion_matrix(true_categories, pred_categories, labels=list(categories.keys()),normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(categories.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'AQI Categories Predictions ({cm.trace():.2f}% correct)')
    plt.show()