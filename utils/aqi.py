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


def compute_breakpoints_AQI(value, agent):
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
                    aqi['AQI'] = aqi[value_column].apply(lambda x: compute_breakpoints_AQI(x, agent))
                else: 
                    aqi['AQI'] = aqi[value_column] / limit * 100
                
                aqi = aqi.set_index(aqi.index)  # Ensure the index is used
                aqi = aqi.resample('1h').ffill().bfill()
                aqi = aqi.drop(columns=[value_column])
            else:
                aqi = pd.DataFrame()
        elif agent == 'O3' and breakpoints:
            aqi['AQI'] = aqi[value_column].apply(lambda x: compute_breakpoints_AQI(x, agent))
            aqi = aqi.drop(columns=[value_column])
        else:
            aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])

    if period == 'day':
        if agent in ('PM2.5', 'PM10'):
            if include_hourly_pm:
                if breakpoints:
                    aqi['AQI'] = aqi[value_column].apply(lambda x: compute_breakpoints_AQI(x, agent))
                else: 
                    aqi['AQI'] = aqi[value_column] / limit * 100
                
            aqi = aqi.drop(columns=[value_column])

        # Average agents
        if agent in ('NO2', 'C6H6'):
            aqi = aqi.resample('D').mean()
            aqi['AQI'] = aqi[value_column] / limit * 100
            aqi = aqi.drop(columns=[value_column])

        # maximum rolling average on 8 hours
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


def plot_AQI(station_AQI, title='', categories=None, figsize=(20, 5), s=None, e=None, y_aspect='equal'):
    if isinstance(station_AQI, dict):
        station_AQI = [station_AQI]
    if isinstance(title, str):
        title = [title] * len(station_AQI)
    elif isinstance(title, dict):
        title = [title]

    hour = (station_AQI[0].index[1] - station_AQI[0].index[0]) < pd.Timedelta(hours=23)

    consistent_agents = set()
    for aqi in station_AQI[:2]:
        consistent_agents.update(aqi['agent'].unique())

    base_cmap = plt.get_cmap('Dark2')
    base_colors = base_cmap(np.linspace(0, 1, len(consistent_agents)))
    consistent_color_map = dict(zip(sorted(consistent_agents), base_colors))  # sorted for deterministic color assignment

    num_plots = len(station_AQI)
    fig, axs = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]

    for i, (ax, aqi, titl) in enumerate(zip(axs, station_AQI, title)):
        if s and e:
            aqi_to_plot = aqi.copy()[(aqi.index >= s) & (aqi.index <= e)]
        elif s:
            aqi_to_plot = aqi.copy()[(aqi.index >= s)]
        elif e:
            aqi_to_plot = aqi.copy()[(aqi.index <= e)]
        else:
            aqi_to_plot = aqi.copy()

        is_diff_AQI = np.sum(aqi_to_plot['AQI'] < 0)
        hour_bar_width = 0.03 if hour else 0.9

        if i < 2: # aqi plots
            for agent, segment in aqi_to_plot.groupby('agent'):
                color = consistent_color_map.get(agent, 'gray')
                ax.bar(segment.index, segment['AQI'], label=agent, color=color, width=hour_bar_width)
            if categories:
                ymin, ymax = ax.get_ylim()
                names = list(categories.keys())
                for i, category in enumerate(names,0):
                    value = categories[names[i]]
                    if ymin <= value <= ymax:
                        ax.axhline(y=value, color='black', linestyle='--', linewidth=0.8)
                        ax.text(
                            aqi_to_plot.index.max(), 
                            value + (ymax - ymin) * 0.01,
                            f'{category} (> {value})',
                            color='black', fontsize=8, verticalalignment='bottom',
                            ha='left', va='center',
                            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
                        )

        else: # diff plot
            cmap = plt.get_cmap('tab20b' if is_diff_AQI else 'Dark2')
            unique_agents = aqi_to_plot['agent'].unique()
            colors = cmap(np.linspace(0, 1, len(unique_agents)))
            for color, (agent, segment) in zip(colors, aqi_to_plot.groupby('agent')):
                ax.bar(segment.index, segment['AQI'], label=agent, color=color, width=hour_bar_width)

        ax.set_title(titl)
        if y_aspect=='equal' and not is_diff_AQI and len(station_AQI) > 2:
            ymax1, ymax2 = station_AQI[0].loc[s:e]['AQI'].max(), station_AQI[1].loc[s:e]['AQI'].max()
            ylims = [0, max(ymax1, ymax2) + 5]
            ax.set_ylim(ylims)
        ax.legend(loc='upper left')

    plt.tight_layout()
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


AQI_CATEGORIES = {
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
    return list(categories.keys())[0]


def print_AQI_category_comparison(pred_AQI, true_AQI, categories, figsize=(6,6)):
    pred_AQI = pred_AQI['AQI'].copy().dropna()
    true_AQI = true_AQI['AQI'].copy().dropna()
    pred_categories = pred_AQI.apply(map_category, categories=categories)
    true_categories = true_AQI.apply(map_category, categories=categories)

    cm = confusion_matrix(true_categories, pred_categories, labels=list(categories.keys()), normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(categories.keys()))
    
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    plt.title(f'AQI Categories Predictions ({cm.trace():.2f}% correct)')
    plt.show()


def compute_AQI_and_show_analysis(predictions_dict, true_values_dict, categories=AQI_CATEGORIES):
    pred_AQIs_daily = {s: {agent: get_AQI(predictions_dict[s][agent]['predictions'], agent=agent, period='day', value_column='Agent_value',) for agent in predictions_dict[s].keys()} for s in predictions_dict}
    pred_AQI_daily = merge_AQIs(pred_AQIs_daily, period='day')
    pred_AQIs_hourly = {s: {agent: get_AQI(predictions_dict[s][agent]['predictions'],agent=agent,period='hour', value_column='Agent_value',include_hourly_pm=False) for agent in predictions_dict[s].keys()} for s in predictions_dict}
    pred_AQI_hourly = merge_AQIs(pred_AQIs_hourly, period='hour')
    true_AQIs_daily = {s: {agent: get_AQI(true_values_dict[s][agent]['y'], agent=agent, period='day', value_column='Agent_value',) for agent in true_values_dict[s].keys()} for s in true_values_dict}
    true_AQI_daily = merge_AQIs(true_AQIs_daily, period='day')
    true_AQIs_hourly = {s: {agent: get_AQI(true_values_dict[s][agent]['y'],agent=agent,period='hour', value_column='Agent_value',include_hourly_pm=False) for agent in true_values_dict[s].keys()} for s in true_values_dict}
    true_AQI_hourly = merge_AQIs(true_AQIs_hourly, period='hour')
    AQI_diff_daily = {}
    AQI_diff_hourly = {}
    for station in true_AQI_hourly:
        AQI_diff_daily[station] = AQI_difference(pred_AQI_daily[station], true_AQI_daily[station])
        AQI_diff_hourly[station] = AQI_difference(pred_AQI_hourly[station], true_AQI_hourly[station])
    for station in pred_AQI_hourly:
        print('=============================================')
        print(f'Station {station} Hourly')
        print('=============================================')
        print_AQI_category_comparison(pred_AQI_hourly[station],true_AQI_hourly[station], categories=categories)
        print('---------------------------------------------')
        print_AQI_difference_metrics(AQI_diff_hourly[station])
        print('=============================================')
    for station in pred_AQI_daily:
        print('=============================================')
        print(f'Station {station} Daily')
        print('=============================================')
        print_AQI_category_comparison(pred_AQI_daily[station],true_AQI_daily[station], categories=categories)
        print('---------------------------------------------')
        print_AQI_difference_metrics(AQI_diff_daily[station])
        print('=============================================')


