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
def df_to_agents_dict(df, column='Agent', drop_stations=False, drop_agents=True, drop_duplicates=True, v=1):
    s=df['Station'].iloc[0]
    if v>0: print(f'Splitting station "{s}"...')
    agents_dict = {}
    for agent in np.unique(df[column]):
        agent_df = df[df[column] == agent].sort_values(by='Date')
        if drop_stations:
            agent_df = agent_df.drop(columns='Station')
        if drop_agents:
            agent_df = agent_df.drop(columns='Agent')
        if drop_duplicates:
            agent_df = agent_df[~agent_df.index.duplicated(keep='first')]
        
        agents_dict[agent] = agent_df
    
    return agents_dict


def join_datasets(*args, dropna=True):
    """
    Merges multiple DataFrames using their index.
    """
    merged_df = args[0]
    
    for df in args[1:]:
        merged_df = merged_df.join(df, how='left')
    
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
        #TODO
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
        df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)].copy()
    elif start_date:
        df = df[(df[date_column] >= start_date)].copy()
    elif end_date:
        df = df[(df[date_column] <= end_date)].copy()
    
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


def plot_missing_value_gaps(df, date_column='Date', value_col='Value', start_date=None, end_date=None, surrounding_vals=10, min_gap=2):
    df=df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    if start_date and end_date:
        df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)].copy()
    elif start_date:
        df = df[(df[date_column] >= start_date)].copy()
    elif end_date:
        df = df[(df[date_column] <= end_date)].copy()
    df.sort_values(by=date_column, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    missing_indices = df[df[value_col].isna()].index
    gaps = []
    current_gap = []
    
    for idx in missing_indices:
        if not current_gap or idx == current_gap[-1] + 1:
            current_gap.append(idx)
        else:
            if len(current_gap) >= min_gap:
                gaps.append(current_gap)
            current_gap = [idx]
    if len(current_gap) >= min_gap:
        gaps.append(current_gap)
    
    dataframes = []
    
    for gap in gaps:
        start_idx, end_idx = gap[0], gap[-1]
        selected_indices = list(range(max(0, start_idx - surrounding_vals), min(len(df), end_idx + 1 + surrounding_vals)))
        df_selected = df.loc[selected_indices].copy()
        df_selected.reset_index(drop=True, inplace=True)
        df_selected["Index"] = range(len(df_selected))
        dataframes.append(df_selected)
    
    num_plots = len(dataframes)
    cols = min(3, num_plots)
    rows = (num_plots // cols) + (num_plots % cols > 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)[:num_plots]
    
    for i, (ax, gap_df) in enumerate(zip(axes, dataframes)):
        l=len(gap_df["Index"])
        ax.plot(gap_df["Index"], gap_df[value_col], marker="o", linestyle="-")
        ax.set_title(f'Missing {l - 2*surrounding_vals} consecutive values')
        xticks = [surrounding_vals-1, l-surrounding_vals]
        ax.set_xticks(xticks)
        ax.set_xticklabels(gap_df[date_column].iloc[xticks].dt.strftime('%Y-%m-%d %H:%M')
                           , rotation=10, ha='right'
                           )

    plt.tight_layout()
    plt.show()

def plot_gp(target=None, pred=None, std=None, samples=None,
        target_samples=None, figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(target.index, target, c='black', label='target')
    if pred is not None:
        plt.plot(pred.index, pred, c='tab:blue',
                label='predictions')
    if std is not None:
        plt.fill_between(pred.index, pred-1.96*std, pred+1.96*std,
                alpha=.3, fc='tab:blue', ec='None',
                label='95% C.I.')
    # Add scatter plots
    if samples is not None:
        try:
            x = samples.index
            y = samples.values
        except AttributeError:
            x = samples[0]
            y = samples[1]
        plt.scatter(x, y, color='tab:orange',
              label='samples', marker='x')
    if target_samples is not None:
        try:
            x = target_samples.index
            y = target_samples.values
        except AttributeError:
            x = target_samples[0]
            y = target_samples[1]
        plt.scatter(x, y,
                color='black', label='target', s=5)
    plt.legend()
    plt.grid(':')
    plt.tight_layout()


def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=(20,5), s=4):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(labels))
    plt.tight_layout()

    