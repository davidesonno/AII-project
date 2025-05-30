import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import shap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from .training import create_sequences
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
# shap raises this warning
warnings.filterwarnings(
    "ignore",
    message="`tf.keras.backend.set_learning_phase` is deprecated",
    category=UserWarning,
    module="keras.backend"
)

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

def apply_accuracy_df(measurements_df, accuracies_df, add_verified_col=False):

    def accuracy_coeff(accuracies:pd.Series):
        return accuracies.where(accuracies == 0, 1 / accuracies)

    merged_df = measurements_df.merge(accuracies_df, on=['data', 'codice spira'], 
                                  suffixes=('_reading', '_accuracy'), how='left').fillna(1)

    if add_verified_col:
        merged_df['accurate'] = merged_df.iloc[:, 2 + len(measurements_df.columns[2:]) :].notna().all(axis=1)

    # Multiply only where accuracy is available, keeping original value if missing
    for col in measurements_df.columns[2:]:  # Skip 'data' and 'codice spira'
        merged_df[col + '_reading'] = (merged_df[col + '_reading']  * accuracy_coeff(merged_df[col + '_accuracy'])).astype(int)

    cols = ['data', 'codice spira', 'accurate'] if add_verified_col else ['data', 'codice spira']
    final_df = merged_df[cols + [col + '_reading' for col in measurements_df.columns[2:]]]
    final_df.columns = cols + list(measurements_df.columns[2:])  # Rename columns back

    return final_df

def normalize_columns(df:pd.DataFrame, columns:list=[], skip:list=[], return_dists:list=[], scaler=StandardScaler()):
    '''
	Applies `scaler` to the specified columns, skipping `skip` columns.
    
    If no skip columns are specified, all the columns are attempted to be scaled.
    If columns appear in `return_dist`, a dict with their mean and std will be returned.
    '''
    if not columns:
        columns = df.columns
    dist_dict = {}
    aux = df.copy()

    for col in return_dists:
        mean = aux[col].mean()
        std = aux[col].std()
        dist_dict[col] = {'mean':mean, 'std': std}
        
    columns_to_normalize = [ col for col in columns if (col not in skip) ]
    aux[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    if len(dist_dict) > 0:
        return aux, dist_dict
    return aux

def extract_ordered_features_by_shap(nested_data, data):
    result = {}
    for station, agents in nested_data.items():
        result[station] = {}
        for agent, info in agents.items():
            shap_values = info['shap_values']
            
            values = np.abs(shap_values).mean(axis=0)
            feature_names = data[station][agent]['x'].columns
            sorted_indices = np.argsort(-values)
            ordered_features = [feature_names[i] for i in sorted_indices]
            result[station][agent] = ordered_features

    return result

# === FEATURE IMPORTANCE ===
def neg_mae_scorer(model, X, y):
    try:
        y_pred = model.predict(X, verbose=0)
    except: # sklearn doesnt have `verbose`
        y_pred = model.predict(X)
    return -mean_absolute_error(y, y_pred)

def lstm_neg_mae_scorer(model, x, y, time_steps, n_features):
    x = x.reshape(-1, time_steps, n_features) # since that this is getting used by permutation_importance, it gets 2d data as input and needs to be reshaped
    y_pred = model.predict(x, verbose=0)
    return -mean_absolute_error(y, y_pred)
 
def compute_permutation_importances(models, test_sets, n_samples, n_repeats): 
    permutation_importances = {}
    for station in models:
        if station not in permutation_importances:   
            permutation_importances[station] = {}
        for agent in models[station].keys():
            model = models[station][agent]
            X_test = test_sets[station][agent]['x']
            y_test = test_sets[station][agent]['y']
            importances = {}
            scoring = neg_mae_scorer
            seqs = False
            input_shape=None

            if isinstance(model, tf.keras.models.Sequential):
                input_shape = model.input_shape
                
                if len(input_shape) == 2: #ffnn
                    pass
                    
                if len(input_shape) == 3: # seqs or cnn
                    seqs = True
                    time_steps = input_shape[1] 
                    n_features = input_shape[2] 
                    use_mask = isinstance(model.layers[0], tf.keras.layers.Masking)
                    X_test, y_test = create_sequences(test_sets[station][agent]['x'], test_sets[station][agent]['y'], time_steps, use_mask=use_mask)
                    y_test = y_test.to_numpy()
                    test_data_idx = np.random.choice(X_test.shape[0], size=n_samples, replace=False)
                    X_test = X_test[test_data_idx]
                    y_test = y_test[test_data_idx]
                    X_test = X_test.reshape(X_test.shape[0], -1) # permutation_importance does not allow for 3d data

                    scoring = lambda model, x, y: lstm_neg_mae_scorer(model, x, y, time_steps, n_features) # lambda because we had to pass time steps and num. features    
                
            else: # other models
                X_test = X_test.sample(n_samples, random_state=42)
                y_test = y_test.sample(n_samples, random_state=42)

            # print(f'Computing importance for {station} {agent}')

            # compute importances
            importances_result = permutation_importance(model,
                                                        X_test,
                                                        y_test,
                                                        scoring=scoring,
                                                        n_repeats=n_repeats,
                                                        random_state=42)
            
            for key in ('importances_mean', 'importances_std'): # we don't care about the full repetitions values (`importances`)
                val = importances_result[key]
                if seqs: # for the seqs we had to flatten the sequences, now we have one ts*n_feat. total features, so we go back to the original shape averaging
                    val = val.reshape(input_shape[1], input_shape[2])
                    val = val.mean(axis=0)
                importances[key] = val

            # save the importances
            permutation_importances[station][agent] = importances

    return permutation_importances

def compute_shap_values(models, train_sets, test_sets, n_samples, plot=False, stations=None, agents=None, figsize=None):
    '''
	Uses shap.KernelExplainer for deep models and shap.TreeExplainer for tree models.
    '''
    def model_predict(data): # used in kernel explainers
        return model.predict(data,verbose=0).reshape(-1) # with no reshaping it does not work
    
    n_rows = len(agents)
    n_cols = len(stations)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 6.5)
    if plot:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    results_shap_values = {}
    for station in models:
        if station not in results_shap_values:   
            results_shap_values[station] = {}
        for agent in models[station].keys():
            model = models[station][agent]
            if isinstance(model, tf.keras.models.Sequential):
                training_data = None
                test_data = None
                shap_values = None
                input_shape = model.input_shape

                if len(input_shape) == 2: # ffnn
                    training_data = train_sets[station][agent]['x'].sample(n_samples, random_state=42)
                    test_data = test_sets[station][agent]['x'].sample(n_samples, random_state=42)
                    
                    # kernel (has to predict and becomes so slow with many samples)
                    # explainer = shap.KernelExplainer(model_predict, training_data)
                    # shap_values = explainer.shap_values(test_data)

                    # gradient
                    explainer = shap.GradientExplainer(model, training_data)
                    shap_values = explainer.shap_values(np.array(test_data))
                    shap_values = shap_values.squeeze()

                if len(input_shape) == 3: # lstm or cnn
                    time_steps = input_shape[1] 
                    use_mask = isinstance(model.layers[0], tf.keras.layers.Masking)
                    training_data, _ = create_sequences(train_sets[station][agent]['x'], train_sets[station][agent]['y'], time_steps, use_mask=use_mask)
                    test_data, _ = create_sequences(test_sets[station][agent]['x'], test_sets[station][agent]['y'], time_steps, use_mask=use_mask)
                    # training_data, _ = create_sequences(train_sets[station][agent]['x'].iloc[:3*n_samples], train_sets[station][agent]['y'].iloc[:3*n_samples], time_steps, use_mask=use_mask)
                    # test_data, _ = create_sequences(test_sets[station][agent]['x'].iloc[:3*n_samples], test_sets[station][agent]['y'].iloc[:3*n_samples], time_steps, use_mask=use_mask)
                    
                    # if you sample before creating the sequences you would separate consecutive values. We might want to sample groups of time_steps elements
                    # and then create sequences with no sliding window. Or simply shrinken the dataset before, you need num_samples + time_steps -1 hours in total,
                    # so maybe keep 2/3 times num_samples before creating sequences <-- currenty doing this (missing the first `times_steps` -1 elements)
                    training_data_idx = np.random.choice(training_data.shape[0], size=n_samples, replace=False)
                    test_data_idx = np.random.choice(test_data.shape[0], size=n_samples, replace=False)
                    training_data = training_data[training_data_idx]
                    test_data = test_data[test_data_idx]

                    # gradient
                    explainer = shap.GradientExplainer(model, training_data)
                    shap_values = explainer.shap_values(test_data)
                    shap_values = np.mean(np.squeeze(shap_values), axis=1)
                    test_data = pd.DataFrame(test_data.mean(axis=1), columns=test_sets[station][agent]['x'].columns)

                    # kernel (might work if passed time averaged data)
                    # explainer = shap.KernelExplainer(model_predict, training_data)
                    # shap_values = explainer.shap_values(test_data.reshape(test_data.shape[0], -1))
                    # shap_values = explainer.shap_values(test_data)

            elif isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
                explainer = shap.TreeExplainer(model)
                test_data = test_sets[station][agent]['x'].sample(n_samples, random_state=42)
                shap_values = explainer.shap_values(test_data)

            # elif isinstance(model, (xgb.XGBRegressor,)):
            #     training_data = shap.sample(train_sets[station][agent]['x'], n_samples, random_state=42)
            #     test_data = test_sets[station][agent]['x'].sample(n_samples, random_state=42)
                
            #     f = lambda x: model.predict(x)
            #     explainer = shap.KernelExplainer(f, training_data, link='logit')
            #     shap_values = explainer.shap_values(test_data)

            # remove outliers (removes everything when the model is a xgboost)
            # z_scores = np.abs((shap_values - shap_values.mean(axis=0)) / shap_values.std(axis=0))
            # shap_values = shap_values[(z_scores < 3).all(axis=1)]
            # test_data = test_data[(z_scores < 3).all(axis=1)]
            
            # save the shap values
            results_shap_values[station][agent] = {
                'shap_values': shap_values,
                'explainer':  explainer
            }

            if plot:
                ax = axes[agents.index(agent)][stations.index(station)]
                tmp_fig, tmp_ax = plt.subplots(figsize=(5, 4))
                shap.summary_plot(shap_values,
                                test_data,
                                max_display=999,
                                show=False,
                                color_bar=True,
                                rng=42
                                )
                canvas = FigureCanvas(tmp_fig)
                canvas.draw()

                # Convert canvas to image
                image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                image = image.reshape(tmp_fig.canvas.get_width_height()[::-1] + (4,))

                # Show the image on the target subplot
                ax.imshow(image)
                ax.axis('off')
                ax.set_title(f'SHAP summary for {agent} at {station}')

                plt.close(tmp_fig)

    if plot:
        for station in stations:
            for agent in agents:
                if agent not in models[station]:
                    axes[agents.index(agent)][stations.index(station)].axis('off') # remove unused subplots
        plt.tight_layout()
        plt.show()

    return results_shap_values


# === PLOTS ===
def plot_permutation_importances(permutation_importances, stations, agents, test_sets, figsize=None):
    n_rows = len(agents)
    n_cols = len(stations)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 6.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for station in permutation_importances:
        for agent in permutation_importances[station]:
            X_test = test_sets[station][agent]['x']
            features = X_test.columns
            importances = permutation_importances[station][agent]
            ax = axes[agents.index(agent)][stations.index(station)]
            ax.barh(features, importances['importances_mean'], height=0.7, xerr=importances['importances_std'])
            ax.set_title(f'Permutation Importance for {agent} at {station}')
            ax.grid(linestyle=':')

    for station in stations:
        for agent in agents:
            if agent not in permutation_importances[station]:
                axes[agents.index(agent)][stations.index(station)].axis('off') # remove unused subplots

    plt.tight_layout()
    plt.show()


def plot_shap_values(plot_type, shap_values, stations, agents, test_sets, figsize=None):
    n_rows = len(agents)
    n_cols = len(stations)
    if figsize is None:
        figsize = (n_cols * 10, n_rows * 7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for station in stations:
        for agent in shap_values[station]:
            explanation = shap.Explanation(
                values=shap_values[station][agent]['shap_values'],
                data=test_sets[station][agent]['x'],
                feature_names=test_sets[station][agent]['x'].columns.tolist()
            )

            kwargs = {}
            
            if plot_type == shap.plots.heatmap:
                kwargs = {
                    'feature_values': explanation.abs.max(0), 
                    'plot_width': 10,
                }
            if plot_type == shap.plots.bar:
                clustering = shap.utils.hclust(test_sets[station][agent]['x'], test_sets[station][agent]['y'])
                kwargs = {
                    'clustering' :clustering, 
                    'clustering_cutoff' :0.5,
                }

            # Draw to canvas and convert to image
            ax = axes[agents.index(agent)][stations.index(station)]
            plot_type(
                shap_values=explanation,
                **kwargs,
                max_display=40,
                show=False,
                ax=ax
            )
            ax.set_title(f"{agent} @ {station}", fontsize=15)

    for station in stations:
        for agent in agents:
            if agent not in shap_values[station]:
                axes[agents.index(agent)][stations.index(station)].axis('off') # remove unused subplots
    plt.tight_layout()
    plt.show()


def plot_time_series(dfs, value_column, date_column, legends, start_date=None, end_date=None, max_rows=5000, downsample_factor=8, title=''):
    plt.figure(figsize=(40, 6))

    for df, legend in zip(dfs, legends):
        if start_date and end_date:
            if date_column is None: # use the index
                df = df[(df.index >= start_date) & (df.index < end_date)]
            else:
                df = df[(df[date_column] >= start_date) & (df[date_column] < end_date)]
        
        if date_column:
            df = df.sort_values(date_column)

        if len(df) > max_rows:
            df = df.iloc[::downsample_factor]  
        if date_column:
            plt.plot(df[date_column], df[value_column], label=legend, linewidth=1)
        else:
            plt.plot(df.index, df[value_column], label=legend, linewidth=1)

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


def plot_history(history, metrics=['loss']):
    plt.figure(figsize=(12, 5))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    linestyles = ['-', '--']
    title_lines = []

    for idx, metric in enumerate(metrics):
        train_metric = history.history.get(metric, [])
        val_metric = history.history.get(f'val_{metric}', [])
        epochs = range(1, len(train_metric) + 1)

        if train_metric:
            best_train_epoch = int(np.argmin(train_metric)) + 1
            best_train_value = train_metric[best_train_epoch - 1]
            title_lines.append(f'Best Train {metric}: {best_train_value:.4f} (epoch {best_train_epoch})')
            plt.plot(epochs, train_metric, label=f'Train {metric.capitalize()}', 
                     color=colors[idx % len(colors)], linestyle=linestyles[0], linewidth=2)
        
        if val_metric:
            best_val_epoch = int(np.argmin(val_metric)) + 1
            best_val_value = val_metric[best_val_epoch - 1]
            title_lines.append(f'Best Val {metric}: {best_val_value:.4f} (epoch {best_val_epoch})')
            plt.plot(epochs, val_metric, label=f'Val {metric.capitalize()}', 
                     color=colors[idx % len(colors)], linestyle=linestyles[1], linewidth=2)

    plt.title('\n'.join(['Training History'] + title_lines), fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(ticks=range(1, len(train_metric) + 1))
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_months_predictions(y_true, y_pred, dist_dict=None, metrics=None, yaspect='equal', title='', figsize=(17,8), show=True):
    plt.figure(figsize=figsize)
    ymin = min(min(y_true.values), min(y_pred.values))
    ymax = max(max(y_true.values), max(y_pred.values))
    if dist_dict:
        ymin = ymin * dist_dict['std'] + dist_dict['mean']
        ymax = ymax * dist_dict['std'] + dist_dict['mean']

    for month in range(1, 13):
        start = datetime(2024, month, 1)
        end = datetime(2024, month, 29 if month == 2 else 30 if month in [4, 6, 9, 11] else 31)
        
        plt.subplot(6, 2, month)
        plt.suptitle(title)
        yt = y_true[(y_true.index >= start) & (y_true.index <= end)]
        yp = y_pred[(y_pred.index >= start) & (y_pred.index <= end)]
        if dist_dict:
            yt = yt * dist_dict['std'] + dist_dict['mean']
            yp = yp * dist_dict['std'] + dist_dict['mean']
        plt.plot(yt, label='True values')
        plt.plot(yp, label='Predicted values')
        plt.title(f'{start.strftime("%B")}')
        plt.xticks([])
        if yaspect == 'equal':
            plt.ylim((ymin, ymax))
        if month==1:plt.legend()

    if show:
        plt.tight_layout()
        plt.show()

    if metrics:
        for m in metrics:
            score = m(y_true,y_pred)
            print(f'{m.__name__}: {score}')


def plot_year_predictions(y_true, y_pred, dist_dict=None, metrics=None, figsize=(15,3)):
    start = datetime(2024,1,1)
    end = datetime(2024,12,31)

    plt.figure(figsize=figsize)
    yt = y_true[(y_true.index >= start) & (y_true.index <= end)]
    yp = y_pred[(y_pred.index >= start) & (y_pred.index <= end)]
    if dist_dict:
        yt = yt * dist_dict['std'] + dist_dict['mean']
        yp = yp * dist_dict['std'] + dist_dict['mean']
    plt.plot(yt, label='True values')
    plt.plot(yp, label='Predicted values')
    plt.legend()
    plt.show()

    if metrics:
        for m in metrics:
            rfr_score = m(y_true,y_pred)
            print(f'{m.__name__}: {rfr_score}')

def plot_predictions(d_agent_values, dists=None, show_months=False):
    if show_months:
        for station, agents_true in d_agent_values['true'].items():
            agents_pred = d_agent_values['predictions'][station]
            num_agents = len(agents_true)

            for i, (agent, true_vals) in enumerate(agents_true.items()):
                pred_vals = agents_pred[agent]

                if dists:
                    mean = dists[agent][0]
                    std = dists[agent][1]
                    true_vals = true_vals.apply(lambda x:x* std + mean)
                    pred_vals = pred_vals.apply(lambda x:x* std + mean)

                plot_months_predictions(true_vals, pred_vals, title=f'{station}-{agent}')
    else: 
        for station, agents_true in d_agent_values['true'].items():
            agents_pred = d_agent_values['predictions'][station]
            num_agents = len(agents_true)

            plt.figure(figsize=(15, 4 * num_agents))
            plt.suptitle(f"Station {station} - True vs Predicted", fontsize=16)

            for i, (agent, true_vals) in enumerate(agents_true.items()):
                pred_vals = agents_pred[agent]

                if dists:
                    mean = dists[agent][0]
                    std = dists[agent][1]
                    true_vals = true_vals.apply(lambda x:x* std + mean)
                    pred_vals = pred_vals.apply(lambda x:x* std + mean)

                plt.subplot(num_agents, 1, i + 1)
                plt.title(f"Agent: {agent}")
                plt.plot(true_vals, label='True', marker='o')
                plt.plot(pred_vals, label='Pred', marker='x')
                plt.ylabel("Value")
                plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

