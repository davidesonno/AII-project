import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import random
import os
import xgboost as xgb
import joblib


# === TRAINING UTILITIES ===

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
def create_train_test(df, split_date, y):
    train = df[df.index <= split_date]
    test = df[df.index > split_date]

    x_train = train.drop(columns=[y])
    x_test = test.drop(columns=[y])
    aux=df.columns
    y_train = train.drop(columns=[col for col in aux if col !=y])
    y_test = test.drop(columns=[col for col in aux if col !=y])

    # Drop duplicated indexes, keeping the first occurrence
    x_train = x_train[~x_train.index.duplicated(keep='first')]
    x_test = x_test[~x_test.index.duplicated(keep='first')]
    y_train = y_train[~y_train.index.duplicated(keep='first')]
    y_test = y_test[~y_test.index.duplicated(keep='first')]

    return x_train,y_train,x_test,y_test

# === METRICS ===
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


def log_cosh_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))

def huber(y_true,y_pred):
    return tf.keras.losses.Huber(delta=0.2)(y_true,y_pred).numpy()

METRICS = [root_mean_squared_error, r2_score, mean_absolute_error, huber]

# === METRICS UTILITIES ===
def display_metric_scores(metric_dict, start='', round=None):
    for metric, score in metric_dict.items():
        if round:
            print(f'{start}- {metric}: {np.round(score,round)}')
        else:
            print(f'{start}- {metric}: {score}')

def update_metrics(old_results, true_values, metrics):
    '''
    Computes new metric values for the predicted data.

    - If the value of the agent key is a dict containing a dict that maps model name
    to the predictions and metrics, update the metrics dict.
    - If the value of the agent key are the predictions, returns a new dict with 
    the scores of the station-agent.
    '''
    new_results = copy.deepcopy(old_results)
    
    for station, station_results in new_results.items():
        for agent, agent_results in station_results.items():
            if isinstance(agent_results, dict):
                for model, model_results in agent_results.items():
                    metric_scores = model_results['metric_scores']
                    predictions = model_results['predictions']
                    for m in metrics:
                        name = 'metric'
                        try:
                            name = m.__name__
                        except: 
                            name = type(m).__name__
                            pass
                        if name not in metric_scores:
                            score = m(true_values[station][agent]['y'], predictions)
                            metric_scores[name] = score
            else:
                metric_scores = {}
                for m in metrics:
                    name = 'metric'
                    try:
                        name = m.__name__
                    except: 
                        name = type(m).__name__
                        pass
                    score = m(true_values[station][agent]['y'], predictions)
                    metric_scores[name] = score

                new_results[station][agent] = metric_scores

    return new_results

# === TRAINING LOOP ===
def prepare_execution_values(agents, stations, model_names, to_execute, ignore):
    '''
	Returns a dict containing all the agents, stations and models to execute.
    '''
    key_defaults = {'stations': stations, 'agents': agents, 'models': model_names}

    if to_execute == 'all':
        to_execute = key_defaults.copy()
    elif isinstance(to_execute, list): # the user provided a list of agents to execute
        to_execute = {'stations': stations, 'agents': to_execute, 'models': model_names}
    elif isinstance(to_execute, dict): # the user provided a dict with elements to execute
        unknown_keys = set(to_execute) - key_defaults.keys()
        if unknown_keys:
            print(f'Warning: Unknown keys in `to_execute`: {", ".join(unknown_keys)}')

        to_execute = {
            key: (to_execute[key] if isinstance(to_execute[key], list) else key_defaults[key])
            if key in to_execute else key_defaults[key]
            for key in key_defaults
        }

    if ignore is None:
        ignore = {key: [] for key in key_defaults}
    elif isinstance(ignore, list): # the user provided a list of agents to ignore
        unknown_agents = [a for a in ignore if a not in agents]
        if unknown_agents:
            print(f'Warning: Unknown agents in `ignore`: {", ".join(unknown_agents)}')

        ignore = {'stations': [], 'agents': ignore, 'models': []}
    elif isinstance(ignore, dict): # the user provided a dict with elements to ignore
        unknown_keys = set(ignore) - key_defaults.keys()
        if unknown_keys:
            print(f'Warning: Unknown keys in `ignore`: {", ".join(unknown_keys)}')

        ignore = {key: ignore.get(key, []) for key in key_defaults}

    return {key: [item for item in to_execute[key] if item not in ignore[key]] for key in key_defaults}

def check_execution_values(to_execute, data, return_dict=False):
    '''
	Used to filter out from `to_execute` agents that would not be executed because no station has them.
    This can happen if some station with unique agents gets ignored.

    if `return_dict`, return an empty dictionary already initialized with the needed elements to store the results.
    '''
    for agent in to_execute['agents']:
        c=0
        for station in to_execute['stations']: # cpunt the times the agent appears in the stations
            if agent in data[station]:
                c+=1
        if c==0: # if it never appears, remove it from the agents to execute
            to_execute['agents'] = [a for a in to_execute['agents'] if a != agent]

    if return_dict:
        results = {}    
        for agent in to_execute['agents']:
            for model in to_execute['models']:
                for station in to_execute['stations']:                
                        if station not in results:
                            results[station] = {}
                        if agent in data[station]:
                            if agent not in results[station]:
                                results[station][agent] = {}
                            if model not in results[station][agent]:
                                results[station][agent][model] = {}
        
        return to_execute, results

    return to_execute

def create_sequences(x_df, y_df, time_steps, use_mask=True, mask_value=-999.0, sliding_window=True):
    '''
	If `use_mask` a masking value is applied to the missing data, else `bfill()` is applied.
    If `sliding_window` the sequences are a sliding window, else the input data is sliced evenly.
    '''
    X = []
    y = []
    resampled = x_df.copy().resample('1h').max()
    
    if sliding_window:
        for i in range(len(resampled) - time_steps + 1):
            seq = resampled.iloc[i : i+time_steps]
            if not (np.isnan(seq.values).all() or np.isnan(seq.iloc[-1].values).all()):
                if use_mask:
                    seq = seq.fillna(mask_value)
                else:
                    seq = pd.DataFrame(seq).ffill().bfill()
                X.append(seq) # time_steps values are needed to predict the next value
                y.append(y_df.copy().loc[seq[-1:].index])
        y = pd.concat(y)
    else: 
        # add rows if the last hour is not 23:00
        last_midnight = resampled.index[-1].normalize()
        last_hour = last_midnight + pd.Timedelta(hours=23)
        if resampled.index[-1] < last_hour:
            filling_hours = pd.date_range(resampled.index[-1] + pd.Timedelta(hours=1), last_hour, freq='1h')
            resampled = resampled.reindex(resampled.index.append(filling_hours))
        for i in range(0, len(resampled), time_steps):
            seq = resampled.iloc[i : i+time_steps]
            if not (np.isnan(seq.values).all()):
                if use_mask:
                    seq = seq.fillna(mask_value)
                else:
                    seq = pd.DataFrame(seq).ffill().bfill()
                # idx = seq[:1].index.map(lambda x: x.date())
                idx = seq[:1].index.tolist()[0]
                # print(y_df.loc[idx])
                try:
                    y_val = y_df.copy().loc[idx] # try to see if y_df has the date
                    # if y_val.shape != (1,1):
                        # continue # on the dataset some readings arent precisely at midnight...
                                 # so there might be two values per day 
                    y.append(y_val)
                    X.append(seq)
                except: pass   

    X = np.array(X)
    return X, y

def split_dataset(ds, test_frac=1/6, val_frac=0.1, batch_size=32):
    # Get the total size of the dataset
    total_size = len(list(ds))  # List the entire dataset to get size (might be memory-intensive for large datasets)
    
    # Calculate the number of test samples
    test_size = int(total_size * test_frac)
    train_val_size = total_size - test_size

    # Split the dataset into train+val and test portions
    train_val_dataset = ds.take(train_val_size)
    test_dataset = ds.skip(train_val_size)

    # Shuffle train + val dataset and create static validation set
    train_val_dataset = train_val_dataset.shuffle(train_val_size, reshuffle_each_iteration=False)

    # Calculate the size of the validation set
    val_size = int(val_frac * train_val_size)

    # Split the train + val dataset into training and validation datasets
    val_dataset = train_val_dataset.take(val_size)
    train_dataset = train_val_dataset.skip(val_size)

    # Batch and prefetch datasets for performance
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def save_model(model, folder, station, agent):
    station = station.replace(' ','_')
    agent = agent.replace('.','_')
    filename = os.path.join(folder, station+'.'+agent)

    if isinstance(model, tf.keras.Model):
        model.save(f'{filename}') # cant make it work with just .h5 :(
    elif isinstance(model, xgb.XGBModel):
        model.save_model(f'{filename}.json')
    else: # rfr
        joblib.dump(model, f'{filename}.pkl')

def load_models(folder):
    models = {}

    for filename in os.listdir(folder):
        if filename.endswith(('.h5', '.json', '.pkl')) or os.path.isdir(os.path.join(folder, filename)):
            parts = filename.split('.')
            if len(parts) > 2:
                station = parts[0]
                agent = '.'.join(parts[1:])
            else:
                station, agent = parts

            station = station.replace('_',' ')

            if station not in models:
                models[station] = {}

            print(f"Loading model: {station}.{agent}")
            agent = agent.split('.')[0].replace('_','.')

            if os.path.isdir(os.path.join(folder, filename)):
                try:
                    models[station][agent] = tf.keras.models.load_model(os.path.join(folder, filename))
                except ValueError:
                    models[station][agent] = tf.keras.layers.TFSMLayer(os.path.join(folder, filename), call_endpoint='serving_default')

            elif filename.endswith('.json'):
                model = xgb.XGBModel()
                model.load_model(os.path.join(folder, filename))
                models[station][agent] = model
            elif filename.endswith('.pkl'):
                models[station][agent] = joblib.load(os.path.join(folder, filename))
    
    return models

# === TRAINING FUNCTIONS ===

def train_models(models, training_data, test_data, metrics=[], to_execute:list|dict='all', ignore:list|dict=None, random_state=42, v=1):
    '''
    Run all the models at once. 

	`to_execute` and `ignore` can be a list of agents to execute or ignore, respectively.
    It could also be a dict with stations/agents/models to execute or ignore. If so, values can be lists or the string 'all'.
    If a key is not present, all the possible values are used. e.g. if `stations` is missing, all the stations are executed.
    The idea is to be able to chose what to execute or what not to execute. Anyways, both can be specified at the 
    same time and will be merged.

    NOTE: to specify models, use the same name that appears in `models` keys.

    `v`=2 will print metric scores after each training.

    Returns a dict:
    ```
    {   station: {  # Key: station identifier
            agent: {  # Key: agent identifier
                model: {  # Key: model identifier
                    "prediction": predictions,
                    "metric_scores": {
                        metric: score 
    }}}}}
    ```
    '''
    # retrieve all the possible values for agents, STATIONS and models
    agents = list({agent for station in training_data.values() for agent in station.keys()})
    stations = list(training_data.keys())
    model_names = list(models.keys())

    to_execute = prepare_execution_values(agents, stations, model_names, to_execute, ignore)
    to_execute, results = check_execution_values(to_execute, test_data, return_dict=True)

    if v>=0:
        print('==================================================================================')
        print('Train settings:')
        for key, value in to_execute.items():
            print(f'{key}: {value}')
        print('==================================================================================')

    for agent in to_execute['agents']:
        if v>0: print(f'Agent {agent}')
        for model in to_execute['models']:
            if v>0: print(f'> {model} model:')
            for station in to_execute['stations']:
                if agent in training_data[station]:
                    x_train, y_train, x_test, y_test = training_data[station][agent]['x'],training_data[station][agent]['y'],test_data[station][agent]['x'],test_data[station][agent]['y']
                    model_generator, model_params, training_params, uses_sequences = models[model]
                    if v>0: print(f' >> Training station {station}...')
                    if training_params is None:
                        training_params = {}
                    if uses_sequences:
                        if 'time_steps' not in model_params:
                            raise KeyError('No `time_steps` key found in the model parameters to compute the sequences')
                        ts = model_params['time_steps']

                        x_test = pd.concat([x_train.iloc[-ts+1:],x_test]) # add the needed values
                        
                        use_mask = model_params.get('use_mask', False)
                        x_train, y_train = create_sequences(x_train, y_train, ts, use_mask=use_mask)

                        x_test, y_test = create_sequences(x_test, y_test, ts, use_mask=use_mask)
                    else: # if not using sequences, flatten
                        y_train = y_train.to_numpy().ravel()

                    model_instance = model_generator(**model_params)
                    try:
                        tf.keras.backend.clear_session()  
                        set_random_seed(random_state)
                        model_instance.fit(x_train, y_train, **training_params, verbose=0)
                        predictions = model_instance.predict(x_test, verbose=0)
                    except TypeError:
                        tf.keras.backend.clear_session()  
                        set_random_seed(random_state)
                        model_instance.fit(x_train, y_train, **training_params)
                        predictions = model_instance.predict(x_test)

                    predictions = pd.DataFrame(predictions, index=y_test.index, columns=['Agent_value'])

                    metric_scores = {}
                    for m in metrics:
                        name = 'metric'
                        try:
                            name = m.__name__
                        except: 
                            name = type(m).__name__
                            pass
                        score = m(test_data[station][agent]['y'], predictions)
                        metric_scores[name] = score


                    if v>1:display_metric_scores(metric_scores,'\t')

                    results[station][agent][model]['predictions'] = predictions
                    results[station][agent][model]['metric_scores'] = metric_scores

    return results

def train_agents(models, training_data, test_data, metrics=METRICS, model_out_folder=None, random_state=42, v=1):
    if model_out_folder:
        os.makedirs(model_out_folder, exist_ok=True)
        for filename in os.listdir(model_out_folder):
            file_path = os.path.join(model_out_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
    results = {station:{agent:{} for agent in agents} for station,agents in models.items()}
    for station, agents in models.items():
        for agent, model in agents.items():
            x_train, y_train, x_test, y_test = training_data[station][agent]['x'],training_data[station][agent]['y'],test_data[station][agent]['x'],test_data[station][agent]['y']
            model_desc, model_generator, model_params, training_params, uses_sequences = model
            if v>0: print(f'Predicting {agent} in {station} using {model_desc}...')
            if training_params is None:
                training_params = {}
            if uses_sequences:
                if 'time_steps' not in model_params:
                    raise KeyError('No `time_steps` key found in the model parameters to compute the sequences')
                ts = model_params['time_steps']

                x_test = pd.concat([x_train.iloc[-ts+1:],x_test]) # add the needed values

                use_mask = model_params.get('use_mask', False)
                x_train, y_train = create_sequences(x_train, y_train, ts, use_mask=use_mask)

                x_test, y_test = create_sequences(x_test, y_test, ts, use_mask=use_mask)
            else: # if not using sequences, flatten
                y_train = y_train.to_numpy().ravel()

            model_instance = model_generator(**model_params)
            try:
                tf.keras.backend.clear_session()  
                set_random_seed(random_state)
                model_instance.fit(x_train, y_train, **training_params, verbose=0)
                predictions = model_instance.predict(x_test, verbose=0)
            except TypeError:
                tf.keras.backend.clear_session()  
                set_random_seed(random_state)
                model_instance.fit(x_train, y_train, **training_params)
                predictions = model_instance.predict(x_test)

            if model_out_folder:
                save_model(model_instance, model_out_folder, station, agent)

            predictions = pd.DataFrame(predictions, index=y_test.index, columns=['Agent_value'])
            
            metric_scores = {}
            for m in metrics:
                name = 'metric'
                try:
                    name = m.__name__
                except: 
                    name = type(m).__name__
                    pass
                score = m(test_data[station][agent]['y'], predictions)
                metric_scores[name] = score


            if v>1:display_metric_scores(metric_scores,'\t')

            results[station][agent]['predictions'] = predictions
            results[station][agent]['metric_scores'] = metric_scores

    return results


# === RESULTS ===
def training_results_to_dataframe(results, multiple_models=True):
    data = []
    names = []
    for station, station_dict in results.items():
        for agent, agent_dict in station_dict.items():
            if multiple_models:
                for model, model_dict in agent_dict.items():
                    try:
                        metrics = model_dict["metric_scores"]
                        names = metrics.keys()
                        model_dict = metrics.values()
                        data.append([station, agent, model] + list(model_dict))
                    except: pass
            else:
                # try:
                    metrics = agent_dict["metric_scores"]
                    names = metrics.keys()
                    agent_dict = metrics.values()
                    data.append([station, agent] + list(agent_dict))
                # except: pass

    return pd.DataFrame(data, columns=["Station", "Agent"] + (["Model"] if multiple_models else []) + list(names))


# === PLOTS ===
def plot_train_results(): pass
    # adapt the plot used in the section to compare test results