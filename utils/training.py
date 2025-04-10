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
def log_cosh_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))

def huber(y_true,y_pred):
    return tf.keras.losses.Huber(delta=0.2)(y_true,y_pred).numpy()


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
    '''
    new_results = copy.deepcopy(old_results)
    
    for station, station_results in new_results.items():
        for agent, agent_results in station_results.items():
            for model, model_results in agent_results.items():
                metric_scores = model_results['metric_scores']
                predictions = model_results['predictions']
                for m in metrics:
                    name = 'loss'
                    try:
                        name = m.__name__
                    except: 
                        name = type(m).__name__
                        pass
                    if name not in metric_scores:
                        score = m(true_values[station][agent]['y'], predictions)
                        metric_scores[name] = score

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
                y.append(y_df.loc[seq[-1:].index])
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
                idx = seq[:1].index.map(lambda x: x.date())
                try:
                    y_val = y_df.loc[idx] # try to see if y_df has the date
                    if y_val.shape != (1,1):
                        continue # on the dataset some readings arent precisely at midnight...
                                 # so there might be two values per day 
                    y.append(y_val)
                    X.append(seq)
                except: pass   

    X = np.array(X)
    y = pd.concat(y)
    return X, y


def save_model(model, folder, station, agent):
    station = station.replace(' ','_')
    agent = agent.replace('.','_')
    filename = os.path.join(folder, station+'.'+agent)

    if isinstance(model, tf.keras.Model):
        model.save(f'{filename}') # cant make it work with just .h5 :(
    elif isinstance(model, xgb.XGBModel):
        model.save_model(f'{filename}.json')
    else:
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
                models[station][agent] = tf.keras.models.load_model(os.path.join(folder, filename))
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
                        score = m(test_data[station][agent]['y'],predictions)
                        metric_scores[m.__name__] = score

                    if v>1:display_metric_scores(metric_scores,'\t')

                    results[station][agent][model]['predictions'] = predictions
                    results[station][agent][model]['metric_scores'] = metric_scores

    return results

def train_agents(models, training_data, test_data, model_out_folder=None, random_state=42, v=1):
    if model_out_folder:
        os.makedirs(model_out_folder, exist_ok=True)
        for filename in os.listdir(model_out_folder):
            file_path = os.path.join(model_out_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
    results = {station:{agent:[] for agent in agents} for station,agents in models.items()}
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

            results[station][agent] = predictions

    return results


# === RESULTS ===
def extract_data(results):
    data = []
    names = []
    for station, agents in results.items():
        for agent, models in agents.items():
            for model, values in models.items():
                try:
                    metrics = values["metric_scores"]
                    names = metrics.keys()
                    values = metrics.values()
                    data.append([station, agent, model] + list(values))
                except: pass
    return pd.DataFrame(data, columns=["Station", "Agent", "Model"] + list(names))

# === PLOTS ===
def plot_train_results(): pass
    # adapt the plot used in the section to compare test results