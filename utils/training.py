import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# === TRAINING UTILITIES ===
def create_train_test(df, split_date, y):
    train = df[df.index <= split_date]
    test = df[df.index > split_date]

    x_train = train.drop(columns=[y])
    x_test = test.drop(columns=[y])
    aux=df.columns
    y_train = train.drop(columns=[col for col in aux if col !=y])
    y_test = test.drop(columns=[col for col in aux if col !=y])

    return x_train,y_train,x_test,y_test

# === METRICS UTILITIES ===
def display_metric_scores(metric_dict, start=''):
    for metric, score in metric_dict.items():
        print(f'{start}- {metric}: {score}')

def update_metrics(old_results, true_values, metrics):
    new_results = old_results.copy()
    for station, station_results in old_results.items():
        for agent, agent_results in station_results.items():
            for model, model_results in agent_results.items():
                metric_scores = {}
                predictions = model_results['predictions']
                for m in metrics:
                    score = m(true_values[station][agent]['y'],predictions)
                    metric_scores[m.__name__] = score

                new_results[station][agent][model]['metric_scores'] = metric_scores

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
                        if agent not in results[station]:
                            results[station][agent] = {}
                        if model not in results[station][agent]:
                            results[station][agent][model] = {}
        
        return to_execute, results

    return to_execute

def create_sequences(x_df, time_steps=10):
    X = []
    for i in range(len(x_df) - time_steps):
        X.append(x_df.iloc[i:i+time_steps].values) # time_steps values are needed to predict the next value
    return np.array(X)

def train_models(models, training_data, test_data, metrics, to_execute:list|dict='all', ignore:list|dict=None, v=1):
    '''
	For `to_execute` and `ignore` it can be a list of agents to execute or ignore, respectively.
    It could also be a dict with stations/agents/models to execute or ignore. If so, values must be lists.
    The idea is to be able to chose what to execute or what not to execute. Anyways, both can be specified at the 
    same time and will be merged.

    NOTE: to specify models, use the same name that appears in `models` keys.

    Returns: station: dict[agent: dict[model: dict[prediction: predictions, metric_scores: dict[metric:score]]]]
    '''
    # retrieve all the possible values for agents, STATIONS and models
    agents = list({agent for station in training_data.values() for agent in station.keys()})
    stations = list(training_data.keys())
    model_names = list(models.keys())

    to_execute = prepare_execution_values(agents, stations, model_names, to_execute, ignore)
    to_execute, results = check_execution_values(to_execute, test_data, return_dict=True)

    if v>0:
        print('==========================================================')
        print('Train settings:')
        for key, value in to_execute.items():
            print(f'{key}: {value}')
        print('==========================================================')

    for agent in to_execute['agents']:
        if v>0: print(f'Agent {agent}')
        for model in to_execute['models']:
            if v>0: print(f'> {model} model:')
            for station in to_execute['stations']:
                if agent in training_data[station]:
                    x_train, y_train, x_test, y_test = training_data[station][agent]['x'],training_data[station][agent]['y'],test_data[station][agent]['x'],test_data[station][agent]['y']
                    model_generator, model_params, training_params, uses_sequences = models[model]
                    if training_params is None:
                        training_params = {}
                    if uses_sequences:
                        if 'time_steps' not in model_params:
                            raise KeyError('No `time_steps` key found in the model parameters to compute the sequences')
                        ts = model_params['time_steps']

                        x_test = pd.concat([x_train.iloc[-ts:],x_test]) # add the needed values

                        x_train = create_sequences(x_train, ts)
                        y_train = y_train.iloc[ts:]

                        x_test = create_sequences(x_test, ts)
                    else: # if not using sequences, flatten
                        y_train = y_train.to_numpy().ravel()

                    model_instance = model_generator(**model_params)
                    if v>0: print(f' >> Training station {station}...')
                    try:
                        model_instance.fit(x_train, y_train, **training_params, verbose=0)
                        predictions = model_instance.predict(x_test, verbose=0)
                    except TypeError:
                        model_instance.fit(x_train, y_train, **training_params)
                        predictions = model_instance.predict(x_test)

                    predictions = pd.DataFrame(predictions, index=y_test.index)

                    metric_scores = {}
                    for m in metrics:
                        score = m(test_data[station][agent]['y'],predictions)
                        metric_scores[m.__name__] = score

                    if v>1:display_metric_scores(metric_scores,'\t')

                    results[station][agent][model]['predictions'] = predictions
                    results[station][agent][model]['metric_scores'] = metric_scores

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