from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam
import xgboost


def build_lstm_model(
    time_steps: int,
    n_features: int,
    lstm_units: int | list[int],
    mask_value= -999.0,
    activation= 'relu',
    optimizer= 'adam',
    loss= 'mean_absolute_error',
    **kwargs
):
    if isinstance(lstm_units, int):
        lstm_units = [lstm_units]

    model = []
    if mask_value is not None:
        model.append(Masking(mask_value=mask_value, input_shape=(time_steps, n_features), name='initial_mask'))

    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        if i==0 and mask_value is None:
            model.append(LSTM(units, return_sequences=return_sequences, input_shape=(time_steps, n_features), name=f'LSTM_{i}'))
        else:
            model.append(LSTM(units, return_sequences=return_sequences, name=f'LSTM_{i}'))
        
    model.extend([
        Dense(32, activation=activation, name='classifier'),
        Dense(1, name='classifier_head')
    ])

    model = Sequential(model)
    model.compile(optimizer=optimizer, loss=loss)

    return model

def check_param(param, type, expected_len, name):
    if isinstance(param, type):
        param = [param] * expected_len
    elif len(param) != expected_len:
        raise ValueError(f'len: {len(param)} of `{name}` is different from expected len: {expected_len}')
    return param

def build_ffnn_model(
    input_size: int,
    neurons: int|list[int],
    activation: str|list[str] = 'relu',
    batch_norm: bool|list[bool] = False,
    dropout: float|list[float] = 0.2,
    optimizer = 'adam',
    loss = 'mean_absolute_error',
    seed=42,
    **kwargs
):
    if isinstance(neurons, int):
        neurons = [neurons]

    num_layers = len(neurons)

    dropout = check_param(dropout, float, num_layers, 'dropout')
    activation = check_param(activation, str, num_layers, 'activation')
    batch_norm = check_param(batch_norm, bool, num_layers, 'batch_norm')
    
    model = []

    for i, (neurons_i, activation_i, batch_norm_i, dropout_i) in enumerate(zip(neurons, activation, batch_norm, dropout)):
        additional_params = {}
        if i == 0:
            additional_params['input_shape'] = (input_size,)

        model.append(Dense(neurons_i, activation=activation_i, **additional_params, name=f'dense_{i}'))
        if batch_norm_i:
            model.append(BatchNormalization(name=f'batch_norm_{i}'))
        model.append(Dropout(dropout_i, name=f'dropout_{i}_val_{dropout_i}', seed=seed))

    model.append(Dense(1, name='classification_head'))
    model = Sequential(model)
    model.compile(optimizer=optimizer, loss=loss)

    return model

def build_conv_model(
    time_steps: int,
    n_features: int,
    filters: int|list[int],
    kernel_size: int|list[int] = 3,
    activation: str|list[str] = 'relu',
    padding: str|list[str] = 'same',
    optimizer = 'adam',
    loss = 'mean_absolute_error',
    **kwargs
):
    if isinstance(filters, int):
        filters = [filters]

    num_layers = len(filters)

    kernel_size = check_param(kernel_size, int, num_layers, 'kernel_size')
    activation = check_param(activation, str, num_layers, 'activation')
    padding = check_param(padding, str, num_layers, 'padding')

    model = []

    for i, (filters_i, kernel_size_i, activation_i, padding_i) in enumerate(zip(filters, kernel_size, activation, padding)):
        additional_params = {}
        if i == 0:
            additional_params['input_shape'] = (time_steps, n_features)

        model.append(Conv1D(filters=filters_i, kernel_size=kernel_size_i, activation=activation_i, padding=padding_i, **additional_params, name=f'conv1d_{i}_k-{kernel_size_i}_p-{padding_i}'))

    model.extend([
        GlobalAveragePooling1D(name='global_avg_pool'),
        Dense(64, activation='relu', name='classifier'),
        Dense(1, name='classification_head')
    ])
    model = Sequential(model)
    model.compile(optimizer=optimizer, loss=loss)

    return model

# === SELECTED MODELS ===
def get_models(n_hour_features, n_daily_features):
    # hour models
    lstm_params = {
        'time_steps': 3,
        'n_features': n_hour_features,
        'lstm_units': 128,
        'optimizer': Adam(learning_rate=1e-3),
        'loss': 'mean_absolute_error',
        'use_mask': True
    }
    lstm_train_params = {
        'epochs': 20,
        'batch_size': 64
    }
    lstm = ('LSTM-masked', build_lstm_model, lstm_params, lstm_train_params, True)
    lstm2_params = {
    'time_steps': 5,
    'n_features': n_hour_features,
    'lstm_units': [64, 32],
    'optimizer': Adam(learning_rate=1e-3),
    'loss': 'mean_squared_error',
    'use_mask': True
    }
    lstm2_train_params = {
        'epochs': 20,
        'batch_size': 32,
    }
    lstm2 = ('LSTM_2-masked', build_lstm_model, lstm2_params, lstm2_train_params, True)
    ffnn_params = {
        'input_size': n_hour_features,
        'neurons': [1024, 512, 256, 128],
        'dropout': 0.2,
        'optimizer': Adam(learning_rate=1e-2),
        'loss': 'mean_absolute_error'
    }
    ffnn_train_params = {
        'epochs': 10,
        'batch_size': 32,
    }
    FFNN = ('Feed Forward NN', build_ffnn_model, ffnn_params, ffnn_train_params, False)
    conv_params = {
        'time_steps': 8,
        'n_features': n_hour_features,
        'filters': 64,
        'optimizer': Adam(learning_rate=3e-3),
        'loss': 'mean_absolute_error'
    }
    conv_train_params = {
        'epochs': 10,
        'batch_size': 64,
    }
    conv1d = ('Conv1D', build_conv_model, conv_params, conv_train_params, True)

    # daily models
    ffnn_daily_params = {
        'input_size': n_daily_features,
        'neurons': [1024, 512, 256, 128],
        'optimizer': Adam(learning_rate=3e-4),
        'loss': 'mean_absolute_error'
    }
    ffnn_daily_train_params = {
        'epochs':20,
        'batch_size':32,
    }
    FFNN_daily = ('Feed Forward NN', build_ffnn_model, ffnn_daily_params, ffnn_daily_train_params, False)
    bn_ffnn_daily_params = {
    'input_size': n_daily_features,
    'neurons': [1024, 512, 256, 128],
    'batch_norm': True,
    'dropout': 0.2,
    'optimizer': Adam(learning_rate=3e-4),
    'loss': 'mean_absolute_error'
    }   
    bn_ffnn_daily_train_params = {
        'epochs':20,
        'batch_size':32,
    }
    FFNN_daily_BN = ('Feed Forward NN BatchNorm', build_ffnn_model, bn_ffnn_daily_params, bn_ffnn_daily_train_params, False)
    rfr_params = {
        'n_estimators':100,
        'max_depth':10,
        'min_samples_leaf': 5,
        'max_features': 'log2',
    }
    rfr = ('Random Forest Regressor', RandomForestRegressor, rfr_params, None, False)

    models = {
        'GIARDINI MARGHERITA':{
            'NO2': lstm, 
            'O3': lstm2,
            'PM10':rfr,
            'PM2.5':FFNN_daily_BN
        },
        'PORTA SAN FELICE':{
            'C6H6': lstm, 
            'CO': FFNN, 
            'NO2': conv1d,
            'PM10':FFNN_daily,  
            'PM2.5':FFNN_daily 
        },
        'VIA CHIARINI':{
            'NO2': FFNN, 
            'O3': lstm2, 
            'PM10':FFNN_daily  
        }
    }

    return models


def detailed_model_summary(model):
    print(f"\nModel: {model.name}")
    print("="*80)
    print(f"{'Layer (type)':<35}{'Output Shape':<30}{'Param #':<15}")
    print("="*80)

    total_params = 0

    # Encoder
    for layer in model.encoder.layers:
        total_params += print_layer(layer)
    print()

    # Heads
    for tid, head in zip(model.task_ids.numpy(), model.heads):
        task_id = tid.decode("utf-8")  # decode tf.string to str
        for layer in head.layers:
            total_params += print_layer(layer, prefix=f"{task_id}_head.")
        print()

    print("="*80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_params:,}")
    print(f"Non-trainable params: 0")
    print("="*80)

def print_layer(layer, prefix=""):
    try:
        output_shape = str(layer.output_shape)
    except AttributeError:
        output_shape = "?"
    n_params = layer.count_params()
    r = 35 - len(prefix)
    print(f"{prefix}{layer.name:<{r}} {output_shape:<30} {n_params:<15}")
    return n_params
