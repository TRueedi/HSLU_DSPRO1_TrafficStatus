import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_random_baseline_models(train_data, save_path): 
    """
    Store the min and max range of 'traffic' for each sensor in the train data as a baseline model.
    """
    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    train_data['weekday'] = train_data['weekday'].map(weekday_mapping)
    
    for sensor, sensor_data in train_data.groupby('detid'):
        y_train = sensor_data['traffic']
        
        Q1 = y_train.quantile(0.25)
        Q3 = y_train.quantile(0.75)
        
        random_lower_bound = int(np.round(Q1, 0))
        random_higher_bound = int(np.round(Q3, 0))
        
        traffic_range = {'min': random_lower_bound, 'max': random_higher_bound}
        
        # Save the traffic range for each sensor as a baseline "model"
        sensor = sensor.replace('/', '-')
        model_path = f'{save_path}/{sensor}_baseline'
        joblib.dump(traffic_range, model_path)


def get_random_baseline_prediction(models_path, weekday, interval_values=
                                   [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]):
    
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        if '_baseline' in model_filename:
            model_path = os.path.join(models_path, model_filename)
            sensor_baseline = joblib.load(model_path)
            
            # Generate random traffic values within the saved range
            y_pred = np.random.uniform(sensor_baseline['min'], sensor_baseline['max'], len(X_values))
            
            predictions.append(pd.DataFrame({
                'traffic': y_pred,
                'detid': model_filename.replace('_baseline', '').replace('-', '/'),
                'interval': X_values['interval']
            }))
    
    return pd.concat(predictions)


def evaluate_random_baseline_models(test_data, models_path):
    """
    Evaluate the random baseline by generating random predictions within the saved range 
    for each sensor and calculating performance metrics.
    """
    performance_metrics = {
        'detid': [],
        'MAE': [],
        'MSE': [],
        'R2': []
    }

    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    test_data['weekday'] = test_data['weekday'].map(weekday_mapping)

    for sensor, sensor_data in test_data.groupby('detid'):
        sensor_baseline_filename = sensor.replace('/', '-') + '_baseline'
        model_path = os.path.join(models_path, sensor_baseline_filename)
        
        if os.path.isfile(model_path):
            sensor_baseline = joblib.load(model_path)
            y_test = sensor_data['traffic']
            
            # Generate random predictions within the saved range
            y_random = np.random.uniform(sensor_baseline['min'], sensor_baseline['max'], len(y_test))
            
            # Calculate baseline metrics
            mae = mean_absolute_error(y_test, y_random)
            mse = mean_squared_error(y_test, y_random)
            r2 = r2_score(y_test, y_random)
            
            performance_metrics['detid'].append(sensor)
            performance_metrics['MAE'].append(mae)
            performance_metrics['MSE'].append(mse)
            performance_metrics['R2'].append(r2)
        else:
            print(f"Baseline model for sensor {sensor} not found.")
    
    performance_df = pd.DataFrame(performance_metrics)
    return performance_df
