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
    interval_values = np.arange(0, 86101, 300)
    
    for sensor, sensor_data in train_data.groupby('detid'):
        y_train = sensor_data['traffic']
        
        Q1 = y_train.quantile(0.25)
        Q3 = y_train.quantile(0.75)
        
        random_lower_bound = int(np.round(Q1, 0))
        random_higher_bound = int(np.round(Q3, 0))
        
        random_predictions = {
            'weekday': [],
            'interval': [],
            'traffic': []
        }
        
        random_traffic_values = np.random.randint(random_lower_bound, random_higher_bound + 1, 7 * len(interval_values))
        
        traffic_index = 0
        for weekday in range (0, 7):
            for interval_val in range(0, 86101, 300):
                random_predictions['weekday'].append(weekday)
                random_predictions['interval'].append(interval_val)
                random_predictions['traffic'].append(random_traffic_values[traffic_index])
                traffic_index += 1
        
        random_predictions_df = pd.DataFrame(random_predictions)
        
        # Save the traffic range for each sensor as a baseline "model"
        sensor = sensor.replace('/', '-')
        model_path = f'{save_path}/{sensor}_baseline'
        joblib.dump(random_predictions_df, model_path)


def get_random_baseline_prediction(models_path, weekday, interval_values=
                                   [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]):
    
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        if '_baseline' in model_filename:
            model_path = os.path.join(models_path, model_filename)
            sensor_baseline = joblib.load(model_path)
            
            prediction_data = sensor_baseline[sensor_baseline['weekday'] == weekday]
            prediction_data = prediction_data[prediction_data['interval'].isin(interval_values)]
            
            predictions.append(prediction_data)
            
    return pd.concat(predictions)


def evaluate_random_baseline_models(test_data, models_path):

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

    mae_scores = []
    mse_scores = []

    for sensor, sensor_data in test_data.groupby('detid'):
        model_filename = f'{sensor.replace("/", "-")}_baseline'
        model_path = os.path.join(models_path, model_filename)
        
        if os.path.exists(model_path):
            baseline_data = joblib.load(model_path)
            
            merged_data = pd.merge(
                sensor_data[['weekday', 'interval', 'traffic']], 
                baseline_data[['weekday', 'interval', 'traffic']], 
                on=['weekday', 'interval'], 
                suffixes=('_actual', '_pred')
            )
            
            mae = mean_absolute_error(merged_data['traffic_actual'], merged_data['traffic_pred'])
            mse = mean_squared_error(merged_data['traffic_actual'], merged_data['traffic_pred'])
            
            mae_scores.append(mae)
            mse_scores.append(mse)
        
        else:
            print(f"Baseline model for sensor {sensor} not found in {models_path}")
    
    average_mae = sum(mae_scores) / len(mae_scores) if mae_scores else None
    average_mse = sum(mse_scores) / len(mse_scores) if mse_scores else None

    evaluation_results = {
        'Average MAE': average_mae,
        'Average MSE': average_mse,
    }
    
    return pd.DataFrame(evaluation_results, index=[0])