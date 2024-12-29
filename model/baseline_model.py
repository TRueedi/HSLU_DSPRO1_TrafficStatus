import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_random_baseline_models(train_data, save_path): 
    """
    Train random baseline models for traffic prediction and save the models for each sensor.

    This function generates random baseline predictions for traffic values based on the 
    interquartile range (IQR) of the actual traffic values in the training data. It maps 
    weekdays to numerical values and creates predictions for each sensor by simulating 
    traffic values at 5-minute intervals (300 seconds) for each day of the week.

    Parameters:
        train_data (pd.DataFrame): A pandas DataFrame containing the training data. 
                                   It must include the following columns:
                                   - 'detid': Sensor identifier (categorical).
                                   - 'weekday': Day of the week (categorical as strings: 'Monday', 'Tuesday', etc.).
                                   - 'traffic': Traffic values (numerical).
        save_path (str): The directory path where the baseline models will be saved. 
                         Each sensor's baseline will be stored as a separate file.

    Process:
        1. Maps weekday names to integers (Monday=0, ..., Sunday=6).
        2. Groups the data by sensor ('detid').
        3. For each sensor:
            a. Calculates the interquartile range (IQR) of the 'traffic' column.
            b. Generates random traffic values within the IQR for 7 days and all 
               intervals (5-minute increments, 0 to 86,100 seconds).
            c. Creates a DataFrame of the random predictions with columns:
               - 'weekday': Integer representation of the day of the week.
               - 'interval': Time interval in seconds (0 to 86,100).
               - 'traffic': Randomly generated traffic values.
            d. Saves the DataFrame as a baseline "model" using `joblib.dump`.

    Outputs:
        A baseline model file for each sensor, saved in the specified `save_path` 
        directory. The file name is formatted as `<sensor>_baseline`.
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
    """
    Retrieve random baseline traffic predictions for a specific weekday and set of intervals.

    This function loads pre-saved baseline models for sensors and extracts the predicted 
    traffic values for the specified weekday and time intervals. The predictions are 
    concatenated into a single DataFrame for all sensors.

    Parameters:
        models_path (str): The directory path where the baseline models are stored. 
                           Each model should be a file saved with `joblib.dump`, named in the format 
                           `<sensor_id>_baseline`.
        weekday (int): The numerical representation of the weekday (0=Monday, ..., 6=Sunday).
        interval_values (list, optional): A list of time intervals (in seconds) for which to retrieve predictions. Defaults to intervals spaced hourly.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions for all sensors, with the following columns:
                      - 'interval': Time interval in seconds.
                      - 'traffic': Predicted traffic value.
                      - 'detid': Sensor identifier.
                      
                      The weekday column is dropped from the output, as it is fixed for the query.

    Process:
        1. Reads all files in `models_path` that include "_baseline" in their name.
        2. Loads each baseline model file and filters predictions for the specified `weekday` and `interval_values`.
        3. Adds a 'detid' column to identify the sensor, derived from the file name.
        4. Concatenates the filtered predictions from all sensors into a single DataFrame.
    """
    
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        if '_baseline' in model_filename:
            model_path = os.path.join(models_path, model_filename)
            sensor_baseline = joblib.load(model_path)
            
            prediction_data = sensor_baseline[sensor_baseline['weekday'] == weekday]
            prediction_data = prediction_data[prediction_data['interval'].isin(interval_values)]
            prediction_data['detid'] = model_filename.replace('_baseline', '').replace('-', '/')
            prediction_data = prediction_data.drop('weekday', axis=1)
            
            predictions.append(prediction_data)
            
    return pd.concat(predictions)


def evaluate_random_baseline_models(test_data, models_path):
    """
    Evaluate random baseline traffic prediction models using test data.

    This function compares actual traffic values from the test data against the predicted 
    values from pre-saved random baseline models for each sensor. It calculates evaluation 
    metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) 
    for each sensor and provides the average metrics across all sensors.

    Parameters:
        test_data (pd.DataFrame): A pandas DataFrame containing the test data. 
                                  It must include the following columns:
                                  - 'detid': Sensor identifier (categorical).
                                  - 'weekday': Day of the week (categorical as strings: 'Monday', 'Tuesday', etc.).
                                  - 'interval': Time interval in seconds.
                                  - 'traffic': Actual traffic values (numerical).
        models_path (str): The directory path where the baseline models are stored. 
                           Each model should be a file saved with `joblib.dump`, named in the format `<sensor_id>_baseline`.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the average evaluation metrics across all sensors:
                      - 'Average MAE': Mean Absolute Error averaged over all sensors.
                      - 'Average MSE': Mean Squared Error averaged over all sensors.
                      - 'Average R2': R-squared averaged over all sensors.
                      
                      If no sensors have matching baseline models, the metrics will be `None`.

    Process:
        1. Maps weekday names in `test_data` to numerical values (Monday=0, ..., Sunday=6).
        2. Groups the test data by sensor ('detid') and evaluates predictions for each sensor:
            a. Loads the baseline model for the sensor from `models_path`.
            b. Merges test data and baseline predictions on 'weekday' and 'interval' columns.
            c. Computes MAE, MSE, and R² between actual and predicted traffic values.
        3. Collects evaluation metrics for all sensors and calculates their averages.
        4. Returns the average metrics as a DataFrame.
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
    test_data['weekday'] = test_data['weekday'].map(weekday_mapping)

    mae_scores = []
    mse_scores = []
    r2_scores = []

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
            r2 = r2_score(merged_data['traffic_actual'], merged_data['traffic_pred'])
            
            mae_scores.append(mae)
            mse_scores.append(mse)
            r2_scores.append(r2)
        
        else:
            print(f"Baseline model for sensor {sensor} not found in {models_path}")
    
    average_mae = sum(mae_scores) / len(mae_scores) if mae_scores else None
    average_mse = sum(mse_scores) / len(mse_scores) if mse_scores else None
    average_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else None

    evaluation_results = {
        'Average MAE': average_mae,
        'Average MSE': average_mse,
        'Average R2': average_r2,
    }
    
    return pd.DataFrame(evaluation_results, index=[0])