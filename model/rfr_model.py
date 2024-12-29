from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
import os
        

def train_rfr_models(train_data, save_path): 
    """
    Train and save Random Forest Regressor models for traffic prediction.

    This function trains a Random Forest Regressor (RFR) model for each sensor in the training data 
    using a Grid Search with cross-validation to optimize hyperparameters. The best model for each 
    sensor is saved to the specified directory.

    Parameters:
        train_data (pd.DataFrame): A pandas DataFrame containing the training data. 
                                   It must include the following columns:
                                   - 'detid': Sensor identifier (categorical).
                                   - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                   - 'interval': Time interval in seconds from the start of the day.
                                   - 'traffic': Traffic values (numerical).
        save_path (str): The directory path where the trained models will be saved. Each model will be saved as a file named after the sensor ID.

    Process:
        1. Maps weekday names to integers (Monday=0, ..., Sunday=6).
        2. Groups the training data by sensor ('detid').
        3. For each sensor:
            a. Prepares the features (`X_train`) and target (`y_train`).
            b. Initializes a Random Forest Regressor (RFR) with a random state for reproducibility.
            c. Performs a Grid Search with 5-fold cross-validation to find the best hyperparameters:
               - 'n_estimators': Number of trees in the forest.
               - 'max_depth': Maximum depth of each tree.
               - 'min_samples_split': Minimum number of samples required to split an internal node.
               - 'min_samples_leaf': Minimum number of samples required to be at a leaf node.
            d. Trains the best model and saves it using `joblib.dump` with the file name formatted as `<sensor>`.

    Returns:
        None: The function saves the trained models to the specified `save_path`.
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
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    train_data['weekday'] = train_data['weekday'].map(weekday_mapping)
        
    for sensor, sensor_data in train_data.groupby('detid'):
        sensor_data = sensor_data.drop('detid', axis=1)
        X_train = sensor_data.drop('traffic', axis=1)
        y_train = sensor_data['traffic']
        
        rfr = RandomForestRegressor(random_state=27)
        rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
        rfr_cv.fit(X_train, y_train)
        
        sensor = sensor.replace('/', '-')
        model_path = f'{save_path}/{sensor}'
        joblib.dump(rfr_cv, model_path)    

def get_rfr_prediction(models_path, weekday, interval_values=
               [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]):
    """
    Retrieve traffic predictions from Random Forest Regressor (RFR) models for specific intervals and a given weekday.

    This function loads pre-trained RFR models for each sensor, uses the models to predict 
    traffic values for the specified weekday and intervals, and combines the predictions 
    into a single DataFrame.

    Parameters:
        models_path (str): The directory path where the RFR models are stored. Each model should be a file saved with `joblib.dump`.
        weekday (int): The numerical representation of the weekday (0=Monday, ..., 6=Sunday).
        interval_values (list, optional): A list of time intervals (in seconds) for which to retrieve predictions. Defaults to hourly intervals.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions for all sensors, with the following columns:
                      - 'traffic': Predicted traffic values.
                      - 'detid': Sensor identifier derived from the model file name.
                      - 'interval': Time intervals corresponding to the predictions.

    Process:
        1. Creates a DataFrame (`X_values`) for the prediction input with columns:
           - 'interval': Time intervals in seconds.
           - 'weekday': Fixed value for the given weekday.
        2. Iterates over all model files in `models_path`:
            a. Loads each model using `joblib.load`.
            b. Makes predictions using `sensor_model.predict(X_values)`.
            c. Appends the predictions along with sensor IDs (derived from file names) to the output list.
        3. Concatenates predictions from all sensors into a single DataFrame.
    """
    
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        model_path = os.path.join(models_path, model_filename)
        if os.path.isfile(model_path):
            sensor_model = joblib.load(model_path)
            y_pred = sensor_model.predict(X_values)
            
            predictions.append(pd.DataFrame({
            'traffic': y_pred,
            'detid': model_filename.replace('-', '/'),
            'interval': X_values['interval'],
        }))
        
    return pd.concat(predictions)


def evaluate_rfr_models(test_data, models_path):
    """
    Evaluate Random Forest Regressor (RFR) models on test data.

    This function evaluates the performance of pre-trained RFR models for each sensor using 
    test data. It calculates the following metrics for each sensor:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - R-squared (R²)

    Parameters:
        test_data (pd.DataFrame): A pandas DataFrame containing the test data. 
                                  It must include the following columns:
                                  - 'detid': Sensor identifier (categorical).
                                  - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                  - 'interval': Time interval in seconds from the start of the day.
                                  - 'traffic': Actual traffic values (numerical).
        models_path (str): The directory path where the pre-trained models are stored. 
                           Each model should be a file saved with `joblib.dump`, named after 
                           the sensor ID (with '/' replaced by '-').

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics for each sensor, with the following columns:
                      - 'detid': Sensor identifier.
                      - 'MAE': Mean Absolute Error.
                      - 'MSE': Mean Squared Error.
                      - 'R2': R-squared score.

    Process:
        1. Maps weekday names in `test_data` to numerical values (Monday=0, ..., Sunday=6).
        2. Groups the test data by sensor ('detid').
        3. For each sensor:
            a. Locates the corresponding model file in `models_path`.
            b. Loads the model using `joblib.load`.
            c. Prepares the test features (`X_test`) and target (`y_test`).
            d. Predicts traffic values using the model.
            e. Calculates MAE, MSE, and R² metrics.
            f. Stores the metrics and sensor ID in a dictionary.
        4. Converts the performance metrics dictionary into a DataFrame.
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
        sensor_model_filename = sensor.replace('/', '-')
        model_path = os.path.join(models_path, sensor_model_filename)
        
        if os.path.isfile(model_path):
            sensor_model = joblib.load(model_path)
            
            X_test = sensor_data.drop(columns=['traffic', 'detid'])
            y_test = sensor_data['traffic']
            
            y_pred = sensor_model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            performance_metrics['detid'].append(sensor)
            performance_metrics['MAE'].append(mae)
            performance_metrics['MSE'].append(mse)
            performance_metrics['R2'].append(r2)
        else:
            print(f"Model for sensor {sensor} not found.")
    
    performance_df = pd.DataFrame(performance_metrics)
    return performance_df