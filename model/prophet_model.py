import pandas as pd
import joblib
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import re
from model import model_functions as mf

def onehot_encode_categorical(df, column_name):
    """
    Perform one-hot encoding on a categorical column in a DataFrame.

    This function applies one-hot encoding to a specified column in the DataFrame, 
    adding new binary columns for each unique category in the original column. 
    The original column is dropped after encoding.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the column to encode.
        column_name (str): The name of the column to one-hot encode.

    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded columns added and the original column removed.
    """
    
    encoder = OneHotEncoder(sparse_output=False)
    column_encoded = encoder.fit_transform(df[[column_name]])

    encoded_columns = encoder.get_feature_names_out([column_name])
    column_encoded_df = pd.DataFrame(column_encoded, columns=encoded_columns)

    df_encoded = pd.concat([df, column_encoded_df], axis=1)
    df_encoded.drop(columns=[column_name], inplace=True)

    return df_encoded


def label_encode_categorical(df, column_name):
    """
    Perform label encoding on a categorical column in a DataFrame.

    This function applies label encoding to a specified column in the DataFrame, 
    converting each unique category into a corresponding integer value.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the column to encode.
        column_name (str): The name of the column to label encode.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A new DataFrame with the specified column label-encoded.
            - LabelEncoder: The `LabelEncoder` instance used for the transformation, 
                            which can be used to reverse the encoding or transform other datasets.
    """
    
    df_cp = df.copy()
    label_encoder = LabelEncoder()
    df_cp[column_name] = label_encoder.fit_transform(df_cp[column_name])
    return df_cp, label_encoder


def interval_to_datetime(df, interval_column_name):
    """
    Convert an interval column (in seconds) to a datetime column.

    This function converts a column of intervals (representing seconds since a reference point) 
    into a corresponding datetime column. The intervals are added to a reference datetime 
    (default: Monday, May 18, 2015, midnight).

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the interval column.
        interval_column_name (str): The name of the column containing interval values in seconds.

    Returns:
        pd.DataFrame: A new DataFrame with the converted datetime column added and the original interval column removed.
    """
    
    df_cp = df.copy()
    monday_midnight = datetime(2015, 5, 18)
    df_cp['datetime'] = df_cp[interval_column_name].apply(lambda x: monday_midnight + timedelta(seconds=x))
    df_cp = df_cp.drop(interval_column_name, axis=1)
    return df_cp



def train_prophet_model(train_data, save_path):
    """
    Train and save a Prophet model for traffic prediction with sensor data.

    This function trains a Prophet model using time-series data and a regressor for the sensor ID. 
    The trained model and the label encoder for sensor IDs are saved to the specified directory.

    Parameters:
        train_data (pd.DataFrame): A pandas DataFrame containing the training data.
                                   It must include the following columns:
                                   - 'detid': Sensor identifier (categorical).
                                   - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                   - 'interval': Time interval in seconds from the start of the week.
                                   - 'traffic': Traffic values (numerical).
        save_path (str): The directory path where the trained model and label encoder will be saved.

    Process:
        1. Creates a copy of the training data to avoid modifying the original.
        2. Encodes the 'detid' column using `label_encode_categorical` to convert sensor IDs into integers.
        3. Converts 'weekday' and 'interval' into a continuous datetime column using `unfold_weekday_to_interval` and `interval_to_datetime`.
        4. Renames the traffic column to 'y' and the datetime column to 'ds' as required by Prophet.
        5. Initializes a Prophet model with a specified changepoint prior scale.
        6. Adds the encoded sensor ID ('detid') as a regressor to the Prophet model.
        7. Fits the model to the transformed training data.
        8. Saves the trained model and the label encoder to the specified paths.

    Returns:
        None: The function saves the trained Prophet model and label encoder to the specified directory.
    """
    
    train_data_cp = train_data.copy()
    train_data_enc, le = label_encode_categorical(train_data_cp, 'detid')
    train_data_enc = mf.unfold_weekday_to_interval(train_data_enc)
    train_data_enc = interval_to_datetime(train_data_enc, 'interval')
    train_data_enc = train_data_enc.rename(columns={'traffic' : 'y', 'datetime' : 'ds'})
    
    model = Prophet(changepoint_prior_scale=0.01)
    model.add_regressor('detid')
    model.fit(train_data_enc[['ds', 'y', 'detid']])
    
    model_path = f'{save_path}/prophet_model.pkl'
    le_path = f'{save_path}/prophet_lable_encoder.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    
    print(f'Training complete. Model saved under {model_path}')
    
    
def evaluate_prophet_model(test_data, model_path, le_path):
    """
    Evaluate a trained Prophet model on test data.

    This function evaluates a pre-trained Prophet model by comparing predicted traffic values 
    against actual traffic values in the test data. It calculates evaluation metrics such as 
    Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

    Parameters:
        test_data (pd.DataFrame): A pandas DataFrame containing the test data.
                                  It must include the following columns:
                                  - 'detid': Sensor identifier (categorical).
                                  - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                  - 'interval': Time interval in seconds from the start of the week.
                                  - 'traffic': Actual traffic values (numerical).
        model_path (str): The file path to the trained Prophet model saved as a `.pkl` file.
        le_path (str): The file path to the saved `LabelEncoder` for the 'detid' column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with evaluation metrics (MAE, MSE, R² Score).
            - np.ndarray: The actual traffic values (`y_true`).
            - np.ndarray: The predicted traffic values (`y_pred`).
            - pd.DataFrame: The full forecast generated by the Prophet model.

    Process:
        1. Loads the trained Prophet model and the label encoder using `joblib.load`.
        2. Copies the test data to avoid modifying the original.
        3. Encodes the 'detid' column in the test data using the loaded label encoder.
        4. Converts 'weekday' and 'interval' into a continuous datetime column using 
           `unfold_weekday_to_interval` and `interval_to_datetime`.
        5. Renames the traffic column to 'y' and the datetime column to 'ds' as required by Prophet.
        6. Prepares the future dataframe with 'ds' (datetime) and 'detid' (encoded sensor ID) columns.
        7. Predicts traffic values using the Prophet model.
        8. Calculates evaluation metrics (MAE, MSE, R² Score) by comparing actual (`y_true`) and 
           predicted (`y_pred`) traffic values.
        9. Returns the evaluation metrics, actual and predicted values, and the full forecast.
    """

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    test_data_cp = test_data.copy()
    test_data_cp['detid'] = le.transform(test_data_cp['detid'])
    test_data_cp = mf.unfold_weekday_to_interval(test_data_cp)
    test_data_cp = interval_to_datetime(test_data_cp, 'interval')
    test_data_cp = test_data_cp.rename(columns={'traffic': 'y', 'datetime': 'ds'})
    
    future = test_data_cp[['ds', 'detid']]
    forecast = model.predict(future)
    
    y_true = test_data_cp['y'].values
    y_pred = forecast['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R² Score'],
        'Value': [mae, mse, r2]
    }
    
    return pd.DataFrame(metrics), y_true, y_pred, forecast

    
    
def train_prophet_model_per_sensor(train_data, save_path):
    """
    Train and save a separate Prophet model for each sensor in the data.

    This function trains a Prophet model for each unique sensor (`detid`) in the training data. 
    Each model is trained on the time-series data for that specific sensor, and all models 
    are saved to the specified directory.

    Parameters:
        train_data (pd.DataFrame): A pandas DataFrame containing the training data.
                                   It must include the following columns:
                                   - 'detid': Sensor identifier (categorical).
                                   - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                   - 'interval': Time interval in seconds from the start of the week.
                                   - 'traffic': Traffic values (numerical).
        save_path (str): The directory path where the trained models will be saved.

    Process:
        1. Copies the input training data to avoid modifying the original.
        2. Converts 'weekday' and 'interval' into a continuous datetime column using 
           `unfold_weekday_to_interval` and `interval_to_datetime`.
        3. Renames the traffic column to 'y' and the datetime column to 'ds' as required by Prophet.
        4. Iterates through each unique sensor ID (`detid`):
            a. Filters the data for the specific sensor.
            b. Initializes a Prophet model with specified hyperparameters:
               - `changepoint_prior_scale=0.5`: Regularization for changepoints.
               - `n_changepoints=50`: Number of changepoints to allow.
            c. Fits the model to the time-series data for the sensor.
            d. Saves the trained model to `save_path` with the sensor ID in the file name.
        5. Prints progress every 100 sensors to monitor training status.

    Returns:
        None: The function saves the trained Prophet models to the specified directory.
    """
    
    train_data_cp = train_data.copy()

    train_data_cp = mf.unfold_weekday_to_interval(train_data_cp)
    train_data_cp = interval_to_datetime(train_data_cp, 'interval')
    train_data_cp = train_data_cp.rename(columns={'traffic' : 'y', 'datetime' : 'ds'})
    
    sensor_amout = train_data_cp.detid.unique().shape[0]
    counter = 0
    
    for detid in train_data_cp.detid.unique():
        
        sensor_data = train_data_cp[train_data_cp['detid'] == detid]
        
        model = Prophet(changepoint_prior_scale=0.5)
        model.fit(sensor_data[['ds', 'y']])
        
        detid_string = detid.replace('/', '-')
        model_path = f'{save_path}/prophet_{detid_string}.pkl'
        joblib.dump(model, model_path)
        
        counter += 1
        if counter % 100 == 0:
            print(f'Sensor {detid} training complete. {counter}/{sensor_amout} done.')
        
        
def evaluate_prophet_models_per_sensor(test_data, save_path):
    """
    Evaluate Prophet models for each sensor in the dataset.

    This function evaluates the performance of individual Prophet models trained for each unique sensor (`detid`) 
    by comparing the model's predictions with the actual traffic values in the test data. 
    It computes the following metrics for each sensor:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - R-squared (R²)

    Parameters:
        test_data (pd.DataFrame): A pandas DataFrame containing the test data.
                                  It must include the following columns:
                                  - 'detid': Sensor identifier (categorical).
                                  - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                  - 'interval': Time interval in seconds from the start of the week.
                                  - 'traffic': Actual traffic values (numerical).
        save_path (str): The directory path where the trained Prophet models are saved. 
                         Each model should be named in the format `prophet_<sensor_id>.pkl`.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics for each sensor, with columns:
                      - 'detid': Sensor identifier.
                      - 'MAE': Mean Absolute Error.
                      - 'MSE': Mean Squared Error.
                      - 'R2': R-squared score.

    Process:
        1. Creates a copy of the test data to avoid modifying the original.
        2. Iterates through each unique sensor ID (`detid`):
            a. Loads the trained Prophet model for the sensor from `save_path`.
            b. Filters the test data for the specific sensor.
            c. Converts 'weekday' and 'interval' into a datetime column using 
               `unfold_weekday_to_interval` and `interval_to_datetime`.
            d. Renames columns to 'ds' (datetime) and 'y' (actual traffic) as required by Prophet.
            e. Prepares the 'future' DataFrame with the 'ds' column for predictions.
            f. Predicts traffic values using the loaded Prophet model.
            g. Computes MAE, MSE, and R² between the actual and predicted traffic values.
            h. Appends the metrics and sensor ID to the results dictionary.
        3. Converts the results dictionary into a DataFrame.
    """
    
    test_data_cp = test_data.copy()
    metrics = {
        'detid' : [],
        'MAE' : [],
        'MSE' : [],
        'R2' : []}

    for detid in test_data['detid'].unique():

        detid_string = detid.replace('/', '-')
        
        model_path = f'{save_path}/prophet_{detid_string}.pkl'
        model = joblib.load(model_path)

        sensor_data = test_data_cp[test_data_cp['detid'] == detid]
        sensor_data = mf.unfold_weekday_to_interval(sensor_data)
        sensor_data = interval_to_datetime(sensor_data, 'interval')
        sensor_data = sensor_data.rename(columns={'traffic': 'y', 'datetime': 'ds'})

        future = sensor_data[['ds']]
        forecast = model.predict(future)
        
        y_true = sensor_data['y'].values
        y_pred = forecast['yhat'].values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics['detid'].append(detid)
        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['R2'].append(r2)

    return pd.DataFrame(metrics)



def get_prediction_per_sensor(model_folder_path, weekday):
    """
    Generate traffic predictions for a specified weekday using Prophet models for each sensor.

    This function uses pre-trained Prophet models to predict traffic values for each hour 
    of a given weekday. The results are combined into a single DataFrame containing predictions 
    for all sensors.

    Parameters:
        model_folder_path (str): The directory path containing Prophet models for each sensor.
                                 Models should be named in the format `prophet_<sensor_id>.pkl`.
        weekday (int): The numerical representation of the weekday (0=Monday, ..., 6=Sunday).

    Returns:
        pd.DataFrame: A DataFrame containing traffic predictions for the specified weekday, with columns:
                      - 'detid': Sensor identifier.
                      - 'ds': Datetime of the prediction.
                      - 'traffic': Predicted traffic values (yhat).
                      - 'yhat_lower': Lower bound of the prediction interval.
                      - 'yhat_upper': Upper bound of the prediction interval.

    Process:
        1. Creates a DataFrame with hourly intervals (168 hours for a week) and maps intervals to weekdays.
        2. Filters the DataFrame to include only the specified weekday.
        3. Converts interval values into datetime using `interval_to_datetime`.
        4. Iterates through all Prophet model files in `model_folder_path`:
            a. Loads the model using `joblib.load`.
            b. Extracts the sensor ID (`detid`) from the filename using a regex pattern.
            c. Predicts traffic values for the filtered hourly intervals.
            d. Appends the predictions to a results list.
        5. Combines predictions from all sensors into a single DataFrame.
        6. Renames the main prediction column (`yhat`) to 'traffic' and retains prediction intervals.
    """

    regex_pattern = r"prophet_(.*?)\.pkl"
    get_data = pd.DataFrame({'ds': [i * 3600 for i in range(168)]})        
    get_data['weekday'] = (get_data['ds'] // (3600 * 24)) 
    get_data_weekday = get_data[get_data['weekday'] == weekday]
    get_data_weekday = interval_to_datetime(get_data_weekday, 'ds')
    get_data_weekday = get_data_weekday.rename(columns={'datetime' : 'ds'})
    
    forecasts = []

    for model_filename in os.listdir(model_folder_path):
        model_path = os.path.join(model_folder_path, model_filename)
        
        if not os.path.isfile(model_path):  
            print(f'{model_path} is not a file.')
            continue
        
        model = joblib.load(model_path)
        detid = re.findall(regex_pattern, model_filename)[0].replace('-', '/')
        
        future = get_data_weekday[['ds']]
        forecast = model.predict(future)
        forecast['detid'] = detid
        forecasts.append(forecast[['detid', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
    forecasts_df = pd.concat(forecasts, ignore_index=True)
    forecasts_df = forecasts_df.rename(columns={'yhat' : 'traffic', 'ds': 'interval'})
    forecasts_df['interval'] = forecasts_df['interval'].dt.hour * 3600
    return forecasts_df
       