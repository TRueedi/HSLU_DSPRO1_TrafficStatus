import pandas as pd
import joblib
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import re

def unfold_weekday_to_interval(df):
    """
    Adjusts the 'interval' column in the DataFrame by adding the number of seconds corresponding to the day of the week.

    This function converts the 'weekday' column to a numerical representation (0 for Monday, 1 for Tuesday, etc.),
    multiplies it by the number of seconds in a day (86400), and adds this value to the 'interval' column. The 'weekday'
    column is then dropped from the DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing 'interval' and 'weekday' columns.

    Returns:a
    pandas.DataFrame: The modified DataFrame with the updated 'interval' column and without the 'weekday' column.
    """
    weekday_to_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    seconds_per_day = 86400
    df['interval'] = df.apply(lambda row: row['interval'] + (weekday_to_num[row['weekday']] * seconds_per_day), axis=1)
    df = df.drop(columns=['weekday'])
    return df

def onehot_encode_categorical(df, column_name):
    encoder = OneHotEncoder(sparse_output=False)
    column_encoded = encoder.fit_transform(df[[column_name]])

    encoded_columns = encoder.get_feature_names_out([column_name])
    column_encoded_df = pd.DataFrame(column_encoded, columns=encoded_columns)

    df_encoded = pd.concat([df, column_encoded_df], axis=1)
    df_encoded.drop(columns=[column_name], inplace=True)

    return df_encoded


def label_encode_categorical(df, column_name):
    df_cp = df.copy()
    label_encoder = LabelEncoder()
    df_cp[column_name] = label_encoder.fit_transform(df_cp[column_name])
    return df_cp, label_encoder


def interval_to_datetime(df, interval_column_name):
    df_cp = df.copy()
    monday_midnight = datetime(2015, 5, 18)
    df_cp['datetime'] = df_cp[interval_column_name].apply(lambda x: monday_midnight + timedelta(seconds=x))
    df_cp = df_cp.drop(interval_column_name, axis=1)
    return df_cp



def train_prophet_model(train_data, save_path):
    train_data_cp = train_data.copy()
    train_data_enc, le = label_encode_categorical(train_data_cp, 'detid')
    train_data_enc = unfold_weekday_to_interval(train_data_enc)
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
    Evaluates the performance of the trained Prophet model using MAE, MSE, and R² metrics.
    
    Parameters:
    test_data (pandas.DataFrame): The test dataset with 'interval', 'weekday', 'detid', and 'traffic' columns.
    model_path (str): Path to the saved Prophet model.
    le_path (str): Path to the saved LabelEncoder for the 'detid' column.

    Returns:
    dict: A dictionary containing the MAE, MSE, and R² score of the model.
    """

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    test_data_cp = test_data.copy()
    test_data_cp['detid'] = le.transform(test_data_cp['detid'])
    test_data_cp = unfold_weekday_to_interval(test_data_cp)
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
    train_data_cp = train_data.copy()

    train_data_cp = unfold_weekday_to_interval(train_data_cp)
    train_data_cp = interval_to_datetime(train_data_cp, 'interval')
    train_data_cp = train_data_cp.rename(columns={'traffic' : 'y', 'datetime' : 'ds'})
    
    sensor_amout = train_data_cp.detid.unique().shape[0]
    counter = 0
    
    for detid in train_data_cp.detid.unique():
        
        sensor_data = train_data_cp[train_data_cp['detid'] == detid]
        
        model = Prophet(changepoint_prior_scale=0.01)
        model.fit(sensor_data[['ds', 'y']])
        
        detid_string = detid.replace('/', '-')
        model_path = f'{save_path}/prophet_{detid_string}.pkl'
        joblib.dump(model, model_path)
        
        counter += 1
        print(f'Sensor {detid} training complete. {counter}/{sensor_amout} done.')
        
        
def evaluate_prophet_models_per_sensor(test_data, save_path):
    """
    Evaluates the performance of multiple Prophet models trained for each sensor.

    Parameters:
    test_data (pandas.DataFrame): The test dataset with 'interval', 'weekday', 'detid', and 'traffic' columns.
    save_path (str): Path to the folder containing saved Prophet models and label encoders, 
                     with filenames formatted as 'prophet_model_<sensor_id>.pkl' and 
                     'prophet_label_encoder_<sensor_id>.pkl'.

    Returns:
    dict: A dictionary containing the MAE, MSE, and R² score of all sensors.
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
        sensor_data = unfold_weekday_to_interval(sensor_data)
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
        forecasts.append(forecast[['detid', 'ds', 'yhat']])
        
        
    forecasts_df = pd.concat(forecasts, ignore_index=True)
    forecasts_df = forecasts_df.rename(columns={'yhat' : 'traffic'})
    return forecasts_df
       