import pandas as pd
import joblib
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder


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
    
    