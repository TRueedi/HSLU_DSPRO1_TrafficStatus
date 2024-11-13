from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
import os





def train_knn_models(train_data, save_path): 
    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    # Parameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }
    
    train_data['weekday'] = train_data['weekday'].map(weekday_mapping)
        
    for sensor, sensor_data in train_data.groupby('detid'):
        sensor_data = sensor_data.drop('detid', axis=1)
        X_train = sensor_data.drop('traffic', axis=1)
        y_train = sensor_data['traffic']
        
        knn = KNeighborsRegressor()
        knn_cv = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
        knn_cv.fit(X_train, y_train)
        
        # Replace '/' to avoid issues with folder paths
        sensor = sensor.replace('/', '-')
        model_path = f'{save_path}/{sensor}'
        joblib.dump(knn_cv, model_path)


import joblib
import pandas as pd
import os

def get_knn_prediction(models_path, weekday, interval_values=[
               0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 
               36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 
               68400, 72000, 75600, 79200, 82800]):
    
    # Create DataFrame with interval and weekday values
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        model_path = os.path.join(models_path, model_filename)
        if os.path.isfile(model_path):
            # Load the KNN model
            sensor_model = joblib.load(model_path)
            y_pred = sensor_model.predict(X_values)
            
            # Store predictions in DataFrame format
            predictions.append(pd.DataFrame({
                'traffic': y_pred,
                'detid': model_filename.replace('-', '/'),
                'interval': X_values['interval'],
            }))
        
    # Concatenate all predictions
    return pd.concat(predictions)


def evaluate_knn_models(test_data, models_path):
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
        sensor_model_filename = sensor.replace('/', '-') + '.pkl'
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
