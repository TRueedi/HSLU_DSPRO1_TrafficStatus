from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import os

def train_rfr_models(train_data, save_path): 
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
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    train_data['weekday'] = train_data['weekday'].map(weekday_mapping)
        
    for sensor, sensor_data in train_data.groupby('detid'):
        sensor_data = sensor_data.drop('detid', axis=1)
        X_train = sensor_data.drop('traffic', axis=1)
        y_train = sensor_data['traffic']
        
        rfr = RandomForestRegressor(random_state=27)
        rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, n_jobs=-1)
        rfr_cv.fit(X_train, y_train)
        
        sensor = sensor.replace('/', '_')
        model_path = f'{save_path}/{sensor}'
        joblib.dump(rfr_cv, model_path)
    
    
def get_rfr_prediction(models_path, weekday, interval_values=
               [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]):
    
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
            'detid': model_filename.replace('_', '/'),
            'interval': X_values['interval'],
        }))
        
    return pd.concat(predictions)



        
        
        
    
