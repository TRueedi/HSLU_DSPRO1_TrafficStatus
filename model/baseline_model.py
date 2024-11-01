import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def get_baseline_prediction(train_data, weekday):
    weekday_train_data = train_data[train_data['weekday'] == weekday]
    
    train_detid_dfs = {detid: data for detid, data in weekday_train_data.groupby('detid')}
    
    predictions = []
    
    np.random.seed(24)
    
    for sensor, sensor_data in train_detid_dfs.items():
        y_train = sensor_data['traffic']
        
        Q1 = y_train.quantile(0.25)
        Q3 = y_train.quantile(0.75)
        
        random_lower_bound = int(np.round(Q1, 0))
        random_higher_bound = int(np.round(Q3, 0))
        
        for i in range(0, 24):
        
            if random_lower_bound == random_higher_bound:
                new_prediction = random_lower_bound
            else:
                new_prediction = np.random.randint(random_lower_bound, random_higher_bound)
            
            predictions.append({'detid': sensor, 'interval' : i*3600,'traffic': new_prediction})
    
    random_predictions = pd.DataFrame(predictions)
    
    return random_predictions
