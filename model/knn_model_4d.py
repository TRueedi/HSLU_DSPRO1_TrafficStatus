from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import os


def train_4d_knn_model(train_data, save_path): 
    """
    Train a K-Nearest Neighbors (KNN) regressor for multi-sensor traffic prediction.

    This function trains a KNN regression model using a 4-dimensional feature space (sensor ID, weekday, 
    interval, and traffic) and performs hyperparameter optimization with Grid Search and cross-validation. 
    The trained model and label encoder are saved to the specified directory.

    Parameters:
        train_data (pd.DataFrame): A pandas DataFrame containing the training data. 
                                   It must include the following columns:
                                   - 'detid': Sensor identifier (categorical).
                                   - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                   - 'interval': Time interval in seconds from the start of the day.
                                   - 'traffic': Traffic values (numerical).
        save_path (str): The directory path where the trained model and label encoder will be saved.

    Process:
        1. Maps weekday names to integers (Monday=0, ..., Sunday=6).
        2. Encodes the 'detid' column using `LabelEncoder` to convert sensor identifiers into numerical values.
        3. Prepares the features (`X_train`) and target (`y_train`) for training.
        4. Initializes a KNN regressor and defines a parameter grid for Grid Search:
           - 'n_neighbors': Number of neighbors to consider.
           - 'weights': Weight function used in prediction ('uniform' or 'distance').
           - 'p': Distance metric (1 for Manhattan, 2 for Euclidean).
        5. Performs a Grid Search with 5-fold cross-validation to find the best hyperparameters.
        6. Trains the model on the entire dataset using the best parameters.
        7. Saves the trained model and the label encoder to `save_path`.

    Returns:
        None: The function saves the trained model and label encoder to the specified `save_path`.
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
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }
    
    train_data['weekday'] = train_data['weekday'].map(weekday_mapping)
    
    label_encoder = LabelEncoder()
    train_data['detid'] = label_encoder.fit_transform(train_data['detid'])    
    
    X_train = train_data.drop('traffic', axis=1)
    y_train = train_data['traffic']
    
    knn = KNeighborsRegressor()
    knn_cv = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
    knn_cv.fit(X_train, y_train)
    
    model_path = f'{save_path}/knn_4d_model.pkl'
    encoder_path = f'{save_path}/label_encoder.pkl'
    joblib.dump(knn_cv, model_path)
    joblib.dump(label_encoder, encoder_path)
    


def evaluate_knn_4d_model(test_data, model_path, label_encoder_path):
    """
    Evaluate a trained 4D K-Nearest Neighbors (KNN) model for traffic prediction.

    This function evaluates a pre-trained KNN regression model using test data. It calculates 
    evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

    Parameters:
        test_data (pd.DataFrame): A pandas DataFrame containing the test data.
                                  It must include the following columns:
                                  - 'detid': Sensor identifier (categorical).
                                  - 'weekday': Day of the week as strings (e.g., 'Monday', 'Tuesday', etc.).
                                  - 'interval': Time interval in seconds from the start of the day.
                                  - 'traffic': Actual traffic values (numerical).
        model_path (str): The file path to the trained KNN model saved as a `.pkl` file.
        label_encoder_path (str): The file path to the saved `LabelEncoder` for the 'detid' column.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the evaluation metrics:
                      - 'MAE': Mean Absolute Error.
                      - 'MSE': Mean Squared Error.
                      - 'R2': R-squared score.

    Process:
        1. Validates the existence of the model and label encoder files. If either is missing, 
           the function prints an error message and exits.
        2. Loads the trained model and label encoder using `joblib.load`.
        3. Maps weekday names in the test data to numerical values (Monday=0, ..., Sunday=6).
        4. Encodes the 'detid' column in the test data using the loaded label encoder.
        5. Prepares the test features (`X_test`) and target (`y_test`).
        6. Uses the loaded model to predict traffic values (`y_pred`) for the test data.
        7. Calculates the following evaluation metrics:
           - Mean Absolute Error (MAE).
           - Mean Squared Error (MSE).
           - R-squared (R²).
        8. Returns the metrics as a DataFrame.
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
    
    
    if not os.path.isfile(model_path):
        print(f"Model not fount. Path: {model_path}")
        return 
        
    if not os.path.isfile(label_encoder_path):
        print(f"Label encoder not fount. Path: {label_encoder_path}")
        return
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    
    test_data['weekday'] = test_data['weekday'].map(weekday_mapping)
    test_data['detid'] = label_encoder.fit_transform(test_data['detid'])
    
    X_test = test_data.drop('traffic', axis=1)
    y_test = test_data['traffic']
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2]
    }) 
        
