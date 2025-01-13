import pandas as pd 
from datetime import timedelta
import time
import data_engineering_library as dlib

def get_user_input():
    """
    Prompts the user to input file paths for loading and saving data.

    This function prompts the user to enter three file paths:
    1. The path to the file from which the data is loaded.
    2. The path to the file from which the detectors data is loaded.
    3. The path to where the file should be saved.

    Example paths:
    - C:\\Users\\samue\\OneDrive\\AIML\\HS2024\\Data Sicence Projekt\\Data\\London_UTD19.csv
    - C:\\Users\\samue\\OneDrive\\AIML\\HS2024\\Data Sicence Projekt\\Data\\London_detectors.csv
    - C:\\Users\\samue\\OneDrive\\AIML\\HS2024\\Data Sicence Projekt\\Data

    Returns:
    tuple: A tuple containing three strings:
        - pathFrom (str): The path to the file from which the data is loaded.
        - pathDetectors (str): The path to the file from which the detectors data is loaded.
        - pathTo (str): The path to where the file should be saved.
    """
    pathFrom = input("Enter the path to the file from which the data is loaded: ")
    pathDetectors = input("Enter the path to the file from which the detectors data is loaded: ")
    pathTo = input("Enter the path to where the file should be saved is saved: ")
    #Only for testing
    #pathFrom = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London\London_UTD19.csv"
    #pathDetectors = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London\London_detectors.csv"
    #pathTo = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data"
    return pathFrom, pathTo, pathDetectors

def export_modified_dataset(df, path):
    """
    Export the modified DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The modified DataFrame.
    path (str): The path to save the CSV file.
    """
    df.to_csv(f"{path}\\London_UTD19_modified.csv", index=False)
#-------------------------Main-------------------------------------
print("Starting script")
path_from, path_to, path_detectors = get_user_input()
start_time = time.time()

print("Loading data from: ", path_from)
dataframe_London_UTD19 = pd.DataFrame(dlib.load_data(path=path_from, nrows=None))
print("Loading data from: ", path_detectors)
dataframe_detectors = pd.DataFrame(dlib.load_data(path=path_detectors))
print("Data loaded")

print("Preprocessing data")
preprocess_start = time.time()
dataframe_London_UTD19, errors = dlib.preprocess_dataframe(dataframe_London_UTD19)
print(f"Errors found and dropped: {errors}")
print(f"Preprocessing data took {round(time.time() - preprocess_start)} seconds")

print("Drop bad days")
drop_bad_days_start = time.time()
dataframe_London_UTD19 = dlib.drop_false_values_by_date(dataframe_London_UTD19, column='flow')
print(f"Drop bad days took {round(time.time() - drop_bad_days_start)} seconds")

print("Clipping outliers on occ")
clip_outliers_start = time.time()
dataframe_London_UTD19 = dlib.clip_outliers(dataframe_London_UTD19, column='occ', group_by_detid=True, outlier_factor=3)
print(f"Clipping outliers on occ took {round(time.time() - clip_outliers_start)} seconds")

print("Clipping outliers on flow")
clip_outliers_start = time.time()
dataframe_London_UTD19 = dlib.clip_outliers(dataframe_London_UTD19, column='flow', group_by_detid=True, outlier_factor=3)
print(f"Clipping outliers on flow took {round(time.time() - clip_outliers_start)} seconds")

print("Calculating traffic")
traffic_start = time.time()
dataframe_London_UTD19 = dlib.calculate_traffic_speed(dataframe_London_UTD19)
print(f"Calculating traffic took {round(time.time() - traffic_start)} seconds")

print("Droping outliers on traffic")
drop_outliers_start = time.time()
dataframe_London_UTD19 = dlib.drop_outliers(dataframe_London_UTD19, column='traffic', group_by_detid=True, outlier_factor=2)
print(f"Droping outliers on traffic took {round(time.time() - drop_outliers_start)} seconds")

print("Detecting anomalies")
detect_anomalies_start = time.time()
dataframe_anomalies = dlib.detect_anomalies(dataframe_London_UTD19, column='traffic', factor=3, min_IQR=5, min_days=10, min_daily_records=230)
dataframe_London_UTD19 = dataframe_London_UTD19[~dataframe_London_UTD19['detid'].isin(dataframe_anomalies['detid'])]
print(f"Detecting anomalies took {round(time.time() - detect_anomalies_start)} seconds")

print("Exporting anomalies to: ", path_to)
exporting_anomalies_start = time.time()
dataframe_anomalies.to_csv(f"{path_to}\\Anomalies.csv", index=False)
print(f"Exporting anomalies took {round(time.time() - exporting_anomalies_start)} seconds")

print("Combine datapoints")
combine_datapoints_start = time.time()
dataframe_London_UTD19 = dlib.combine_datapoints(dataframe_London_UTD19, ratio=3600)
print(f"Combine datapoints took {round(time.time() - combine_datapoints_start)} seconds")

print("Clipping to max traffic value")
clip_max_traffic_start = time.time()
dataframe_London_UTD19 = dlib.clip_to_high_values(dataframe_London_UTD19, column='traffic', threshold=200)
print(f"Clipping to max traffic value took {round(time.time() - clip_max_traffic_start)} seconds")

print("Normalizing traffic")
normalize_traffic_start = time.time()
dataframe_London_UTD19 = dlib.normalize_traffic(dataframe_London_UTD19)
print(f"Normalizing traffic took {round(time.time() - normalize_traffic_start)} seconds")

print("Merging dataframes")
merge_dataframes_start = time.time()
dataframe_London_UTD19 = dlib.merge_dataframes_on_detid(dataframe_London_UTD19, dataframe_detectors)
print(f"Merging dataframes took {round(time.time() - merge_dataframes_start)} seconds")

print("Final processing")
final_process_start = time.time()
dataframe_London_UTD19 = dlib.final_process_dataframe(dataframe_London_UTD19)
print(f"Final processing took {round(time.time() - final_process_start)} seconds")

print("Exporting modified dataset to: ", path_to)
export_start = time.time()
export_modified_dataset(dataframe_London_UTD19, path_to)
print(f"Exporting modified dataset took {round(time.time() - export_start)} seconds")

total_time = time.time() - start_time
print("Script finished")
print(f"Total script execution time: {round(total_time)} seconds")