import pandas as pd 
from datetime import timedelta
import time
import DataEngineeringLibrary as dlib

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
    #pathFrom = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv"
    #pathDetectors = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_detectors.csv"
    #pathTo = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data"
    return pathFrom, pathTo, pathDetectors

def export_modified_dataset(df, path):
    """
    Export the modified DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The modified DataFrame.
    path (str): The path to save the CSV file.
    """
    df.to_csv(f"{path}\\London_UTD19_Modified.csv", index=False)
#-------------------------Main-------------------------------------
print("Starting script")
pathFrom, pathTo, pathDetectors = get_user_input()
start_time = time.time()

print("Loading data from: ", pathFrom)
dataframeLondonUTD19 = pd.DataFrame(dlib.loadData(path=pathFrom, nrows=None))
print("Loading data from: ", pathDetectors)
dataframeDetectors = pd.DataFrame(dlib.loadData(path=pathDetectors))
print("Data loaded")

print("Preprocessing data")
preprocess_start = time.time()
dataframeLondonUTD19 = dlib.preprocess_dataframe(dataframeLondonUTD19)
print(f"Preprocessing data took {round(time.time() - preprocess_start)} seconds")

print("Calculating traffic speed")
traffic_speed_start = time.time()
dataframeLondonUTD19 = dlib.calculate_traffic_speed(dataframeLondonUTD19)
print(f"Calculating traffic speed took {round(time.time() - traffic_speed_start)} seconds")

print("Clipping outliers")
clip_outliers_start = time.time()
dataframeLondonUTD19 = dlib.clip_outliers(dataframeLondonUTD19, column='traffic', group_by_detid=True)
dataframeLondonUTD19 = dlib.clip_outliers(dataframeLondonUTD19, column='traffic', group_by_detid=False, outlier_factor=2.5)
print(f"Clipping outliers took {round(time.time() - clip_outliers_start)} seconds")

print("Detecting anomalies")
detect_anomalies_start = time.time()
dataframeLondonUTD19 = dlib.detect_anomalies(dataframeLondonUTD19)
print(f"Detecting anomalies took {round(time.time() - detect_anomalies_start)} seconds")

print("Merging dataframes")
merge_dataframes_start = time.time()
dataframeLondonUTD19 = dlib.merge_dataframes_on_detid(dataframeLondonUTD19, dataframeDetectors)
print(f"Merging dataframes took {round(time.time() - merge_dataframes_start)} seconds")

print("Normalizing traffic by lanes")
normalize_traffic_by_lanes_start = time.time()
dataframeLondonUTD19 = dlib.normalize_traffic_by_lanes(dataframeLondonUTD19)
print(f"Normalizing traffic by lanes took {round(time.time() - normalize_traffic_by_lanes_start)} seconds")

print("Normalizing traffic")
normalize_traffic_start = time.time()
dataframeLondonUTD19 = dlib.normalize_traffic(dataframeLondonUTD19)
print(f"Normalizing traffic took {round(time.time() - normalize_traffic_start)} seconds")

print("Final processing")
final_process_start = time.time()
dataframeLondonUTD19 = dlib.final_process_dataframe(dataframeLondonUTD19)
print(f"Final processing took {round(time.time() - final_process_start)} seconds")

print("Exporting modified dataset to: ", pathTo)
export_start = time.time()
export_modified_dataset(dataframeLondonUTD19, pathTo)
print(f"Exporting modified dataset took {round(time.time() - export_start)} seconds")

total_time = time.time() - start_time
print("Script finished")
print(f"Total script execution time: {round(total_time)} seconds")
