import os
import pandas as pd 
import data.data_engineering_library as dlib

#sample path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19_Modified.csv
#sample path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data

def get_user_input():
    """
    Prompts the user to enter the paths for loading and saving data, the splitting method, and the number of splits.

    Returns:
    tuple: A tuple containing:
        - pathFrom (str): The path to the file from which the data is loaded.
        - pathTo (str): The path to the folder where the data is saved.
        - splittingMethod (str): The method to use for splitting the data. Options are "Sniper", "Day", or "Week".
        - numberOfSplits (int): The number of splits to perform.
    """
    path_from = input("Enter the path to the file from which the data is loaded: ")
    path_to = input("Enter the path to the folder where the data is saved: ")
    splitting_method = input("Enter the splitting method (Sniper/Day/Week): ")
    number_of_splits = int(input("Enter the number of splits 1 split = 8 min runtime: "))
    return path_from, path_to, number_of_splits, splitting_method

def spliter(splitting_method, df):
    """
    Splits the given dataframe into train and test sets based on the specified splitting method.

    Parameters:
    splittingMethod (str): The method to use for splitting the data. Options are "Sniper", "Day", or "Week".
    dataframeLondonUTD19 (DataFrame): The dataframe to be split.

    Returns:
    tuple: A tuple containing the train set and the test set.
    """
    if splitting_method == "Sniper":
        train_set, test_set = dlib.splitDataSniper(df)
    elif splitting_method == "Day":
        train_set, test_set = dlib.splitDataDay(df)
    elif splitting_method == "Week":
        train_set, test_set = dlib.splitDataWeek(df)
    else:
        print("Invalid splitting method")
        return None, None
    
    return train_set, test_set

# Get user input
path_from, path_to, number_of_splits, splitting_method = get_user_input()

print("Loading data from: ", path_from)
dataframe_London_UTD19 = dlib.loadData(path= path_from)
print("Data loaded")
print("Splitting data into ", number_of_splits, " splits")

for i in range(number_of_splits):
    print("Splitting ", i)
    df_train, df_test = spliter(splitting_method=splitting_method, df=dataframe_London_UTD19)
    df_train = df_train.drop(columns=['day'])
    df_test = df_test.drop(columns=['day'])
    print("Saving split ", i, " to: ", path_to)
    df_test.to_csv(os.path.join(path_to, f"London_UTD19_test_{splitting_method}_{i}.csv"), index=False)
    df_train.to_csv(os.path.join(path_to, f"London_UTD19_train_{splitting_method}_{i}.csv"), index=False)
    print("Split ", i, " done")

print("All splits done")