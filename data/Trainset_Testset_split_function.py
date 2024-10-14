import os
import pandas as pd 
import DataEngineeringLibrary as dlib

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
    pathFrom = input("Enter the path to the file from which the data is loaded: ")
    pathTo = input("Enter the path to the folder where the data is saved: ")
    splittingMethod = input("Enter the splitting method (Sniper/Day/Week): ")
    numberOfSplits = int(input("Enter the number of splits 1 split = 8 min runtime: "))
    return pathFrom, pathTo, numberOfSplits, splittingMethod

def spliter(splittingMethod, dataframeLondonUTD19):
    """
    Splits the given dataframe into train and test sets based on the specified splitting method.

    Parameters:
    splittingMethod (str): The method to use for splitting the data. Options are "Sniper", "Day", or "Week".
    dataframeLondonUTD19 (DataFrame): The dataframe to be split.

    Returns:
    tuple: A tuple containing the train set and the test set.
    """
    if splittingMethod == "Sniper":
        train_set, test_set = dlib.splitDataSniper(dataframeLondonUTD19)
    elif splittingMethod == "Day":
        train_set, test_set = dlib.splitDataDay(dataframeLondonUTD19)
    elif splittingMethod == "Week":
        train_set, test_set = dlib.splitDataWeek(dataframeLondonUTD19)
    else:
        print("Invalid splitting method")
        return None, None
    
    return train_set, test_set

# Get user input
pathFrom, pathTo, numberOfSplits, splittingMethod = get_user_input()

print("Loading data from: ", pathFrom)
dataLondonUTD19 = dlib.loadData(path=pathFrom)
dataframeLondonUTD19 = pd.DataFrame(dataLondonUTD19)
print("Data loaded")
print("Splitting data into ", numberOfSplits, " splits")

for i in range(numberOfSplits):
    print("Splitting ", i)
    train_set, test_set = spliter(splittingMethod=splittingMethod, dataframeLondonUTD19=dataframeLondonUTD19)
    df_test = pd.DataFrame(test_set)
    df_train = pd.DataFrame(train_set)
    print("Saving split ", i, " to: ", pathTo)
    df_test.to_csv(os.path.join(pathTo, f"London_UTD19_test_{splittingMethod}_{i}.csv"), index=False)
    df_train.to_csv(os.path.join(pathTo, f"London_UTD19_train_{splittingMethod}_{i}.csv"), index=False)
    print("Split ", i, " done")

print("All splits done")