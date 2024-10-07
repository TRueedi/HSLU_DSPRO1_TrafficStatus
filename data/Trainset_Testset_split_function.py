# %%
import pandas as pd 
import numpy as np

def loadData(path):
    return pd.read_csv(path)

def split_data(df):
    test_set_list = []
    train_set_list = []

    # Group by day and sensor, Data should be sorted by day and detid, this is to make sure nothing is mixed up
    grouped = df.groupby(['day', 'detid'])
    print("Grouped")

    for (day, sensor), group in grouped:
        # Randomly select 8 intervals
        intervals = np.random.choice(range(288), 8, replace=False) * 300
        test_indices = group.index[group['interval'].isin(intervals)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))
    
    print("Splitted")

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    print("Concatenated")
    
    return train_set, test_set

#sample path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv
#sample path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data

def get_user_input():
    pathFrom = input("Enter the path to the file from which the data is loaded: ")
    pathTo = input("Enter the path to the folder where the data is saved: ")
    numberOfSplits = int(input("Enter the number of splits 1 split = 8 min runtime: "))
    return pathFrom, pathTo, numberOfSplits

# Get user input
pathFrom, pathTo, numberOfSplits = get_user_input()

print("Loading data from: ", pathFrom)
dataLondonUTD19 = loadData(path=pathFrom)
dataframeLondonUTD19 = pd.DataFrame(dataLondonUTD19)
print("Data loaded")
print("Splitting data into ", numberOfSplits, " splits")

for i in range(numberOfSplits):
    print("Splitting ", i)
    train_set, test_set = split_data(dataframeLondonUTD19)
    df_test = pd.DataFrame(test_set)
    df_train = pd.DataFrame(train_set)
    print("Saving split ", i, " to: ", pathTo)
    df_test.to_csv(f"{pathTo}\\London_UTD19_test_{i}.csv", index=False)
    df_train.to_csv(f"{pathTo}\\London_UTD19_train_{i}.csv", index=False)
    print("Split ", i, " done")

print("All splits done")