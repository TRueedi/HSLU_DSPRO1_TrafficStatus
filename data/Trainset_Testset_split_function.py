# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def loadData(path, nrows=None):
    return pd.read_csv(path)

def split_data(df):
    test_set = pd.DataFrame()
    train_set = pd.DataFrame()

    # Group by day and sensor, Data should be sorted by day and detid, this is to make shure nothing is mixed up
    grouped = df.groupby(['day', 'detid'])

    for (day, sensor), group in grouped:
        # Randomly select 8 intervals
        intervals = np.random.choice(range(288), 8, replace=False) * 300
        test_indices = group.index[group['interval'].isin(intervals)]
        
        # Split into test and train sets
        test_set = pd.concat([test_set, group.loc[test_indices]])
        train_set = pd.concat([train_set, group.drop(test_indices)])
    return train_set, test_set
#Path to the file from which the data is loaded
pathFrom = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv"
#Path to the folder where the data is saved
pathTo = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data"
#Number of splits. keep in mind that it could take a while to split the data 5 min per split
numberOfSplits = 5
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
    df_test.to_csv(pathTo+"\\London_UTD19_test_"+i+".csv", index=False)
    df_train.to_csv(pathTo+"\\London_UTD19_train_"+i+".csv", index=False)
    print("Split ", i, " done")

print("All splits done")