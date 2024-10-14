import pandas as pd 
import Data_engineering_library as dlib

#sample path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19_Modified.csv
#sample path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data

def get_user_input():
    pathFrom = input("Enter the path to the file from which the data is loaded: ")
    pathTo = input("Enter the path to the folder where the data is saved: ")
    numberOfSplits = int(input("Enter the number of splits 1 split = 8 min runtime: "))
    return pathFrom, pathTo, numberOfSplits

# Get user input
pathFrom, pathTo, numberOfSplits = get_user_input()

print("Loading data from: ", pathFrom)
dataLondonUTD19 = dlib.loadData(path=pathFrom, nrows=1000000)
dataframeLondonUTD19 = pd.DataFrame(dataLondonUTD19)
print("Data loaded")
print("Splitting data into ", numberOfSplits, " splits")

for i in range(numberOfSplits):
    print("Splitting ", i)
    train_set, test_set = dlib.splitDataSniper(dataframeLondonUTD19)
    df_test = pd.DataFrame(test_set)
    df_train = pd.DataFrame(train_set)
    print("Saving split ", i, " to: ", pathTo)
    df_test.to_csv(f"{pathTo}\\London_UTD19_test_{i}.csv", index=False)
    df_train.to_csv(f"{pathTo}\\London_UTD19_train_{i}.csv", index=False)
    print("Split ", i, " done")

print("All splits done")