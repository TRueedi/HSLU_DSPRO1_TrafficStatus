import pandas as pd 
import matplotlib.pyplot as plt


def loadData(path, nrows=None):
    return pd.read_csv(path, nrows=nrows)

# Load the data form seperate dictionary, because the data is too big to load onto github
# Example path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv
dataLondonUTD19 = loadData(path=r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv", nrows=None)

dataframeLondonUTD19 = pd.DataFrame(dataLondonUTD19)


# Drop the error column, because it is not needed
dataframeLondonUTD19 = dataframeLondonUTD19.drop(["error"], axis=1)
dataframeLondonUTD19 = dataframeLondonUTD19.drop(["speed"], axis=1)


# Make a new column with the day of the week
dataframeLondonUTD19['day'] = pd.to_datetime(dataframeLondonUTD19['day'])
dataframeLondonUTD19['weekday'] = dataframeLondonUTD19['day'].dt.day_name()

#Calculate the speed using the formula speed = flow / occupancy
dataframeLondonUTD19['traffic'] = dataframeLondonUTD19['flow'] * dataframeLondonUTD19['occ']

# Calculate the mean traffic in n intervals
def calculate_mean_in_intervals(group, column, num_intervals):
    interval_size = len(group) // num_intervals
    means = []
    
    for i in range(num_intervals):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size if i < num_intervals - 1 else len(group)
        interval_mean = group.iloc[start_idx:end_idx][column].mean()
        means.append(interval_mean)
    
    return means

# Clip outliers in a group
def clip_group(group, column, outlier_factor, num_intervals):
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_factor * IQR
    upper_bound = Q3 + outlier_factor * IQR
    
    # Calculate means in intervals
    means = calculate_mean_in_intervals(group, column, num_intervals)
    interval_size = len(group) // num_intervals
    
    def get_interval_mean(index):
        interval_index = index // interval_size
        if interval_index >= num_intervals:
            interval_index = num_intervals - 1
        return means[interval_index]
    
    # Replace outliers with the mean value of their respective interval
    group[column] = group.apply(lambda row: get_interval_mean(row.name) if row[column] < lower_bound or row[column] > upper_bound else row[column], axis=1)
    return group

#Clip functions to remove outliers for every detector
def clip_outliers(df, column, group_by_detid=False, outlier_factor=1.5, num_intervals=24):
    if group_by_detid:
        df = df.groupby('detid').apply(clip_group, column, outlier_factor, num_intervals)
    else:
        df = clip_group(df, column, outlier_factor, num_intervals)
    
    return df

#Detect anomalies for every detector
def detect_anomalies(df):
    df = df.groupby('detid')['traffic'].mean().reset_index()
    Q1 = df['traffic'].quantile(0.25)
    Q3 = df['traffic'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Identify anomalies
    anomalies = df[(df['traffic'] < lower_bound) | (df['traffic'] > upper_bound)]
    
    # Return unique detid values where anomalies are detected
    return anomalies['detid'].unique()

# Filter the DataFrame to clip outliers
# Clip means setting the values outside the bounds to the bounds
# This is done to make the data more readable
dataframeLondonUTD19 = clip_outliers(dataframeLondonUTD19,column='traffic', group_by_detid=True)

# Load the detectors data
dataLondonDetectors = loadData(path=r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_detectors.csv")
dataframeLondonDetectors = pd.DataFrame(dataLondonDetectors)

# Merge the DataFrames on 'detid' and include only the 'lanes' column from dataframeLondonDetectors
if 'detid' in dataframeLondonUTD19.index.names:
    dataframeLondonUTD19 = dataframeLondonUTD19.reset_index(drop=True)

if 'detid' in dataframeLondonDetectors.index.names:
    dataframeLondonDetectors = dataframeLondonDetectors.reset_index(drop=True)

dataframeLondonUTD19 = dataframeLondonUTD19.merge(dataframeLondonDetectors[['detid', 'lanes']], on='detid', how='left')

dataframeLondonUTD19['traffic'] = dataframeLondonUTD19['traffic'] / dataframeLondonUTD19['lanes']

anomalous_detids = detect_anomalies(dataframeLondonUTD19)
dataframeLondonUTD19 = dataframeLondonUTD19[~dataframeLondonUTD19['detid'].isin(anomalous_detids)]
anomalous_detids.size #Todo Print out the number of anomalous detectors

#Check for Outliners in the whole traffic data, not just for one detector
dataframeLondonUTD19 = clip_outliers(dataframeLondonUTD19,column = 'traffic', group_by_detid=False, outlier_factor=2.5)

#Todo make function
# Map the traffic values to a range between 0 and 99
min_traffic = dataframeLondonUTD19['traffic'].min()
max_traffic = dataframeLondonUTD19['traffic'].max()

dataframeLondonUTD19.loc[:,'traffic'] = ((dataframeLondonUTD19['traffic'] - min_traffic) / (max_traffic - min_traffic)) * 99

# Convert the scaled values to integers
dataframeLondonUTD19.loc[:,'traffic'] = dataframeLondonUTD19['traffic'].fillna(0).astype(int)

dataframeLondonUTD19Modified = dataframeLondonUTD19.drop(["lanes", "occ", "flow", "city"], axis=1)

#Todo make function
dataframeLondonUTD19Modified.to_csv(r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19_Modified.csv", index=False)