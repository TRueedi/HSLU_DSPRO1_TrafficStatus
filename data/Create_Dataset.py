import pandas as pd 




def loadData(path, nrows=None):
    return pd.read_csv(path, nrows=nrows)

def get_user_input():
    # Example path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv
    # Example path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_detectors.csv
    # Example path C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data
    #pathFrom = input("Enter the path to the file from which the data is loaded: ")
    #pathDetectors = input("Enter the path to the file from which the detectors data is loaded: ")
    #pathTo = input("Enter the path to where the file should be saved is saved: ")
    pathFrom = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_UTD19.csv"
    pathDetectors = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data\London_detectors.csv"
    pathTo = r"C:\Users\samue\OneDrive\AIML\HS2024\Data Sicence Projekt\Data"
    return pathFrom, pathTo, pathDetectors

def preprocess_dataframe(df):
    # Drop the error and speed columns
    df = df.drop(["error", "speed"], axis=1)
    
    # Convert 'day' column to datetime and add a new column with the day of the week
    df['day'] = pd.to_datetime(df['day'])
    df['weekday'] = df['day'].dt.day_name()
    
    return df

def calculate_traffic_speed(df, flow_column='flow', occ_column='occ', traffic_column='traffic'):
    # Calculate the traffic column using the formula speed = flow / occupancy
    df[traffic_column] = df[flow_column] * df[occ_column]
    return df

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
    """
    Detects anomalies in traffic data based on the Interquartile Range (IQR) method.
    This function groups the input DataFrame by 'detid' and calculates the mean traffic for each group.
    It then identifies anomalies as those 'detid' values where the mean traffic is outside the range
    defined by [Q1 - 3*IQR, Q3 + 3*IQR], where Q1 and Q3 are the first and third quartiles of the traffic data.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing traffic data with at least two columns: 'detid' and 'traffic'.
    Returns:

    numpy.ndarray: An array of unique 'detid' values where anomalies are detected.
    """
    
    tempDf = df.groupby('detid')['traffic'].mean().reset_index()
    Q1 = tempDf['traffic'].quantile(0.25)
    Q3 = tempDf['traffic'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Identify anomalies
    anomalies = tempDf[(tempDf['traffic'] < lower_bound) | (tempDf['traffic'] > upper_bound)]
    
    anomalous_detids = anomalies['detid'].unique()
    df = df[~df['detid'].isin(anomalous_detids)]
    return df

def merge_dataframes_on_detid(df1, df2, merge_column='detid', include_column='lanes'):
    """
    Merge two DataFrames on the specified column and include only the specified column from the second DataFrame.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    merge_column (str): The column name to merge on.
    include_column (str): The column name to include from the second DataFrame.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    if merge_column in df1.index.names:
        df1 = df1.reset_index(drop=True)

    if merge_column in df2.index.names:
        df2 = df2.reset_index(drop=True)

    merged_df = df1.merge(df2[[merge_column, include_column]], on=merge_column, how='left')
    return merged_df

def normalize_traffic_by_lanes(df, traffic_column='traffic', lanes_column='lanes', normalized_column='traffic'):
    """
    Normalize the traffic data by dividing it by the number of lanes.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    traffic_column (str): The name of the column containing traffic data.
    lanes_column (str): The name of the column containing lanes data.
    normalized_column (str): The name of the new column to store the normalized traffic data.

    Returns:
    pd.DataFrame: The DataFrame with the normalized traffic data.
    """
    df[normalized_column] = df[traffic_column] / df[lanes_column]
    return df

def normalize_traffic(df, traffic_column='traffic', normalized_range=(0, 99)):
    """
    Normalize the traffic values to a specified range.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    traffic_column (str): The name of the column containing traffic data.
    normalized_range (tuple): The range to normalize the traffic values to (default is (0, 99)).

    Returns:
    pd.DataFrame: The DataFrame with the normalized traffic data.
    """
    min_traffic = df[traffic_column].min()
    max_traffic = df[traffic_column].max()
    min_range, max_range = normalized_range

    df[traffic_column] = ((df[traffic_column] - min_traffic) / (max_traffic - min_traffic)) * (max_range - min_range) + min_range
    return df

def final_process_dataframe(df):
    """
    Convert the scaled values to integers, fill NaN values with 0, and drop specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    traffic_column (str): The name of the column containing traffic data (default is 'traffic').
    columns_to_drop (list): The list of columns to drop from the DataFrame (default is None).

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    columns_to_drop = ["lanes", "occ", "flow", "city"]
    
    # Convert the scaled values to integers and fill NaN values with 0
    df.loc[:, 'traffic'] = df['traffic'].fillna(0).astype(int)
    
    # Drop specified columns
    df_modified = df.drop(columns_to_drop, axis=1)
    
    return df_modified

def export_modified_dataset(df, path):
    """
    Export the modified DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The modified DataFrame.
    path (str): The path to save the CSV file.
    """
    df.to_csv(f"{path}\\London_UTD19_Modified.csv", index=False)
#-------------------------Main-------------------------------------
print("Strting script")
pathFrom, pathTo, pathDetectors = get_user_input()
print("Loading data from: ", pathFrom)
dataframeLondonUTD19 = pd.DataFrame(loadData(path=pathFrom, nrows=1000000))
print("Loading data from: ", pathDetectors)
dataframeDetectors = pd.DataFrame(loadData(path=pathDetectors))
print("Data loaded")
print("Preprocessing data")
dataframeLondonUTD19 = preprocess_dataframe(dataframeLondonUTD19)
print("Calculating traffic speed")
dataframeLondonUTD19 = calculate_traffic_speed(dataframeLondonUTD19)
print("Clipping outliers")
dataframeLondonUTD19 = clip_outliers(dataframeLondonUTD19, 'traffic', group_by_detid=True)
dataframeLondonUTD19 = clip_outliers(dataframeLondonUTD19, 'traffic', group_by_detid=False)
print("Detecting anomalies")
dataframeLondonUTD19 = detect_anomalies(dataframeLondonUTD19)
print("Merging dataframes")
dataframeLondonUTD19 = merge_dataframes_on_detid(dataframeLondonUTD19, dataframeDetectors)
print("Normalizing traffic by lanes")
dataframeLondonUTD19 = normalize_traffic_by_lanes(dataframeLondonUTD19)
print("Normalizing traffic")
dataframeLondonUTD19 = normalize_traffic(dataframeLondonUTD19)
print("Final processing")
dataframeLondonUTD19 = final_process_dataframe(dataframeLondonUTD19)
print("Exporting modified dataset to: ", pathTo)
export_modified_dataset(dataframeLondonUTD19, pathTo)
print("Script finished")
