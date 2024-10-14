import pandas as pd
import numpy as np

def loadData(path, nrows=None):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    path (str): The path to the CSV file.
    nrows (int, optional): The number of rows to read from the file. If None, all rows are read. Default is None.

    Returns:
    pandas.DataFrame: The DataFrame containing the loaded data.
    """
    return pd.read_csv(path, nrows=nrows)

def splitDataSniper(df, samplesPerDay = 8):
    """
    Splits the DataFrame into training and testing sets based on random intervals.

    This function groups the DataFrame by 'day' and 'detid', then randomly selects a specified number
    of intervals per day for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    samplesPerDay (int): The number of intervals to randomly select per day for the test set. Default is 8.

    Returns:
    tuple: A tuple containing two pandas DataFrames:
        - train_set (pandas.DataFrame): The training set.
        - test_set (pandas.DataFrame): The testing set.
    """
    test_set_list = []
    train_set_list = []

    # Group by day and sensor, Data should be sorted by day and detid, this is to make sure nothing is mixed up
    grouped = df.groupby(['day', 'detid'])

    for (day, sensor), group in grouped:
        # Randomly select samplesPerDay intervals
        intervals = np.random.choice(range(288), samplesPerDay, replace=False) * 300
        test_indices = group.index[group['interval'].isin(intervals)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    
    return train_set, test_set

def splitDataDay(df, testDays=1):
    """
    Splits the DataFrame into training and testing sets based on days.

    This function groups the DataFrame by 'day' and 'detid', then randomly selects a specified number
    of days for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    testDays (int): The number of days to randomly select for the test set. Default is 1.

    Returns:
    tuple: A tuple containing two pandas DataFrames:
        - train_set (pandas.DataFrame): The training set.
        - test_set (pandas.DataFrame): The testing set.
    """
    test_set_list = []
    train_set_list = []

    # Group by sensor, Data should be sorted by day and detid, this is to make sure nothing is mixed up
    grouped = df.groupby('detid')

    for sensor, group in grouped:
        # Randomly select testDays days
        days = group['day'].unique()
        test_days = np.random.choice(days, testDays, replace=False)
        test_indices = group.index[group['day'].isin(test_days)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    
    return train_set, test_set

def splitDataWeek(df, testWeeks=1):
    """
    Splits the DataFrame into training and testing sets based on weeks.

    This function groups the DataFrame by 'week' and 'detid', then randomly selects a specified number
    of weeks for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    testWeeks (int): The number of weeks to randomly select for the test set. Default is 1.

    Returns:
    tuple: A tuple containing two pandas DataFrames:
        - train_set (pandas.DataFrame): The training set.
        - test_set (pandas.DataFrame): The testing set.
    """
    test_set_list = []
    train_set_list = []

    # Ensure the 'day' column is in datetime format and create a 'week' column
    df['day'] = pd.to_datetime(df['day'])
    df['week'] = df['day'].dt.isocalendar().week

    # Group by sensor, Data should be sorted by week and detid, this is to make sure nothing is mixed up
    grouped = df.groupby('detid')

    for sensor, group in grouped:
        # Randomly select testWeeks weeks
        weeks = group['week'].unique()
        test_weeks = np.random.choice(weeks, testWeeks, replace=False)
        test_indices = group.index[group['week'].isin(test_weeks)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    
    return train_set, test_set