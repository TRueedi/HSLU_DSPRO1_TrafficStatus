import pandas as pd
import numpy as np

def load_data(path,skip_rows=None, nrows=None):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    path (str): The path to the CSV file.
    skip_rows (int or list-like, optional): Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. Default is None.
    nrows (int, optional): The number of rows to read from the file. If None, all rows are read. Default is None.

    Returns:
    pandas.DataFrame: The DataFrame containing the loaded data.
    """
    return pd.read_csv(path, nrows=nrows, skiprows=skip_rows)

def split_data_sniper(df, samples_per_day = 8):
    """
    Splits the DataFrame into training and testing sets based on random intervals.

    This function groups the DataFrame by 'day' and 'detid', then randomly selects a specified number
    of intervals per day for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    samples_per_day (int): The number of intervals to randomly select per day for the test set. Default is 8.

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
        intervals = np.random.choice(range(288), samples_per_day, replace=False) * 300
        test_indices = group.index[group['interval'].isin(intervals)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    
    return train_set, test_set

def split_data_day(df, number_of_test_days=1):
    """
    Splits the DataFrame into training and testing sets based on days.

    This function groups the DataFrame by 'detid', then randomly selects a specified number
    of days for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    number_of_test_days (int): The number of days to randomly select for the test set. Default is 1.

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
        test_days = np.random.choice(days, number_of_test_days, replace=False)
        test_indices = group.index[group['day'].isin(test_days)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    
    return train_set, test_set

def split_data_week(df, number_of_test_weeks=1):
    """
    Splits the DataFrame into training and testing sets based on weeks.

    This function groups the DataFrame by 'week' and 'detid', then randomly selects a specified number
    of weeks for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    number_of_test_weeks (int): The number of weeks to randomly select for the test set. Default is 1.

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
        test_weeks = np.random.choice(weeks, number_of_test_weeks, replace=False)
        test_indices = group.index[group['week'].isin(test_weeks)]
        
        # Split into test and train sets
        test_set_list.append(group.loc[test_indices])
        train_set_list.append(group.drop(test_indices))

    # Concatenate all collected groups at once
    test_set = pd.concat(test_set_list)
    train_set = pd.concat(train_set_list)
    
    return train_set, test_set

def preprocess_dataframe(df):
    """
    Preprocesses the input DataFrame by performing the following operations:
    1. Drops the 'error' and 'speed' columns.
    2. Converts the 'day' column to datetime format.
    3. Adds a new column 'weekday' with the day of the week.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """
    df = df.drop(["error", "speed"], axis=1)

    df['day'] = pd.to_datetime(df['day'])
    df['weekday'] = df['day'].dt.day_name()
    
    return df

def calculate_traffic_speed(df, flow_column='flow', occ_column='occ', traffic_column='traffic'):
    """
    Calculates the traffic speed and adds it as a new column to the DataFrame.

    This function calculates the traffic speed using the formula speed = flow * occupancy
    and adds the result as a new column to the DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    flow_column (str): The name of the column representing the flow. Default is 'flow'.
    occ_column (str): The name of the column representing the occupancy. Default is 'occ'.
    traffic_column (str): The name of the new column to store the calculated traffic speed. Default is 'traffic'.

    Returns:
    pandas.DataFrame: The DataFrame with the new traffic speed column added.
    """
    df[traffic_column] = df[flow_column] * df[occ_column]
    return df

def calculate_mean_in_intervals(group, column, num_intervals):
    """
    Calculates the mean values of the specified column in the group DataFrame divided into intervals.

    This function divides the data in the specified column into a given number of intervals and calculates
    the mean value for each interval.

    Parameters:
    group (pandas.DataFrame): The input group DataFrame containing the data.
    column (str): The name of the column to calculate mean values for.
    num_intervals (int): The number of intervals to divide the data into.

    Returns:
    list: A list of mean values for each interval.
    """
    interval_size = len(group) // num_intervals
    means = []
    
    for i in range(num_intervals):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size if i < num_intervals - 1 else len(group)
        interval_mean = group.iloc[start_idx:end_idx][column].mean()
        means.append(interval_mean)
    
    return means

def clip_group(group, column, outlier_factor, num_intervals):
    """
    Clips outliers in the specified column of the group DataFrame.

    This function calculates the interquartile range (IQR) to determine outliers in the specified column.
    Outliers are replaced with the mean value of their respective interval. The data is divided into
    a specified number of intervals, and the mean value for each interval is calculated.

    Parameters:
    group (pandas.DataFrame): The input group DataFrame containing the data.
    column (str): The name of the column to process for outliers.
    outlier_factor (float): The factor used to determine the bounds for clipping outliers.
    num_intervals (int): The number of intervals to divide the data into for calculating mean values.

    Returns:
    pandas.DataFrame: The group DataFrame with outliers clipped.
    """
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

def clip_outliers(df, column, group_by_detid=False, outlier_factor=1.5, num_intervals=24):
    """
    Clips outliers in the specified column of the DataFrame.

    This function can optionally group the DataFrame by 'detid' before clipping outliers.
    Outliers are determined based on the interquartile range (IQR) and replaced with the mean value
    of their respective interval.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to process for outliers.
    group_by_detid (bool): If True, the DataFrame is grouped by 'detid' before processing. Default is False.
    outlier_factor (float): The factor used to determine the bounds for clipping outliers. Default is 1.5.
    num_intervals (int): The number of intervals to divide the data into for calculating mean values. Default is 24.

    Returns:
    pandas.DataFrame: The DataFrame with outliers clipped.
    """
    if group_by_detid:
        df = df.groupby('detid').apply(clip_group, column, outlier_factor, num_intervals)
        # reset the index to avoid issues with the groupby operation
        df = df.reset_index(drop=True)
    else:
        df = clip_group(df, column, outlier_factor, num_intervals)
    
    return df

def drop_outliers(df, column, group_by_detid=True, outlier_factor=3):
    """
    Removes outliers from a DataFrame based on the Interquartile Range (IQR) method.
    This function can optionally group the DataFrame by 'detid' before removing outliers.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing traffic data with at least the specified column.
    column (str): The name of the column to check for outliers.
    group_by_detid (bool, optional): If True, the DataFrame is grouped by 'detid' before removing outliers. Default is True.
    outlier_factor (float, optional): The multiplier for the IQR to define the bounds for detecting outliers. Default is 3.

    Returns:
    pandas.DataFrame: A DataFrame with the outliers removed.
    """
    if group_by_detid:
        df = df.groupby('detid').apply(drop_group, column, outlier_factor)
        # reset the index to avoid issues with the groupby operation
        df = df.reset_index(drop=True)
    else:
        df = drop_group(df, column, outlier_factor)
    
    return df

def drop_group(group, column, outlier_factor):
    """
    Removes outliers from a DataFrame based on the Interquartile Range (IQR) method.
    This function can optionally group the DataFrame by 'detid' before removing outliers.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing traffic data with at least the specified column.
    column (str): The name of the column to check for outliers.
    group_by_detid (bool, optional): If True, the DataFrame is grouped by 'detid' before removing outliers. Default is True.
    outlier_factor (float, optional): The multiplier for the IQR to define the bounds for detecting outliers. Default is 3.

    Returns:
    pandas.DataFrame: A DataFrame with the outliers removed.
    """
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_factor * IQR
    upper_bound = Q3 + outlier_factor * IQR
    
    # Filter out the outliers
    filtered_group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

    return group

def drop_false_values(df, column, outlier_factor=5):
    """
    Drops outliers from a DataFrame based on the value counts of a specified column using the Interquartile Range (IQR) method.
    This function groups the DataFrame by 'detid' before applying the outlier detection and removal process.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    column (str): The name of the column to calculate value counts and identify outliers.
    outlier_factor (float, optional): The multiplier for the IQR to define the bounds for detecting outliers. Default is 5.

    Returns:
    pandas.DataFrame: A DataFrame with the outliers removed.
    """
    def drop_by_group(group):
        # Count the occurrences of each unique value in the specified column
        value_counts = group[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        # Calculate Q1, Q3, and IQR of the value counts
        Q1 = value_counts['count'].quantile(0.25)
        Q3 = value_counts['count'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_factor * IQR
        upper_bound = Q3 + outlier_factor * IQR
        
        # Identify the outliers
        outliers = value_counts[(value_counts['count'] < lower_bound) | (value_counts['count'] > upper_bound)]
        
        # Drop the outliers from the group
        filtered_group = group[~group[column].isin(outliers[column])]
        
        return filtered_group
    
    # Group by 'detid' and apply the drop_group function
    filtered_df = df.groupby('detid').apply(drop_by_group).reset_index(drop=True)
    
    return filtered_df

def detect_anomalies(df, column = 'traffic', factor=3, min_IQR=5, min_data_points=5000):
    """
    Detects anomalies in traffic data based on the Interquartile Range (IQR) method and other criteria.
    This function groups the input DataFrame by 'detid' and calculates the mean traffic for each group.
    It then identifies anomalies as those 'detid' values where the mean traffic is outside the range
    defined by [Q1 - factor*IQR, Q3 + factor*IQR], where Q1 and Q3 are the first and third quartiles of the traffic data.
    Additionally, it identifies anomalies where the IQR is too small or there are not enough data points.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing traffic data with at least two columns: 'detid' and the specified column.
    column (str, optional): The name of the column to calculate the mean and identify anomalies. Default is 'traffic'.
    factor (float, optional): The multiplier for the IQR to define the bounds for detecting anomalies. Default is 3.
    min_IQR (float, optional): The minimum IQR threshold to identify anomalies. Default is 5.
    min_data_points (int, optional): The minimum number of data points required to avoid being classified as an anomaly. Default is 5000.

    Returns:
    tuple: A tuple containing:
        - pandas.DataFrame: A DataFrame with the anomalies removed, containing only the 'detid' values within the normal range.
        - pandas.DataFrame: A DataFrame with the 'detid' values identified as anomalies and columns indicating the type of anomaly.
    """
    anomalies_mean_out_of_bound_list = anomalies_mean_out_of_bound(df, column, factor)
    anomalies_IQR_to_small_list = anomalies_IQR_to_small(df, column, min_IQR=min_IQR)
    anomalies_not_enough_data_list = anomalies_not_enough_data(df, min_data_points=min_data_points)
    anomalies = np.concatenate([anomalies_mean_out_of_bound_list, anomalies_IQR_to_small_list, anomalies_not_enough_data_list])
    anomalies = np.unique(anomalies)
    
    
    df.drop(df[df['detid'].isin(anomalies)].index, inplace=True)

    dataframe_anomalies = pd.DataFrame(anomalies, columns=['detid'])

    dataframe_anomalies['mean_out_of_bound'] = dataframe_anomalies['detid'].isin(anomalies_mean_out_of_bound_list)
    dataframe_anomalies['IQR_to_small'] = dataframe_anomalies['detid'].isin(anomalies_IQR_to_small_list)
    dataframe_anomalies['not_enough_data'] = dataframe_anomalies['detid'].isin(anomalies_not_enough_data_list)
    
    return df, dataframe_anomalies

def anomalies_mean_out_of_bound(df, column, factor=3):
    """
    Identifies anomalies in a DataFrame based on the Interquartile Range (IQR) method.
    This function groups the DataFrame by 'detid' and calculates the mean of the specified column.
    It then identifies anomalies as those 'detid' values where the mean value is outside the range
    defined by [Q1 - factor*IQR, Q3 + factor*IQR], where Q1 and Q3 are the first and third quartiles of the data.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    column (str): The name of the column to calculate the mean and identify anomalies.
    factor (float, optional): The multiplier for the IQR to define the bounds for detecting anomalies. Default is 3.

    Returns:
    numpy.ndarray: An array containing the 'detid' values identified as anomalies.
    """
    # Group by 'detid' and calculate the mean of the specified column
    temp_df = df.groupby('detid')[column].mean().reset_index()
    
    # Calculate Q1, Q3, and IQR
    Q1 = temp_df[column].quantile(0.25)
    Q3 = temp_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Identify anomalies
    temp_df = temp_df[(temp_df[column] < lower_bound) | (temp_df[column] > upper_bound)]
    anomalies = temp_df['detid'].unique()
    print(f"Anomalies detected based on IQR: {anomalies.size}")

    return anomalies

def anomalies_IQR_to_small(df, column='traffic', min_IQR=5):
    """
    Identifies 'detid' values in a DataFrame where the Interquartile Range (IQR) of the specified column is below a given threshold.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    column (str, optional): The name of the column to calculate the IQR. Default is 'traffic'.
    min_IQR (float, optional): The IQR threshold to check against. Default is 5.

    Returns:
    numpy.ndarray: An array containing the 'detid' values with IQR below the specified threshold.
    """
    
    def calculate_iqr(group):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        return IQR
    
    iqr_values = df.groupby('detid').apply(calculate_iqr).reset_index(name='IQR')
    anomalies = iqr_values[iqr_values['IQR'] < min_IQR]['detid'].unique()
    print(f"Anomalies detected based on IQR too small: {anomalies.size}")

    return anomalies

def anomalies_not_enough_data(df, column='detid', min_data_points=5000):
    """
    Identifies 'detid' values in a DataFrame that have fewer than a specified number of data points.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    column (str, optional): The name of the column representing 'detid'. Default is 'detid'.
    min_data_points (int, optional): The minimum number of data points required to avoid being classified as an anomaly. Default is 5000.

    Returns:
    numpy.ndarray: An array containing 'detid' values with fewer than the specified number of data points.
    """
    detid_counts = df.groupby(column).size().reset_index(name='count')
    anomalies = detid_counts[detid_counts['count'] < min_data_points][column].unique()
    print(f"Anomalies detected based on not enough data: {anomalies.size}")
    
    return anomalies

def combine_datapoints(df, fixed_columns = ['interval', 'day', 'detid', 'weekday'], combine_on_column = 'interval', mean_column= 'traffic', ratio=1000):
    """
    Combines multiple data points for the same 'detid' into a single data point by taking the mean of the specified column.
    The 'interval' column is rounded to the nearest specified ratio before combining data points.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    fixed_columns (list, optional): The columns to group by when combining data points. Default is ['interval', 'day', 'detid', 'weekday'].
    combine_on_column (str, optional): The name of the column to round and combine data points on. Default is 'interval'.
    mean_column (str, optional): The name of the column to calculate the mean for. Default is 'traffic'.
    ratio (int, optional): The ratio to round the combine_on_column to. Default is 1000.

    Returns:
    pandas.DataFrame: A DataFrame with the data points combined by taking the mean of the specified column.
    """
    
    df[combine_on_column] = (df[combine_on_column] / ratio).round() * 1000

    df = df.groupby(fixed_columns).mean(mean_column).reset_index()
    
    return df

def merge_dataframes_on_detid(df1, df2, merge_column='detid', include_columns=['lanes', 'long', 'lat', 'pos', 'length']):
    """
    Merge two DataFrames on the specified column and include only the specified columns from the second DataFrame.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    merge_column (str): The column name to merge on.
    include_columns (list): The list of column names to include from the second DataFrame.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    if merge_column in df1.index.names:
        df1 = df1.reset_index(drop=True)

    if merge_column in df2.index.names:
        df2 = df2.reset_index(drop=True)

    # Ensure the merge_column is included in the list of columns to include
    columns_to_include = [merge_column] + include_columns

    merged_df = df1.merge(df2[columns_to_include], on=merge_column, how='left')
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
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The modified DataFrame.
    """
    columns_to_drop = ["lanes", "occ", "flow", "city"]
    
    df.loc[:, 'traffic'] = df['traffic'].fillna(0).astype(int)
    
    df_modified = df.drop(columns_to_drop, axis=1)
    
    return df_modified
