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

def split_data_sniper(df, samples_per_day = 2):
    """
    Splits the DataFrame into training and testing sets based on random intervals.

    This function groups the DataFrame by 'day' and 'detid', then randomly selects a specified number
    of intervals per day for each sensor to create the test set. The remaining data forms the training set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    samples_per_day (int): The number of intervals to randomly select per day for the test set. Default is 2.

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
        intervals = np.random.choice(range(24), samples_per_day, replace=False) * 3600
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
    1. Counts and removes rows where the 'error' column is equal to 1.
    2. Drops the 'error' and 'speed' columns.
    3. Converts the 'day' column to datetime format.
    4. Adds a new column 'weekday' with the name of the day of the week.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        tuple:
            - pandas.DataFrame: The preprocessed DataFrame.
            - int: The number of rows removed due to errors.
    """
    errors = df['error'].value_counts().get(1, 0)
    df = df[df['error'] != 1]
    df = df.drop(["error", "speed"], axis=1)

    df['day'] = pd.to_datetime(df['day'])
    df['weekday'] = df['day'].dt.day_name()
    
    return df, errors

def calculate_traffic_speed(df, flow_column='flow', occ_column='occ', traffic_column='traffic', min_occ_value = 0.001):
    """
    Calculates the traffic speed and adds it as a new column to the DataFrame.

    This function calculates the traffic speed using the formula speed = flow * occupancy
    and adds the result as a new column to the DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    flow_column (str): The name of the column representing the flow. Default is 'flow'.
    occ_column (str): The name of the column representing the occupancy. Default is 'occ'.
    traffic_column (str): The name of the new column to store the calculated traffic speed. Default is 'traffic'.
    min_occ_value (float): The minimum value to set for occupancy. Default is 0.001.

    Returns:
    pandas.DataFrame: The DataFrame with the new traffic speed column added.
    """
    df[occ_column] = df[occ_column].clip(lower=min_occ_value)
    
    df[traffic_column] = df[flow_column] * df[occ_column]
    return df

def calculate_interval_stats(group, column, num_intervals):
    """
    Calculate mean and Interquartile Range (IQR) statistics for each interval within a group.

    This function divides the provided group into a specified number of intervals and calculates
    the mean, first quartile (Q1), third quartile (Q3), and IQR for each interval of the specified column.

    Parameters:
        group (pandas.DataFrame): The DataFrame group to process.
        column (str): The name of the column to compute statistics for.
        num_intervals (int): The number of intervals to divide the group into.

    Returns:
        tuple:
            - list: A list of mean values for each interval.
            - list: A list of tuples, each containing (Q1, Q3, IQR) for the corresponding interval.
    """
    interval_size = len(group) // num_intervals
    means = []
    bounds = []
    
    for i in range(num_intervals):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size if i < num_intervals - 1 else len(group)
        interval_data = group.iloc[start_idx:end_idx][column]
        
        # Calculate interval statistics
        interval_mean = interval_data.mean()
        Q1 = interval_data.quantile(0.25)
        Q3 = interval_data.quantile(0.75)
        IQR = Q3 - Q1
        
        means.append(interval_mean)
        bounds.append((Q1, Q3, IQR))
    
    return means, bounds

def clip_group(group, column, outlier_factor, num_intervals):
    """
    Clips outliers in a DataFrame group using interval-specific IQR bounds.

    This function divides the provided group into a specified number of intervals and calculates
    the mean and Interquartile Range (IQR) for each interval of the specified column.
    Outliers are identified based on the interval-specific IQR bounds and are replaced with
    the corresponding interval mean.

    Parameters:
        group (pandas.DataFrame): The DataFrame group to process.
        column (str): The name of the column to process for outliers.
        outlier_factor (float): The multiplier for the IQR to define the bounds for detecting outliers.
        num_intervals (int): The number of intervals to divide the group into for calculating statistics.

    Returns:
        tuple:
            - pandas.DataFrame: The DataFrame with outliers clipped.
            - int: The total number of outliers that were replaced.
    """
    interval_size = len(group) // num_intervals
    means, bounds = calculate_interval_stats(group, column, num_intervals)
    outliers = 0
    
    # Create copy to avoid modifying original
    group = group.copy()
    
    for i in range(num_intervals):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size if i < num_intervals - 1 else len(group)
        
        Q1, Q3, IQR = bounds[i]
        lower_bound = Q1 - outlier_factor * IQR
        upper_bound = Q3 + outlier_factor * IQR
        
        # Identify and replace outliers for this interval
        interval_mask = (group.index >= start_idx) & (group.index < end_idx)
        outliers_mask = interval_mask & ((group[column] < lower_bound) | (group[column] > upper_bound))
        outliers += outliers_mask.sum()
        
        # Replace outliers with interval mean
        group.loc[outliers_mask, column] = means[i]
    
    return group, outliers

def clip_outliers(df, column, group_by_detid=False, outlier_factor=3, num_intervals=24):
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
    
    Prints:
        Total outliers clipped: int
    """
    total_outliers = 0
    if group_by_detid:
        grouped = df.groupby('detid')
        clipped_groups = []
        for _, group in grouped:
            clipped_group, outliers = clip_group(group, column, outlier_factor, num_intervals)
            total_outliers += outliers
            clipped_groups.append(clipped_group)
        df = pd.concat(clipped_groups).reset_index(drop=True)
    else:
        df, outliers = clip_group(df, column, outlier_factor, num_intervals)
        total_outliers += outliers
    
    print(f"Total outliers clipped: {total_outliers}")
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

    Prints:
        Total outliers dropped: int
    """
    total_outliers = 0
    if group_by_detid:
        grouped = df.groupby('detid')
        filtered_groups = []
        for name, group in grouped:
            filtered_group, outliers = drop_group(group, column, outlier_factor)
            total_outliers += outliers
            filtered_groups.append(filtered_group)
        df = pd.concat(filtered_groups).reset_index(drop=True)
    else:
        df, total_outliers = drop_group(df, column, outlier_factor)
    
    print(f"Total outliers dropped: {total_outliers}")
    return df

def clip_to_high_values(df, column='traffic', threshold= 500):
    """
    Clips the values in the specified column of the DataFrame to a maximum threshold.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to clip.
        threshold (int or float): The maximum value to which the column's values will be clipped.

    Returns:
        pandas.DataFrame: The DataFrame with values in the specified column clipped to the threshold.
    """
    df[column] = df[column].clip(upper=threshold)
    return df

def drop_group(group, column, outlier_factor):
    """
    Removes outliers from a DataFrame based on the Interquartile Range (IQR) method.

    Parameters:
        group (pandas.DataFrame): A DataFrame group containing traffic data with the specified column.
        column (str): The name of the column to check for outliers.
        outlier_factor (float, optional): The multiplier for the IQR to define the bounds for detecting outliers. Default is 3.

    Returns:
        tuple:
            pandas.DataFrame: A DataFrame with the outliers removed.
            int: The number of outliers removed.
    """
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_factor * IQR
    upper_bound = Q3 + outlier_factor * IQR
    
    # Filter out the outliers
    initial_count = len(group)
    filtered_group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
    outliers = initial_count - len(filtered_group)

    return filtered_group,outliers

def drop_false_values(df, column, outlier_factor=5):
    """
    Removes outliers from a DataFrame based on the value counts of a specified column using the Interquartile Range (IQR) method.
    The DataFrame is grouped by 'detid', and outliers are identified and removed within each group.

    Additionally, prints the total number of outliers detected and removed across all groups.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing traffic data.
        column (str): The name of the column to calculate value counts and identify outliers.
        outlier_factor (float, optional): The multiplier for the IQR to define the bounds for detecting outliers. Default is 5.

    Returns:
        pandas.DataFrame: A DataFrame with the outliers removed, grouped by 'detid'.
    """
    total_outliers_count = 0

    def drop_group_by_count(group, column, outlier_factor):
        nonlocal total_outliers_count
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

        outliers_count = outliers.shape[0]
        total_outliers_count += outliers_count
        
        # Drop the outliers from the group
        filtered_group = group[~group[column].isin(outliers[column])]
        
        return filtered_group
    
    # Group by 'detid' and apply the drop_group_by_count function
    filtered_df = df.groupby('detid').apply(drop_group_by_count, column=column, outlier_factor=outlier_factor).reset_index(drop=True)

    print(f"Total outliers detected and removed: {total_outliers_count}")
    
    return filtered_df

def drop_false_values_by_date(df, column):
    """
    Deletes false values from a DataFrame based on the value counts of a specified column.
    This function groups the DataFrame by 'day' before applying the outlier detection and removal process.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    column (str): The name of the column to calculate value counts and identify outliers.

    Returns:
    pandas.DataFrame: A DataFrame with the outliers removed.
    """
    total_outliers_count = 0

    def drop_by_group(group):
        nonlocal total_outliers_count
        # Count the occurrences of each unique value in the specified column
        value_counts = group[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        # Identify the outliers
        outliers = value_counts[value_counts['count'] > 250]

        outliers_count = outliers.shape[0] * 288
        total_outliers_count += outliers_count

        # Drop the outliers from the group
        filtered_group = group[~group[column].isin(outliers[column])]
        
        return filtered_group

    # Group by 'day' and apply the drop_by_group function
    filtered_df = df.groupby(['day', 'detid']).apply(drop_by_group).reset_index(drop=True)
    
    print(f"Total outliers detected and removed: {total_outliers_count}")
    return filtered_df

def detect_anomalies(df, column = 'traffic', factor=3, min_IQR=5, min_range=20, min_days=14, min_daily_records=250):
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
    pandas.DataFrame: A DataFrame containing the 'detid' values identified as anomalies with boolean columns indicating the type of anomaly:
        - 'mean_out_of_bound': True if the mean traffic is out of the defined bounds.
        - 'IQR_to_small': True if the IQR is smaller than the minimum threshold.
        - 'not_enough_data': True if there are not enough data points.
    """
    anomalies_mean_out_of_bound_list = anomalies_mean_out_of_bound(df, column, factor)
    anomalies_IQR_to_small_list = anomalies_IQR_to_small(df, column, min_IQR=min_IQR, min_range=min_range)
    anomalies_not_enough_data_list = anomalies_not_enough_data(df, min_days=min_days, min_daily_records=min_daily_records)
    anomalies = np.concatenate([anomalies_mean_out_of_bound_list, anomalies_IQR_to_small_list, anomalies_not_enough_data_list])
    anomalies = np.unique(anomalies)

    print(f"Total anomalies detected: {anomalies.size}")

    dataframe_anomalies = pd.DataFrame(anomalies, columns=['detid'])

    dataframe_anomalies['mean_out_of_bound'] = dataframe_anomalies['detid'].isin(anomalies_mean_out_of_bound_list)
    dataframe_anomalies['IQR_to_small'] = dataframe_anomalies['detid'].isin(anomalies_IQR_to_small_list)
    dataframe_anomalies['not_enough_data'] = dataframe_anomalies['detid'].isin(anomalies_not_enough_data_list)
    
    return dataframe_anomalies

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

def anomalies_IQR_to_small(df, column='traffic', min_IQR=5, min_range=20):
    """
    Identifies 'detid' values in a DataFrame where either:
    - The Interquartile Range (IQR) is below a given threshold
    - The total range (max-min) is below a given threshold

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing traffic data.
    column (str, optional): The name of the column to analyze. Default is 'traffic'.
    min_IQR (float, optional): The IQR threshold to check against. Default is 5.
    min_range (float, optional): The minimum range threshold. Default is 20.

    Returns:
    numpy.ndarray: An array containing the 'detid' values that meet both condition.
    """
    
    def calculate_metrics(group):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        total_range = group[column].max() - group[column].min()
        return pd.Series({'IQR': IQR, 'total_range': total_range})
    
    metrics = df.groupby('detid').apply(calculate_metrics).reset_index()
    
    anomalies = metrics[
        (metrics['IQR'] < min_IQR) & 
        (metrics['total_range'] < min_range)
    ]['detid'].unique()
    
    print(f"Anomalies detected based on IQR or range conditions: {anomalies.size}")
    
    return anomalies

def anomalies_not_enough_data(df, min_days=14, min_daily_records=250):
    """
    Detects detectors with insufficient data using vectorized operations.
    
    Parameters:
        df (pandas.DataFrame): Input DataFrame with traffic data
        min_days (int): Minimum number of days required (default: 14)
        min_daily_records (int): Minimum records per day (default: 250)
    
    Returns:
        numpy.ndarray: Array of detector IDs that have insufficient data
    """
    # Set detid as index for faster groupby operations
    df = df.set_index('detid')
    
    # Calculate daily records count using vectorized operations
    daily_counts = df.groupby(['detid', 'day']).size().unstack()
    valid_days = (daily_counts >= min_daily_records).sum(axis=1)
    
    # Calculate unique weekdays per detector
    weekday_counts = df.groupby('detid')['weekday'].nunique()
    
    # Find anomalous detectors using boolean indexing
    anomalies = daily_counts[
        (valid_days < min_days) | 
        (weekday_counts < 7)
    ].index.values
    
    print(f"Anomalies not enough data: {len(anomalies)}")
    
    return anomalies

def handle_anomalies(df, anomalies_df):
    """
    Processes and filters anomalies in the provided DataFrame.

    This function handles detectors with bad days, filters out specific anomalies,
    removes them from the original DataFrame, and prints the total number of
    dropped anomalies.

    Parameters:
        df (pd.DataFrame): The main DataFrame to process.
        anomalies_df (pd.DataFrame): DataFrame containing identified anomalies.

    Returns:
        tuple: 
            - pd.DataFrame: The original DataFrame with specified anomalies removed.
            - pd.DataFrame: Filtered DataFrame containing the handled anomalies.
    """
    handled_anomalies_df = handle_detectors_with_bad_days(df, anomalies_df)

    handled_anomalies_df = handled_anomalies_df[
    handled_anomalies_df[['mean_out_of_bound', 'IQR_to_small', 'not_enough_data']].any(axis=1)
    ]
    df = df[~df['detid'].isin(handled_anomalies_df['detid'])]
    print(f"Total amount of dropeed anomalies: {len(handled_anomalies_df)}")
    
    return df,handled_anomalies_df

def handle_detectors_with_bad_days(df, anomalies_df):
    """
    Processes detectors flagged for insufficient data by removing days with too few data points
    and updating their anomaly status based on the remaining data.

    This function filters the input DataFrame to include only detectors marked with the
    'not_enough_data' anomaly. For each of these detectors, it removes days that have fewer
    than 250 data points. If a detector retains at least 14 days and has data for all 7 weekdays
    after this removal, the 'not_enough_data' flag for that detector is set to False in the
    anomalies DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing traffic data. Must include 'detid', 'day', and 'weekday' columns.
        anomalies_df (pandas.DataFrame): The DataFrame containing detected anomalies. Must include 'detid' and 'not_enough_data' columns.

    Returns:
        pandas.DataFrame: The updated anomalies DataFrame with 'not_enough_data' flags adjusted based on the data processing.
    """
    detectors_not_enough_data = anomalies_df[anomalies_df['not_enough_data']]['detid'].unique()
    df = df[df['detid'].isin(detectors_not_enough_data)]
    anomalies_handled = 0

    detids = df['detid'].unique()
    for detid in detids:
        detector_df = df[df['detid'] == detid]
        day_counts = detector_df['day'].value_counts()
        valid_days = day_counts[day_counts >= 230].index
        filtered_df = detector_df[detector_df['day'].isin(valid_days)]

        days = filtered_df['day'].nunique()
        weekdays = filtered_df['weekday'].nunique()
        if days >= 10 and weekdays == 7:
            anomalies_df.loc[anomalies_df['detid'] == detid, 'not_enough_data'] = False
            anomalies_handled += 1
    print(f"Anomalies with not enough data handled: {anomalies_handled}")
    return anomalies_df

def combine_datapoints(df, fixed_columns = ['interval', 'day', 'detid', 'weekday'], combine_on_column = 'interval', mean_column= 'traffic', ratio=3600):
    """
    Combines multiple data points for the same 'detid' into a single data point by calculating the mean of the specified column.
    The values in the 'combine_on_column' are rounded to the nearest multiple of 'ratio' before combining.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing traffic data.
        fixed_columns (List[str], optional): Columns to group by when combining data points. Defaults to ['interval', 'day', 'detid', 'weekday'].
        combine_on_column (str, optional): Column to round and combine data points on. Defaults to 'interval'.
        mean_column (str, optional): Column to calculate the mean for. Defaults to 'traffic'.
        ratio (int, optional): The value to round 'combine_on_column' to. Defaults to 3600.

    Returns:
        pandas.DataFrame: DataFrame with combined data points, averaging the specified column.
    """
    
    df[combine_on_column] = (df[combine_on_column] / ratio).round() * ratio

    df = df.groupby(fixed_columns).mean(mean_column).reset_index()
    
    return df

def merge_dataframes_on_detid(df1, df2, merge_column='detid', include_columns=['lanes', 'long', 'lat', 'pos', 'length', 'fclass']):
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
    
    print(f'traffic range was between:{min_traffic} and {max_traffic}')

    df[traffic_column] = ((df[traffic_column] - min_traffic) / (max_traffic - min_traffic)) * (max_range - min_range) + min_range
    return df

def final_process_dataframe(df):
    """
    Convert the scaled values by rounding to the nearest integer, fill NaN values with 0, and drop specified columns.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    columns_to_drop = ["lanes", "occ", "flow"]
    
    df['traffic'] = df['traffic'].fillna(0).round().astype(int)
    
    df_modified = df.drop(columns_to_drop, axis=1)
    
    return df_modified
