def unfold_weekday_to_interval(df):
    """
    Adjusts the 'interval' column in the DataFrame by adding the number of seconds corresponding to the day of the week.

    This function converts the 'weekday' column to a numerical representation (0 for Monday, 1 for Tuesday, etc.),
    multiplies it by the number of seconds in a day (86400), and adds this value to the 'interval' column. The 'weekday'
    column is then dropped from the DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing 'interval' and 'weekday' columns.

    Returns:
    pandas.DataFrame: The modified DataFrame with the updated 'interval' column and without the 'weekday' column.
    """
    weekday_to_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    seconds_per_day = 86400
    df['interval'] = df.apply(lambda row: row['interval'] + (weekday_to_num[row['weekday']] * seconds_per_day), axis=1)
    df = df.drop(columns=['weekday'])
    return df

def filter_full_hours(df, interval_column='interval'):
    """
    Filters the DataFrame to include only rows where the interval column represents full hour intervals.
    This function adds a temporary column 'interval_in_hours' to the DataFrame, which is the interval column
    divided by 3600 (to convert seconds to hours). It then filters the DataFrame to include only rows where
    'interval_in_hours' is an integer (i.e., the interval is a full hour). The temporary column is dropped
    before returning the filtered DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the interval data.
    interval_column (str): The name of the column in the DataFrame that contains the interval data in seconds.
                           Default is 'interval'.
    Returns:
    pandas.DataFrame: A new DataFrame containing only the rows where the interval is a full hour.
    """
    
    df['interval_in_hours'] = df[interval_column] / 3600
    filtered_df = df[df['interval_in_hours'] % 1 == 0].copy()
    filtered_df = filtered_df.drop(columns=['interval_in_hours'])
    
    return filtered_df
