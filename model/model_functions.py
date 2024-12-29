def unfold_weekday_to_interval(df):
    """
    Convert weekday and interval columns into a single continuous interval column.

    This function transforms the 'weekday' and 'interval' columns in a DataFrame into a single 
    'interval' column representing the total number of seconds elapsed since the beginning 
    of the week. The 'weekday' column is dropped after transformation.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame with the following columns:
                           - 'weekday': Day of the week as a string (e.g., 'Monday', 'Tuesday', etc.).
                           - 'interval': Time interval in seconds from the start of the day.

    Returns:
        pd.DataFrame: A DataFrame with the transformed 'interval' column and the 'weekday' column removed.

    Example:
        Input DataFrame:
            weekday    interval
            Monday     3600
            Tuesday    7200
        Output DataFrame:
            interval
            3600
            93599  # 7200 + (1 * 86400)
    """
    
    weekday_to_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    seconds_per_day = 86400
    df['interval'] = df.apply(lambda row: row['interval'] + (weekday_to_num[row['weekday']] * seconds_per_day), axis=1)
    df = df.drop(columns=['weekday'])
    return df

def filter_full_hours(df, interval_column='interval'):
    """
    Filter rows where the interval corresponds to a full hour.

    This function filters the DataFrame to include only rows where the specified interval 
    represents a full hour (e.g., 0, 3600, 7200, etc.). Rows where the interval does not 
    correspond to a full hour are excluded.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing the interval data.
        interval_column (str, optional): The name of the column containing interval values (in seconds). Defaults to 'interval'.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the interval corresponds to a full hour.

    Example:
        Input DataFrame:
            interval  value
            3600      10
            4500      15
            7200      20

        Output DataFrame:
            interval  value
            3600      10
            7200      20
    """
    
    df['interval_in_hours'] = df[interval_column] / 3600
    filtered_df = df[df['interval_in_hours'] % 1 == 0].copy()
    filtered_df = filtered_df.drop(columns=['interval_in_hours'])
    
    return filtered_df
