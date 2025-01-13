# Description: Functions for creating and plotting grids of traffic data.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import branca.colormap as cm 
import folium
from datetime import datetime, timedelta
from model import baseline_model as bm
from model import knn_model as knn
from model import prophet_model as pm
from model import rfr_model as rfr

#Functions for Grids
def grid(df, detectorid_col, trafficIndex_col, shape=0.01):
    """
    Create a grid with the mean trafficIndex for each grid cell.
    Input:
    - df: DataFrame containing detectors data with longitude and latitude
    - detectorid_col: column name for detectors ids
    - trafficIndex_col: column name for traffic indices (e.g. length or traffic volume)
    - shape: the size of the grid (diameter of the cell)
    
    Output:
    - A DataFrame with the grid and the mean trafficIndex for each grid cell.
    """
    decimal_places = abs(int(round(-np.log10(shape), 0)))
    
    df['long_rounded'] = df['long'].round(decimal_places)
    df['lat_rounded'] = df['lat'].round(decimal_places)
    
    # 2. Create a grid ID based on the rounded coordinates
    df['grid_id'] = df['long_rounded'].astype(str) + "_" + df['lat_rounded'].astype(str)
    
    # 3. Calculate the mean of the trafficIndex for each grid and count detectors
    grid = df.groupby('grid_id').agg(
        mean_trafficIndex=(trafficIndex_col, 'mean'),
        detectors_in_grid=(detectorid_col, 'count'),
        long_rounded=('long_rounded', 'first'),
        lat_rounded=('lat_rounded', 'first')
    ).reset_index()

    return grid


def create_polygon(lat, long, shape='circle', size=0.005):
    """
    Create a polygon with different shapes (rectangle, octagon, triangle) around a central point.
    
    Args:
    - lat: Latitude of the center
    - long: Longitude of the center
    - shape: 'circle', 'rectangle', 'octagon', 'triangle'
    - size: the size of the shape (for polygons, it determines the distance of vertices from the center)
    
    Returns:
    - A list of [lat, long] tuples representing the vertices of the polygon.
    """
    if shape == 'rectangle':
        # Return a square (approximate rectangle) around the center
        return [
            [lat - size, long - size],  # bottom-left
            [lat - size, long + size],  # bottom-right
            [lat + size, long + size],  # top-right
            [lat + size, long - size]   # top-left
        ]
    
    elif shape == 'triangle':
        # Return an equilateral triangle (upward facing)
        return [
            [lat + size, long],              # top
            [lat - size / 2, long - size],   # bottom-left
            [lat - size / 2, long + size]    # bottom-right
        ]
    
    elif shape == 'octagon':
        # Create an approximate octagon (8-sided polygon) around the center
        angle_offset = np.pi / 4  # 45 degrees per side
        return [
            [lat + size * np.cos(i * angle_offset), long + size * np.sin(i * angle_offset)]
            for i in range(8)
        ]
    
    else:
        # Default to a circle (using folium.Circle)
        return None  # No polygon, as Circle will be used in the main function

def plot_grid_with_shapes(grid, shape='circle', city_center=(51.5074, -0.1278), zoom_start=12):
    """
    Plot the grid over a map of London with various shapes (circle, rectangle, octagon, triangle).
    - Red indicates higher mean traffic index.
    - Green indicates lower mean traffic index.
    
    Args:
    - grid: DataFrame containing grid information with mean traffic index, rounded lat/long, and grid_id.
    - shape: Shape to use for plotting ('circle', 'rectangle', 'octagon', 'triangle')
    - city_center: Tuple of (latitude, longitude) for the center of the map (default is central London).
    - zoom_start: Initial zoom level for the map (default is 12).
    
    Output:
    - Folium map with grid visualized.
    """
    # Create a Folium map centered around London
    m = folium.Map(location=city_center, zoom_start=zoom_start)

    # Create a color map that interpolates between green (low) and red (high)  
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=0,
        vmax=99,
        caption='Mean Traffic Index'
    )
        
    
    m.add_child(colormap)  # Add the colormap to the map


    # Add custom CSS to adjust the size of the colormap legend
    custom_css = """
    <style>
    .legend {
        font-size: 16px;
        padding: 10px;
    }
    .legend .caption {
        font-size: 20px;
        font-weight: normal;
        margin-bottom: 20px;
    }
    .legend .colorbar {
        width: 300px;
        height: 20px;
        margin-bottom: 10px;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(custom_css))


    # Plot the grid cells on the map with the chosen shape
    for _, row in grid.iterrows():
        color = colormap(row['mean_trafficIndex'])
        
        # Determine the vertices for the given shape
        polygon = create_polygon(row['lat_rounded'], row['long_rounded'], shape=shape)
        
        
        if shape == 'circle':
            # If shape is 'circle', use folium.Circle
            folium.Circle(
                location=[row['lat_rounded'], row['long_rounded']],
                radius=500,  # 500 meters radius (adjustable)
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"Grid ID: {row['grid_id']}<br>Mean Traffic Index: {row['mean_trafficIndex']}<br>Detectors in Grid: {row['detectors_in_grid']}"
            ).add_to(m)
        
        elif polygon:
            # If the shape is a polygon (rectangle, triangle, octagon), use folium.Polygon
            folium.Polygon(
                locations=polygon,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"Grid ID: {row['grid_id']}<br>Mean Traffic Index: {row['mean_trafficIndex']}<br>Detectors in Grid: {row['detectors_in_grid']}",
            ).add_to(m)

    return m



def get_weekday_prediction(weekday, model=['random', 'knn', 'rfr','prophet']):
    """
    This function generates a prediction for a given weekday using the baseline models.
    It returns a DataFrame with the predicted traffic values for each Detector.
    Weekday mapping:
    - Monday: 0
    - Tuesday: 1
    - Wednesday: 2
    - Thursday: 3
    - Friday: 4
    - Saturday: 5
    - Sunday: 6
    """
    df_detectors = pd.read_csv(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\HSLU_DSPRO1_TrafficStatus\data\RawDataLondon\London_detectors.csv")
    
    if model == 'random':
        df_weekday = bm.get_random_baseline_prediction(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\DaySplit\baseline_day_1", weekday)
    elif model == 'knn':
        df_weekday = knn.get_knn_prediction(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\DaySplit\KNN_day_1", weekday)
    elif model == 'rfr':
        df_weekday = rfr.get_rfr_prediction(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\DaySplit\rfr_day_1", weekday)
    elif model == 'prophet':
        df_weekday = pm.get_prediction_per_sensor(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\DaySplit\prophet_day_1", weekday)
    
    df_weekday_with_coords = pd.merge(df_weekday, df_detectors, on='detid', how='left')
    
    return df_weekday_with_coords


def get_hour_prediction(df, interval_value, mode, shape=0.01):
    """
    This function generates the baseline grids for all weekdays and intervals.
    possible Intervalls between: [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]
    
    
    """
    df_real = df[df['interval'] == interval_value]
    grid_data = grid(df_real, detectorid_col='detid', trafficIndex_col='traffic', shape=0.01)
    
    if mode == 'traffic':
        return grid_data
    elif mode == 'traffic_complete':
        complete_grid = create_complete_grid(df, shape)
        return grid_data, complete_grid


def advanced_grid(grid_data, complete_grid, detectorid_col='detid', trafficIndex_col='traffic', shape=0.01):
    """
    Create a grid with the mean trafficIndex for each grid cell. The grid is filled with all possible grid cells
    based on the minimum and maximum longitude and latitude in the input DataFrame.
    Input:
    - df: DataFrame containing detector data with longitude and latitude
    - detectorid_col: column name for detector ids
    - trafficIndex_col: column name for traffic indices (e.g. length or traffic volume)
    - shape: the size of the grid (diameter of the cell)
    
    Output:
    - A DataFrame with the grid and the mean trafficIndex for each grid cell.
    """
    grid = grid_data.copy()
    
    grid_filled = complete_grid.merge(grid[['grid_id', 'mean_trafficIndex', 'detectors_in_grid']], on='grid_id', how='left')
    grid_filled['mean_trafficIndex'] = grid_filled['mean_trafficIndex_y']
    grid_filled.drop(columns=['mean_trafficIndex_y', 'mean_trafficIndex_x'], inplace=True)
    
    grid_complete = fill_nan_with_neighbors(grid_filled)
    
    return grid_complete

def fill_nan_with_neighbors(grid, radius=0.01, max_iterations=3, default_value=5.5):
    """
    Fills NaN values in the 'mean_trafficIndex' column iteratively with the mean of the neighbors
    within the radius. Remaining NaN values after max_iterations are assigned a default value.
    
    Parameters:
    - grid (pd.DataFrame): DataFrame containing the grid data.
    - radius (float): The radius to consider for the neighbors.
    - max_iterations (int): Maximum number of iterations to try filling NaN values.
    - default_value (float): Value to assign to remaining NaN values after max_iterations.
    
    Returns:
    - pd.DataFrame: DataFrame with NaN values filled.
    """
    df = grid.copy()
    
    def get_neighbors_mean(lat, long):
        """
        Returns the mean of the neighbors within the radius.
        """
        neighbors = df[
            (df['lat_rounded'] >= lat - radius) & (df['lat_rounded'] <= lat + radius) &
            (df['long_rounded'] >= long - radius) & (df['long_rounded'] <= long + radius)
        ]
        valid_neighbors = neighbors['mean_trafficIndex'].dropna()
        return valid_neighbors.mean() if not valid_neighbors.empty else np.nan

    for iteration in range(max_iterations):
        # Identify rows with NaN in 'mean_trafficIndex'
        nan_rows = df['mean_trafficIndex'].isna()
        if not nan_rows.any():
            # If there are no NaN values left, exit the loop
            break
        
        # Update NaN values with the mean of their neighbors
        df.loc[nan_rows, 'mean_trafficIndex'] = df[nan_rows].apply(
            lambda row: get_neighbors_mean(row['lat_rounded'], row['long_rounded']), axis=1
        )
    
    # Assign default value to any remaining NaN values
    df['mean_trafficIndex'].fillna(default_value, inplace=True)
    return df

def create_complete_grid(df, shape=0.01):
    """
    Creates a complete grid based on the minimum and maximum longitude and latitude in the input DataFrame.
    Input:
    - df: DataFrame containing detector data with longitude and latitude
    - shape: the size of the grid (diameter of the cell)
    
    Output:
    - A DataFrame with the complete grid and the mean trafficIndex for each grid cell.
    """
    
    decimal_places = abs(int(round(-np.log10(shape), 0)))
    
    df['long_rounded'] = df['long'].round(decimal_places)
    df['lat_rounded'] = df['lat'].round(decimal_places)
    
    # min max values
    min_long = df['long_rounded'].min()
    max_long = df['long_rounded'].max()
    min_lat = df['lat_rounded'].min()
    max_lat = df['lat_rounded'].max()
    
    # Grid from min to max in steps of shape 
    np.set_printoptions(suppress=True) 
    all_longs = np.arange(min_long, max_long, shape)
    all_lats = np.arange(min_lat, max_lat + shape, shape)
    
    
    # Create a complete grid
    complete_grid = pd.MultiIndex.from_product([all_longs, all_lats], names=['long_rounded', 'lat_rounded']).to_frame(index=False)
    
    # Create id for each grid cell
    complete_grid['grid_id'] = complete_grid.apply(
        lambda row: f"{round(row['long_rounded'], decimal_places)}_{round(row['lat_rounded'], decimal_places)}", axis=1
    )
    
    # set mean_trafficIndex to None
    complete_grid['mean_trafficIndex'] = None
    
    return complete_grid


def plot_detectors_as_points(detectorid_col='detid', trafficIndex_col=None, city_center=(51.5074, -0.1278), zoom_start=12):
    """
    Plot the detector locations as points (CircleMarkers) on a map of London.
    
    Args:
    - lat_col: Name of the column containing latitude values (default: 'lat').
    - long_col: Name of the column containing longitude values (default: 'long').
    - detectorid_col: Name of the column containing detector IDs (default: 'detid').
    - trafficIndex_col: Optional. Name of the column containing traffic index values (for popup display).
    - city_center: Tuple of (latitude, longitude) for the center of the map (default is central London).
    - zoom_start: Initial zoom level for the map (default is 12).
    
    Output:
    - Folium map with detector locations plotted as CircleMarkers (points).
    """
    df = pd.read_csv(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\London_UTD19_modified_22.11.2024.csv")
    
    df.drop_duplicates(subset='detid', keep='first', inplace=True)
    # Create a Folium map centered around London
    m = folium.Map(location=city_center, zoom_start=zoom_start)

    # Plot each detector as a point on the map using CircleMarker
    for _, row in df.iterrows():
        
        # Popup text with detector id and optional traffic index
        popup_text = f"Detector ID: {row[detectorid_col]}"
        if trafficIndex_col:
            popup_text += f"<br>Traffic Index: {row[trafficIndex_col]}"
        
        # Add a CircleMarker for each detector
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,  # Size of the circle (points)
            color='blue',  # Outline color
            fill=True,
            fill_color='blue',  # Fill color of the point
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    return m