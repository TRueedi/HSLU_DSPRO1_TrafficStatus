from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import branca.colormap as cm 
import folium
import geopandas


def get_random_baseline_prediction(models_path, weekday, interval_values=
                                   [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]):
    
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        if '_baseline' in model_filename:
            model_path = os.path.join(models_path, model_filename)
            sensor_baseline = joblib.load(model_path)
            
            # Generate random traffic values within the saved range
            y_pred = np.random.uniform(sensor_baseline['min'], sensor_baseline['max'], len(X_values))
            
            predictions.append(pd.DataFrame({
                'traffic': y_pred,
                'detid': model_filename.replace('_baseline', '').replace('-', '/'),
                'interval': X_values['interval']
            }))
    
    return pd.concat(predictions)


def get_knn_prediction(models_path, weekday, interval_values=[
               0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 
               36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 
               68400, 72000, 75600, 79200, 82800]):
    
    X_values = pd.DataFrame(interval_values, columns=['interval'])
    X_values['weekday'] = weekday
    
    predictions = []
    
    for model_filename in os.listdir(models_path):
        model_path = os.path.join(models_path, model_filename)
        if os.path.isfile(model_path):
            # Load the KNN model
            sensor_model = joblib.load(model_path)
            y_pred = sensor_model.predict(X_values)
            
            # Store predictions in DataFrame format
            predictions.append(pd.DataFrame({
                'traffic': y_pred,
                'detid': model_filename.replace('-', '/').replace('.pkl', ''),
                'interval': X_values['interval'],
            }))
        
    return pd.concat(predictions)


def grid(df, sensorid_col, trafficIndex_col, shape=0.01):
    """
    Input:
    - df: DataFrame containing sensor data with longitude and latitude
    - sensorid_col: column name for sensor ids
    - trafficIndex_col: column name for traffic indices (e.g. length or traffic volume)
    - shape: the size of the grid (diameter of the cell)
    
    Output:
    - A DataFrame with the grid and the mean trafficIndex for each grid cell.
    """
    # 1. Round the coordinates to a precision based on 'shape' (grid diameter)
    df['long_rounded'] = (df['long'] // shape) * shape
    df['lat_rounded'] = (df['lat'] // shape) * shape
    
    # 2. Create a grid ID based on the rounded coordinates
    df['grid_id'] = df['long_rounded'].astype(str) + "_" + df['lat_rounded'].astype(str)
    
    # 3. Calculate the mean of the trafficIndex for each grid and count sensors
    grid = df.groupby('grid_id').agg(
        mean_trafficIndex=(trafficIndex_col, 'mean'),
        sensors_in_grid=(sensorid_col, 'count'),
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
    #colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], 
    #                             vmin=grid['mean_trafficIndex'].min(), 
    #                             vmax=grid['mean_trafficIndex'].max(),
    #                             caption='Mean Traffic Index')
    
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=0,
        vmax=99,
        caption='Mean Traffic Index'
    )
    
    #colormap = cm.StepColormap(
    #    colors=['green', 'yellow', 'red'],  # Farben: grün -> gelb -> rot
    #    index=[grid['mean_trafficIndex'].min(), 25, 50, grid['mean_trafficIndex'].max()],
    #    vmin=grid['mean_trafficIndex'].min(),
    #    vmax=grid['mean_trafficIndex'].max(),
    #    caption='Mean Traffic Index')
        
    
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
                popup=f"Grid ID: {row['grid_id']}<br>Mean Traffic Index: {row['mean_trafficIndex']}<br>Sensors in Grid: {row['sensors_in_grid']}"
            ).add_to(m)
        
        elif polygon:
            # If the shape is a polygon (rectangle, triangle, octagon), use folium.Polygon
            folium.Polygon(
                locations=polygon,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"Grid ID: {row['grid_id']}<br>Mean Traffic Index: {row['mean_trafficIndex']}<br>Sensors in Grid: {row['sensors_in_grid']}"
            ).add_to(m)

    return m



def get_weekday_prediction(weekday):
    """
    This function generates a prediction for a given weekday using the baseline models.
    It returns a DataFrame with the predicted traffic values for each sensor.
    Weekday mapping:
    - Monday: 0
    - Tuesday: 1
    - Wednesday: 2
    - Thursday: 3
    - Friday: 4
    - Saturday: 5
    - Sunday: 6
    """
    df_sensors = pd.read_csv(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\HSLU_DSPRO1_TrafficStatus\data\RawDataLondon\London_detectors.csv")
        
    #df_weekday = get_random_baseline_prediction(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\baseline", weekday)
    df_weekday = get_knn_prediction(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\knn04", weekday)
    
    df_weekday_with_coords = pd.merge(df_weekday, df_sensors, on='detid', how='left')
    
    return df_weekday_with_coords



def get_hour_prediction(df, interval_value):
    """
    This function generates the baseline grids for all weekdays and intervals.
    possible Intervalls between: [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]
    
    
    """
    df_real = df[df['interval'] == interval_value]
    grid_data = grid(df_real, sensorid_col='detid', trafficIndex_col='traffic', shape=0.01)
    return grid_data




def saving_baseline_grids():
    """
    This function generates the baseline grids for all weekdays and intervals and saves them to CSV files.
    They than can be used for later tests.
    """
    
    
    df_sensors = pd.read_csv(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\HSLU_DSPRO1_TrafficStatus\data\RawDataLondon\London_detectors.csv")
    
    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    interval_values= [0, 3600, 7200,10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800]
    
    
    
    for x in range(7):
        df_weekday = get_random_baseline_prediction(r"C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\knn", x)
        df_weekday_with_coords = pd.merge(df_weekday, df_sensors, on='detid', how='left')

        for y in interval_values:
            df_real = df_weekday_with_coords[df_weekday_with_coords['interval'] == y]
            grid_data = grid(df_real, sensorid_col='detid', trafficIndex_col='traffic', shape=0.01)
            grid_data.to_csv(f"baselinegrids/{x}_{y}.csv", index=False)
