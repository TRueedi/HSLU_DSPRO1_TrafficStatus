# !pip install branca
# !pip install folium geopandas

import pandas as pd
import numpy as np
import branca.colormap as cm  # Used for color gradient
import folium
import geopandas


def __init__():
    pass


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
    colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], 
                                 vmin=grid['mean_trafficIndex'].min(), 
                                 vmax=grid['mean_trafficIndex'].max())
    
    colormap.caption = 'Mean Traffic Index'
    m.add_child(colormap)  # Add the colormap to the map

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


# User input
htmloutput = input("Enter if you want to save the output as html file (yes/no): ")

path = input("Enter the path with the filename to the sensor data file: ")
path_london_detectors = input("Enter the path with the filename to the London detectors file: ")
print("Loading data from: ", path + " and " + path_london_detectors)
df_sensors = pd.read_csv(path)
df_detectors = pd.read_csv(path_london_detectors)
print("Data loaded successfully!")

df_merged = pd.merge(df_sensors, df_detectors, on='detector_id', how='left')
print("Data merged with Coords successfully!")


#sensorid_col, trafficIndex_col = input("Enter the column names for sensor ids (standard = detid) and traffic indices (standard = traffic) separated by a comma: ").split(',')

grid_data = grid(df_merged, sensorid_col='detid', trafficIndex_col='traffic', shape=0.01)


map_with_rectangles = plot_grid_with_shapes(grid_data, shape='rectangle', city_center=(51.550, -0.021), zoom_start=15)

if htmloutput == 'yes':
    map_with_rectangles.save('london_grid_rectangles.html')
else:
    map_with_rectangles
