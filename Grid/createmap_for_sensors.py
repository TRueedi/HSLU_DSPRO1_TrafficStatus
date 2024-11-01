# This Script creates a map above London with the sensors out of the given Files.

#lib imports
import pandas as pd
import numpy as np
import branca.colormap as cm  # Used for color gradient
import folium
import geopandas

#User Inputs
anom_path = input("Enter the path to the anomalies file (Bsp. ..C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\Anomalies_18.10.2024.csv): ")
detectors_path = input("Enter the path to the detectors List file (Bsp. C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\HSLU_DSPRO1_TrafficStatus\data\RawDataLondon\London_detectors.csv): ")
usedsensors_path = input("Enter the path to the used sensors file (Bsp. C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\data\London_UTD19_Modified_23.10.2024.csv): ")
store_path = input("Enter the path to store the map (Bsp. C:\Users\rueed\OneDrive\HSLU\3 Semester\DSPRO 1\HSLU_DSPRO1_TrafficStatus\notebooks\grid\anomalies\Anomaliesmap_anomalies_18_10_2024.html): ")
plot_show = input("Do you want to show the plot (yes/no): ")

#Load Data
print("Data loading...")
df_sensors = pd.read_csv(detectors_path)
df_anomalies = pd.read_csv(anom_path)
df_usedsensors = pd.read_csv(usedsensors_path)
print("Data loaded successfully!")

#Merge Data
df_anom = pd.merge(df_anomalies, df_sensors, on='detid', how='left')
df_real = pd.merge(df_usedsensors, df_sensors, on='detid', how='left')

#Drop Duplicates
df_real.drop_duplicates(subset='detid', keep='first', inplace=True)


#Functions
def plot_sensors_as_points(df1_blue, df2_red, lat_col='lat', long_col='long', sensorid_col='detid', trafficIndex_col=None, city_center=(51.5074, -0.1278), zoom_start=12):
    """
    Plot the sensor locations as points (CircleMarkers) on a map of London.
    
    Args:
    - df1_blue and drf2_red: DataFrame containing sensor data with latitude and longitude columns.
    - lat_col: Name of the column containing latitude values (default: 'lat').
    - long_col: Name of the column containing longitude values (default: 'long').
    - sensorid_col: Name of the column containing sensor IDs (default: 'detid').
    - trafficIndex_col: Optional. Name of the column containing traffic index values (for popup display).
    - city_center: Tuple of (latitude, longitude) for the center of the map (default is central London).
    - zoom_start: Initial zoom level for the map (default is 12).
    
    Output:
    - Folium map with sensor locations plotted as CircleMarkers (points).
    """
    # Create a Folium map centered around London
    m = folium.Map(location=city_center, zoom_start=zoom_start)

    # Plot each sensor as a point on the map using CircleMarker
    for _, row in df1_blue.iterrows():
        # Popup text with sensor id and optional traffic index
        popup_text = f"Sensor ID: {row[sensorid_col]}"
        if trafficIndex_col:
            popup_text += f"<br>Traffic Index: {row[trafficIndex_col]}"
        
        # Add a CircleMarker for each sensor
        folium.CircleMarker(
            location=[row[lat_col], row[long_col]],
            radius=3,  # Size of the circle (points)
            color='blue',  # Outline color
            fill=True,
            fill_color='blue',  # Fill color of the point
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)
        
    for _, row in df2_red.iterrows():
        # Popup text with sensor id and optional traffic index
        popup_text = f"Sensor ID: {row[sensorid_col]}"
        if trafficIndex_col:
            popup_text += f"<br>Traffic Index: {row[trafficIndex_col]}"
        
        # Add a CircleMarker for each sensor
        folium.CircleMarker(
            location=[row[lat_col], row[long_col]],
            radius=3,  # Size of the circle (points)
            color='red',  # Outline color
            fill=True,
            fill_color='red',  # Fill color of the point
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    return m


#Create Plot
plot = plot_sensors_as_points(df_real, df_anom, lat_col='lat', long_col='long', sensorid_col='detid')

#Save Plot as HTML
plot.save(store_path)

#Show Plot
if plot_show == "yes":
    plot


