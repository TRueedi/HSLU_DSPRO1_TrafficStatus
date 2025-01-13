# Description: This script creates a Dash app to visualize the traffic status in London.
# By Running this script, a Dash app will be created and can be accessed via a web browser.
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import grid_functions
from flask_caching import Cache
import logging
import os

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# functions to plot the grid
def plot_grid(df_weekday, hour, mode, center, zoom):
    interval_value = hour * 3600  # Calculate the interval value from hours to seconds
    
    if mode == 'detectors':
        map_object = grid_functions.plot_detectors_as_points(city_center=center, zoom_start=zoom)
    elif mode == 'traffic':
        grid_data = grid_functions.get_hour_prediction(df_weekday, interval_value, mode, shape=0.01)
        map_object = grid_functions.plot_grid_with_shapes(grid_data, shape='rectangle', city_center=center, zoom_start=zoom)
    elif mode == 'traffic_complete':
        grid_data, complete_grid = grid_functions.get_hour_prediction(df_weekday, interval_value, mode, shape=0.01)
        advanced_grid = grid_functions.advanced_grid(grid_data, complete_grid, detectorid_col='detid', trafficIndex_col='traffic', shape=0.01)
        map_object = grid_functions.plot_grid_with_shapes(advanced_grid, shape='rectangle', city_center=center, zoom_start=zoom)
    else:
        raise ValueError("Invalid mode. Choose 'detectors', 'traffic' or 'traffic_complete'.")
    
    return map_object

# Function to convert the Folium map to HTML
def map_to_html(map_object):
    map_html = map_object._repr_html_()
    return map_html

# Dash App
app = dash.Dash(__name__)
server = app.server

# Configure the Flask-Caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': os.path.join(os.path.dirname(__file__), 'cache-directory'),
    'CACHE_DEFAULT_TIMEOUT': 10 * 365 * 24 * 60 * 60  # 10 years in Seconds
})

# function to precompute the predictions
def precompute_predictions():
    models = ['random', 'knn', 'rfr', 'prophet']
    weekdays = range(7)
    for model in models:
        for weekday in weekdays:
            cache_key = f"{model}_{weekday}"
            if not cache.get(cache_key):
                logging.info(f"Caching {model} model for weekday {weekday}")
                df_weekday = grid_functions.get_weekday_prediction(weekday, model)
                cache.set(cache_key, df_weekday.to_json(date_format='iso', orient='split'))
                logging.info(f"Cached {model} model for weekday {weekday} with Key {cache_key}")

app.layout = html.Div([
    dcc.Store(id='map-center', data={'lat': 51.480, 'lng': -0.081}),  # Save the center of the map
    dcc.Store(id='map-zoom', data=12.3),  # Save the zoom level of the map
    html.Div(id='map', style={'position': 'absolute', 'top': '0', 'left': '0', 'right': '0', 'bottom': '0', 'z-index': '1'}),
    html.Div(id='weekday-data', style={'display': 'none'}), 
    html.Div([
        html.H1("Traffic Status", style={'color': 'white', 'margin-right': '10px'}),
        html.Div([
            html.Label("Select Model:", style={'color': 'white', 'margin-right': '10px'}),
            dcc.RadioItems(
                id='model-radio',
                options=[
                    {'label': 'Baseline', 'value': 'random'},
                    {'label': 'knn', 'value': 'knn'},
                    {'label': 'rfr', 'value': 'rfr'},
                    {'label': 'prophet', 'value': 'prophet'}
                ],
                value='knn',
                labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': 'white'}
            )
        ], style={'display': 'flex', 'align-items': 'center'}),
        html.Div([
            html.Label("Select Mode:", style={'color': 'white', 'margin-top': '10px'}),
            dcc.RadioItems(
                id='mode-radio',
                options=[
                    {'label': 'Detectors', 'value': 'detectors'},
                    {'label': 'Traffic grid', 'value': 'traffic'},
                    {'label': 'Complete Traffic grid', 'value': 'traffic_complete'}
                ],
                value='traffic',
                labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': 'white'}
            )
        ], style={'display': 'flex', 'align-items': 'center'}),
        html.Div([
            html.Label("Select Weekday:", style={'color': 'white', 'margin-right': '10px'}),
            dcc.RadioItems(
                id='weekday-radio',
                options=[
                    {'label': 'Monday', 'value': 0},
                    {'label': 'Tuesday', 'value': 1},
                    {'label': 'Wednesday', 'value': 2},
                    {'label': 'Thursday', 'value': 3},
                    {'label': 'Friday', 'value': 4},
                    {'label': 'Saturday', 'value': 5},
                    {'label': 'Sunday', 'value': 6}
                ],
                value=0, # Default value: Monday
                labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': 'white'}
            ),
        ], style={'display': 'flex', 'align-items': 'center'}),
        html.Div([
            html.Label("Select Hour:", style={'color': 'white', 'margin-top': '10px'}),
            dcc.Slider(
                id='hour-slider',
                className='custom-hour-slider',
                min=0,
                max=24,
                step=1,
                value=12,
                marks={i: f'{i}:00' for i in range(25)},
                updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={'margin-top': '10px', 'width': '100%', 'color': 'white'}),
    ], style={'position': 'fixed', 'bottom': '0', 'left': '0', 'right': '0', 'width': '100%', 'background-color': 'rgba(0, 0, 0, 0.5)', 'padding': '10px', 'z-index': '1000', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
], style={'position': 'relative', 'height': '100vh', 'width': '100vw'})

# cache the callbacks
@cache.memoize()
# Callbacks to update the weekday data and the map
@app.callback(
    Output('weekday-data', 'children'),
    Input('weekday-radio', 'value'),
    Input('model-radio', 'value')
)
def update_weekday_data(weekday, model):
    cache_key = f"{model}_{weekday}"
    logging.info(f"Searching for: {cache_key}")
    df_weekday = cache.get(cache_key)
    if df_weekday:
        logging.info(f"Retrieved from cache: {cache_key}")
        return df_weekday
    else:
        logging.warning(f"Cache miss for key: {cache_key}")
        df_weekday = grid_functions.get_weekday_prediction(weekday, model)
        cache.set(cache_key, df_weekday.to_json(date_format='iso', orient='split'))
        return df_weekday.to_json(date_format='iso', orient='split')


@app.callback(
    Output('map', 'children'),
    [Input('hour-slider', 'value'),
     Input('weekday-data', 'children'),
     Input('mode-radio', 'value')],
    [State('map-center', 'data'),
     State('map-zoom', 'data')]
)
def update_map(hour, weekday_data, mode, center, zoom):
    if weekday_data:
        df_weekday = pd.read_json(weekday_data, orient='split')
        map_object = plot_grid(df_weekday, hour, mode, center=(center['lat'], center['lng']), zoom=zoom)
        map_html = map_to_html(map_object)
        return html.Iframe(srcDoc=map_html, width='100%', height='100%')
    return None

# Export the app
if __name__ == '__main__':
    precompute_predictions() # Precaching
    app.run_server(debug=False)