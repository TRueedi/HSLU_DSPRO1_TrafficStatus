import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import grid_functions  # Importiere die Funktionen aus grid_functions.py
import folium
import base64
from io import BytesIO

# Funktion zum Plotten des Grids
def plot_grid(df_weekday, hour, mode, center, zoom):
    interval_value = hour * 3600  # Umrechnung von Stunden in Sekunden
    
    if mode == 'sensors':
        map_object = grid_functions.plot_sensors_as_points(city_center=center, zoom_start=zoom)
    elif mode == 'traffic':
        grid_data = grid_functions.get_hour_prediction(df_weekday, interval_value, mode, shape=0.01)
        map_object = grid_functions.plot_grid_with_shapes(grid_data, shape='rectangle', city_center=center, zoom_start=zoom)
    elif mode == 'traffic_complete':
        grid_data, complete_grid = grid_functions.get_hour_prediction(df_weekday, interval_value, mode, shape=0.01)
        advanced_grid = grid_functions.advanced_grid(grid_data, complete_grid, sensorid_col='detid', trafficIndex_col='traffic', shape=0.01)
        map_object = grid_functions.plot_grid_with_shapes(advanced_grid, shape='rectangle', city_center=center, zoom_start=zoom)
    else:
        raise ValueError("Invalid mode. Choose 'sensors', 'traffic' or 'traffic_complete'.")
    
    return map_object

# Funktion zum Konvertieren der Folium-Karte in HTML
def map_to_html(map_object):
    map_html = map_object._repr_html_()
    return map_html

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='map-center', data={'lat': 51.480, 'lng': -0.081}),  # Speichert die Kartenmitte
    dcc.Store(id='map-zoom', data=12.3),  # Speichert den Zoom-Level
    html.Div(id='map', style={'position': 'absolute', 'top': '0', 'left': '0', 'right': '0', 'bottom': '0', 'z-index': '1'}),
    html.Div(id='weekday-data', style={'display': 'none'}),  # Verstecktes Div-Element zum Speichern der Vorhersage
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
                    {'label': 'Sensors', 'value': 'sensors'},
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
                value=0,  # Standardmäßig Montag ausgewählt
                labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': 'white'}
            ),
        ], style={'display': 'flex', 'align-items': 'center'}),
        html.Div([
            html.Label("Select Hour:", style={'color': 'white', 'margin-top': '10px'}),
            dcc.Slider(
                id='hour-slider',
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

@app.callback(
    Output('weekday-data', 'children'),
    Input('weekday-radio', 'value'),
    Input('model-radio', 'value')
)
def update_weekday_data(weekday, model):
    df_weekday = grid_functions.get_weekday_prediction(weekday, model)
    return df_weekday.to_json(date_format='iso', orient='split')

@app.callback(
    Output('map', 'children'),
    [Input('hour-slider', 'value'),
     Input('weekday-data', 'children'),
     Input('mode-radio', 'value'),
     State('map-center', 'data'),
     State('map-zoom', 'data')]
)
def update_map(hour, weekday_data, mode, center, zoom):
    if weekday_data:
        df_weekday = pd.read_json(weekday_data, orient='split')
        map_object = plot_grid(df_weekday, hour, mode, center=(center['lat'], center['lng']), zoom=zoom)
        map_html = map_to_html(map_object)
        return html.Iframe(srcDoc=map_html, width='100%', height='100%')
    return None

if __name__ == '__main__':
    app.run_server(debug=True)