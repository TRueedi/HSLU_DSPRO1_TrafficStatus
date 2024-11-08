# Begin abfrage ob kunde oder dev
#dev vergleicht mean traffic wert immer noch mit baselane und gibt an wie gut diese model/data oder was man verändert hat performet hat.
### Zuerst sollte nach dem path zu den CSV gefragt werden, wenn neue Datenkommen, kann dies so angepasst werden. Evtl. mit Standardvalues


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import grid_functions  # Importiere die Funktionen aus grid_functions.py
import folium
import base64
from io import BytesIO

# Funktion zum Plotten des Grids
def plot_grid(weekday, hour):
    df_weekday = grid_functions.get_weekday_prediction(weekday)
    interval_value = hour * 3600  # Umrechnung von Stunden in Sekunden
    grid_data = grid_functions.get_hour_prediction(df_weekday, interval_value)
    map_object = grid_functions.plot_grid_with_shapes(grid_data, shape='rectangle', city_center=(51.550, -0.021), zoom_start=15)


    return map_object

# Funktion zum Konvertieren der Folium-Karte in HTML
def map_to_html(map_object):
    map_html = map_object._repr_html_()
    return map_html

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Grid Plotter"),
    html.Label("Select Weekday:"),
    dcc.Dropdown(
        id='weekday-dropdown',
        options=[
            {'label': 'Monday', 'value': 0},
            {'label': 'Tuesday', 'value': 1},
            {'label': 'Wednesday', 'value': 2},
            {'label': 'Thursday', 'value': 3},
            {'label': 'Friday', 'value': 4},
            {'label': 'Saturday', 'value': 5},
            {'label': 'Sunday', 'value': 6}
        ],
        value=0
    ),
    html.Label("Select Hour:"),
    dcc.Slider(
        id='hour-slider',
        min=0,
        max=24,
        step=1,
        value=12,
        marks={i: f'{i}:00' for i in range(25)}
    ),
    html.Button('Plot Grid', id='plot-button', n_clicks=0),
    html.Div(id='map')
])

@app.callback(
    Output('map', 'children'),
    Input('plot-button', 'n_clicks'),
    Input('weekday-dropdown', 'value'),
    Input('hour-slider', 'value')
)
def update_map(n_clicks, weekday, hour):
    if n_clicks > 0:
        map_object = plot_grid(weekday, hour)
        map_html = map_to_html(map_object)
        return html.Iframe(srcDoc=map_html, width='100%', height='600')
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)