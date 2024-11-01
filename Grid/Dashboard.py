# Begin abfrage ob kunde oder dev
#dev vergleicht mean traffic wert immer noch mit baselane und gibt an wie gut diese model/data oder was man verändert hat performet hat.
### Zuerst sollte nach dem path zu den CSV gefragt werden, wenn neue Datenkommen, kann dies so angepasst werden. Evtl. mit Standardvalues


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import folium
from datetime import datetime
import createmap_for_sensors
import gridfunction_round
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def plot_all_sensors():
    createmap_for_sensors.create_map()

def plot_london_map():
    def submit():
        try:
            selected_date = datetime.strptime(date_entry.get(), '%Y-%m-%d')
            selected_time = datetime.strptime(time_entry.get(), '%H:%M').time()
            gridfunction_round.plot_london(selected_date, selected_time)
            messagebox.showinfo("Success", "Map generated successfully!")
        except ValueError:
            messagebox.showerror("Error", "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time.")
    
    date_window = tk.Toplevel(root)
    date_window.title("Select Date and Time")

    tk.Label(date_window, text="Date (YYYY-MM-DD):").grid(row=0, column=0)
    date_entry = tk.Entry(date_window)
    date_entry.grid(row=0, column=1)

    tk.Label(date_window, text="Time (HH:MM):").grid(row=1, column=0)
    time_entry = tk.Entry(date_window)
    time_entry.grid(row=1, column=1)

    submit_button = tk.Button(date_window, text="Submit", command=submit)
    submit_button.grid(row=2, columnspan=2)

root = tk.Tk()
root.title("Traffic Status Dashboard")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(main_frame, text="Choose an option:").grid(row=0, column=0, columnspan=2)

plot_sensors_button = ttk.Button(main_frame, text="Plot All Sensors", command=plot_all_sensors)
plot_sensors_button.grid(row=1, column=0)

plot_london_button = ttk.Button(main_frame, text="Plot London Map", command=plot_london_map)
plot_london_button.grid(row=1, column=1)

root.mainloop()
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Traffic Status Dashboard"),
    dcc.RadioItems(
        id='option-selector',
        options=[
            {'label': 'Plot All Sensors', 'value': 'all_sensors'},
            {'label': 'Plot London Map', 'value': 'london_map'}
        ],
        value='all_sensors'
    ),
    html.Div(id='input-container'),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-container')
])

@app.callback(
    Output('input-container', 'children'),
    Input('option-selector', 'value')
)
def update_input_container(selected_option):
    if selected_option == 'london_map':
        return html.Div([
            dcc.Input(id='date-input', type='text', placeholder='YYYY-MM-DD'),
            dcc.Input(id='time-input', type='text', placeholder='HH:MM')
        ])
    return html.Div()

@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('option-selector', 'value'),
    Input('date-input', 'value'),
    Input('time-input', 'value')
)
def update_output(n_clicks, selected_option, date_value, time_value):
    if n_clicks > 0:
        if selected_option == 'all_sensors':
            createmap_for_sensors.create_map()
            return "All sensors map generated successfully!"
        elif selected_option == 'london_map':
            try:
                selected_date = datetime.strptime(date_value, '%Y-%m-%d')
                selected_time = datetime.strptime(time_value, '%H:%M').time()
                gridfunction_round.plot_london(selected_date, selected_time)
                return "London map generated successfully!"
            except ValueError:
                return "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)