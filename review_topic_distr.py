# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px
import plotly.io as pio
import dash_bootstrap_components as dbc
import sys, json


# Command-Line Arguments: fig files, their path
arg_options = json.loads(sys.argv[1])

if len(sys.argv) >= 3:
    arg_path = sys.argv[2]
else:
    arg_path = '.'
    

# Initialize the app - incorporate a Dash Bootstrap theme
#external_stylesheets = [dbc.themes.FLATLY]
external_stylesheets=[dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

header = html.Div(
    [
        html.Div("Topic Distribution"),
        dcc.Dropdown(options=arg_options, value=arg_options[0]['value'], id="topics",
                     multi=True),
    ],
)

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(header, lg=6)),
        dbc.Row(dbc.Col(html.Div(id="graphs"))),
    ],
    className="dbc p-4",
    fluid=True,
)


# Add controls to build the interaction
@callback(
    Output(component_id="graphs", component_property="children"),
    Input(component_id='topics', component_property='value')
)
def plot_topic_distr(files):

    if not isinstance(files, list):
        files = [files]

    graphs = []
    for f in files:
        f = f'{arg_path}/{f}'
        fig = pio.read_json(f)
        g = dcc.Graph(figure=fig, className="border")
        graphs.append(g)

    layout = []
    cols = []
    for i, g in enumerate(graphs, start=1):
        cols.append(dbc.Col(g))
        if (i % 2 == 0) or (i == len(graphs)):
            layout.append(dbc.Row(cols, className="mt-2"))
            cols = []

    return layout


# Run the app
if __name__ == '__main__':
    app.run(debug=False,
            #jupyter_width=arg_width, #"70%"
            #jupyter_height=arg_height, #"70%"
            )
