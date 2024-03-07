# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px
import plotly.io as pio
import dash_bootstrap_components as dbc
import sys, json


# Command-Line Arguments: fig files, their path, layout size (default 800x700)
arg_options = json.loads(sys.argv[1])

if len(sys.argv) >= 3:
    arg_path = sys.argv[2]
else:
    arg_path = '.'

if len(sys.argv) == 4:
    arg_width, arg_height = [int(x) for x in sys.argv[3].split('x')]
else:
    arg_width, arg_height = 800, 700


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)


# App layout
app.layout = dbc.Container([
    html.Br(),

    dbc.Row(
        html.Div('Topic Distribution', className="text-primary text-center fs-5")
    ),

    dbc.Row(
        dbc.Col(
            dcc.Dropdown(options=arg_options, value=arg_options[0]['value'], id="topics"),
            #width="auto"
            width=3,
            ),
        justify="center"
    ),

    dbc.Row(
        dcc.Graph(figure={}, id='topic_distribution')
    ),

], fluid=True)


# Add controls to build the interaction
@callback(
    Output(component_id='topic_distribution', component_property='figure'),
    Input(component_id='topics', component_property='value')
)
def plot_topic_distr(file):
    f = f'{arg_path}/{file}'
    fig = pio.read_json(f)
    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=False,
            jupyter_width=arg_width, #"70%"
            jupyter_height=arg_height, #"70%"
            )