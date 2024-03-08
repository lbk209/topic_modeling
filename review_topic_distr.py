# Import packages
from dash import Dash, html, dcc, callback, Output, Input, State, ctx
import plotly.express as px
import plotly.io as pio
import dash_bootstrap_components as dbc
import sys, json
import io, re

import base64


# Command-Line Arguments: fig files, their path
arg_options = json.loads(sys.argv[1])

if len(sys.argv) >= 3:
    arg_path = sys.argv[2]
else:
    arg_path = '.'

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY]
#external_stylesheets=[dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

header =  html.Div("Topic Distribution")

dropdown = dcc.Dropdown(options=arg_options, value=arg_options[0]['value'], id="topics",
                     multi=True)

download = html.Div(
    [
        dbc.Button("Save", id='save-button',
                   color="secondary", className="me-2"),
        dcc.Upload(dbc.Button("Load", color="secondary", className="me-2"),
                   id='load-topics')

    ], className= "m-1"
)

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(header, lg=6)),
        dbc.Row([dbc.Col(dropdown, lg=6), dbc.Col(download)], align="center"),
        dbc.Row(dbc.Col(html.Div(id="loaded"))), # testing
        dbc.Row(dbc.Col(html.Div(id="graphs"))),
        #dbc.Row(dbc.Col(html.Div(id="save-topics"))), # testing
        dcc.Download(id="save-topics"),
        #dcc.Upload(id='load-topics')

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

    layout = 'figs: '  + ', '.join(files) # testing

    return layout


@callback(
    #Output('save-topics', 'children'),
    Output("save-topics", "data"),
    Input('save-button', 'n_clicks'),
    State('topics', 'value'),
    prevent_initial_call=True
)
def save_topics(n_clicks, files):
    if not isinstance(files, list):
        files = [files]
    layout = ', '.join(files) # testing
    #return f'{layout}'
    return dict(content=f'{layout}', filename="topics.txt")


@callback(#Output(component_id="loaded", component_property="children"),
          Output(component_id="topics", component_property="value"),
          Input('load-topics', 'contents'),
          State('load-topics', 'filename'),
          State('load-topics', 'last_modified'),
          prevent_initial_call=True)
def load_topics(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    files = decoded.decode('utf-8')
    files = re.findall(r'\w+\.json', files) # list of json files
    return files


# Run the app
if __name__ == '__main__':
    app.run(debug=True,
            #jupyter_width=arg_width, #"70%"
            #jupyter_height=arg_height, #"70%"
            )
