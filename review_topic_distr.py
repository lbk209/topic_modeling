# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
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
#external_stylesheets=[dbc.themes.BOOTSTRAP]
external_stylesheets=[dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

header =  html.Div("Topic Distribution")

dropdown = dcc.Dropdown(options=arg_options, value=arg_options[0]['value'], id="topics",
                     multi=True)

download = html.Div(
    [
        dbc.Button("Save Topics", id='submit-button-topic', 
                   color="secondary", className="me-1", ),
        
    ], className= "m-1"
)

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(header, lg=6)),
        dbc.Row([dbc.Col(dropdown, lg=6), dbc.Col(download)], align="center"),
        dbc.Row(dbc.Col(html.Div(id="graphs"))),
        #dbc.Row(dbc.Col(html.Div(id="save"))), # testing
        dcc.Download(id="save-topics"),
        
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

    #layout = ', '.join(files) # testing

    return layout


@callback(
    #Output('save', 'children'),
    Output("save-topics", "data"),
    Input('submit-button-topic', 'n_clicks'),
    State('topics', 'value'),
    prevent_initial_call=True
)
def save_topics(n_clicks, files):
    if not isinstance(files, list):
        files = [files]
    layout = ', '.join(files) # testing
    #return f'{layout}'
    return dict(content=f'{layout}', filename="topics.txt")


# Run the app
if __name__ == '__main__':
    app.run(debug=True,
            #jupyter_width=arg_width, #"70%"
            #jupyter_height=arg_height, #"70%"
            )
