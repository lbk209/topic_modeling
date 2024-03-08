# Import packages
from dash import Dash, html, dcc, callback, Output, Input, State
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

# testing
#arg_options = options
#arg_path = figs_path


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY]
#external_stylesheets=[dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

title =  html.H4("Topic Distribution", className="text-primary")

dropdown = dcc.Dropdown(options=arg_options, value=arg_options[0]['value'],
                        id="topics", multi=True)

buttons = dbc.Stack(
    [
        dbc.Button("Save", id='save-button',
                   color="secondary", className="btn btn-primary btn-sm"),
        dcc.Upload(dbc.Button("Load", color="secondary", className="btn btn-primary btn-sm"),
                   id='load-topics')

    ],
    #className= "m-1",
    direction="horizontal",
    gap=2
)

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(title, lg=6)),
        dbc.Row([dbc.Col(dropdown, lg=6), dbc.Col(buttons)], align="center"),
        dbc.Row(dbc.Col(html.Div(id="graphs"))),
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
    """
    files: list of json file names to plot
    """
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
    Output("save-topics", "data"),
    Input('save-button', 'n_clicks'),
    State('topics', 'value'),
    prevent_initial_call=True
)
def save_topics(n_clicks, files):
    """
    save file names of choice in dropdown in a txt file
    """
    if not isinstance(files, list):
        files = [files]

    pattern = r'\d+(?=\.json)'
    tids = [int(re.findall(pattern, x)[0]) for x in files]
    tids = '_'.join([str(x) for x in tids])
    filename = os.path.commonprefix(files) # get common string
    filename = re.sub(r'\d+$', '', filename) # strip a number suffix
    filename = f'{filename}_{tids}.txt'

    content = ', '.join(files)
    return dict(content=f'{content}', filename=filename)


@callback(Output(component_id="topics", component_property="value"),
          Input('load-topics', 'contents'),
          State('load-topics', 'filename'),
          State('load-topics', 'last_modified'),
          prevent_initial_call=True)
def load_topics(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    files = decoded.decode('utf-8') # a string containing json file names
    files = re.findall(r'\w+\.json', files) # list of json files
    return files


# Run the app
if __name__ == '__main__':
    app.run(debug=True,
            #jupyter_width=arg_width, #"70%"
            #jupyter_height=arg_height, #"70%"
            )
