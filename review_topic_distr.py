# Import packages
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import plotly.express as px
import plotly.io as pio
import dash_bootstrap_components as dbc
import sys
import io, re, os
import base64
import argparse

#from bertopic_custom_util import read_csv


# Parsing command-line options and arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument("prfx", help="prefix of fig files")
parser.add_argument("-d", "--directory", help="path to fig files", default=".")
parser.add_argument("-p", "--pattern", help="regular expression for fig files", default=r'\d+(?=\.json)')
parser.add_argument("-jh", "--height", help="layout height", default=650, type=int)
parser.add_argument("-jw", "--width", help="layout width", default="100%")
parser.add_argument("-tp", "--topic", help="file of df_topic_info")
args = parser.parse_args()

fig_prfx = args.prfx
fig_path = args.directory
pattern = args.pattern
jupyter_width = args.width
jupyter_height = args.height
df_topic_info = args.topic

debug = False
"""

# testing
fig_path = 'cabs2'
fig_prfx = 'sdistr'
pattern = r'\d+(?=\.json)'

jupyter_width = '80%'
jupyter_height = 800
df_topic_info = 'df_topic_info.csv'
debug = True


def create_card(title, content):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle"),
            html.P(content, className="card-text"),
        ])
    )


# get json file list
fig_files = [x for x in os.listdir(fig_path) if x.startswith(fig_prfx)]
fig_files = sorted(fig_files)

n = len(fig_files)
if n == 0:
    print('ERROR!: No fig to read')
else:
    #print(f'{n} figs ready to load')
    pass

# create topic options (dicts of label and value) for dropdown menu
tids = [int(re.findall(pattern, x)[0]) for x in fig_files]
options = [{'label':f'Topic {t}', 'value': f"{t}, {f}"} for t, f in zip(tids, fig_files)]

# import df_topic_info
if df_topic_info is not None:
    # None if no file exists
    df_topic_info = read_csv(df_topic_info, path_data=fig_path)

    if df_topic_info is not None:
        try:
            cols = df_topic_info.columns.to_list()
            aspects = cols[cols.index('Representation'):]
        except ValueError as e:
            print(f'ERROR: {e}')
            df_topic_info = None

    topic_info = df_topic_info
    if topic_info is not None:
        topic_info = df_topic_info[aspects].applymap(eval).to_dict(orient='records')

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY]
#external_stylesheets=[dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

title =  html.H4("Topic Distribution", className="text-primary")

dropdown = dcc.Dropdown(options=options, value=options[0]['value'],
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
        #dbc.Row(dbc.Col(html.Div(id="graphs"))),
        html.Div(id="graphs"),
        #[dbc.Row(x) for x in html.Div(id="graphs")], # error
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
def plot_topic_distr(value):
    """
    value: list of str combining topic id and json file names to plot
    """
    if not isinstance(value, list):
        value = [value]

    layout = []
    for v in value:
        tid, f = [x.strip() for x in v.split(',')]
        tid = int(tid)
        f = f'{fig_path}/{f}'

        fig = pio.read_json(f)
        g = dcc.Graph(figure=fig, className="border")
        # testing
        #g = html.Div(f)

        #info = html.Div(tid)
        #infos = topic_info[tid]
        #infos = [html.Div(f'{k}:\n{", ".join(v)}') for k, v in infos.items()]

        if topic_info is None:
            row = dbc.Row(dbc.Col(g), className="mt-2")
        else:
            infos = []
            for title, content in topic_info[tid].items():

                if title == 'Representative_Docs':
                    content = html.Div([html.P(c) for c in content],
                                        style={
                                            "fontSize": 14,
                                            "height": "120px",
                                            'overflow':'auto'
                                            })
                else:
                    content = html.Div(", ".join(content))

                #infos.append(html.Div([html.H6(title), content]))
                #infos.append(html.Div([html.Div(title), content]))
                infos.append(html.P([html.Div(title), content]))
                #infos.append(create_card(title, content))

            infos = html.Div(infos, style={"height": fig.layout.height})
            row = dbc.Row([dbc.Col(g), dbc.Col(infos)], className="mt-2")

        layout.append(row)

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


@callback([Output(component_id="topics", component_property="value"),
          Output('load-topics', 'contents')],
          Input('load-topics', 'contents'),
          State('load-topics', 'filename'),
          State('load-topics', 'last_modified'),
          prevent_initial_call=True)
def load_topics(contents, filename, date):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        files = decoded.decode('utf-8') # a string containing json file names
        files = re.findall(r'\w+\.json', files) # list of json files
        if len(files) == 0:
            files = no_update
    except Exception as e:
        print(e)
        files = no_update

    return files, None


# Run the app
if __name__ == '__main__':
    app.run(debug=debug,
            jupyter_width=jupyter_width,
            jupyter_height=jupyter_height,
            )
