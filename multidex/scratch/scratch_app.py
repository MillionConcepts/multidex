import os

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_html_components as html


def print_clicks(n_clicks):
    return n_clicks


app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.Button(
            id={'type': 'arrow', 'index': 'apple'},
            className='arrow',
        ),
        html.Div(id={'type': 'dummy-output', 'index': 'apple'})
    ]
)

app.callback(
    Output({"type": "dummy-output", "index": MATCH}, "children"),
    Input({"type": "arrow", "index": MATCH}, "n_clicks"),
)(print_clicks)
app.run_server(
    debug=True, use_reloader=False, dev_tools_silence_routes_logging=True
)
