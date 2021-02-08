import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.H3('Forced Vibration'),

    html.Div(id='SDOF-display-value'),
    dcc.Link('Go to SDOF', href='/apps/SDOF')
])


# @app.callback(
#     Output('SDOF-display-value', 'children'),
#     Input('app-1-dropdown', 'value'))
# def display_value(value):
#     return 'You have selected "{}"'.format(value)