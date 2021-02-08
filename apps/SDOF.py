import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.express as px

from app import app

layout = html.Div([
    html.H3('Single Degree Of Freedom'),

    html.H6("Start Time"),
    dcc.Input(id='start-time', type='number', min=0, max=100, step=0.1, value=0),

    html.H6("End Time"),
    dcc.Input(id='end-time', type='number', min=2, max=100, step=0.1, value=10),

    html.H6("Number of points"),
    dcc.Input(id='npoints', type='number', min=2, max=10000, step=1, value=100),

    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

    dcc.Graph(id='sine_plot', figure={}),

])


@app.callback(Output('sine_plot', 'figure'),
              Input('submit-button-state', 'n_clicks'),
              State('start-time', 'value'),
              State('end-time', 'value'),
              State('npoints', 'value'))
def update_output(n_clicks, tstart, tend, npoints):
    print(tstart + tend)
    tarr = np.linspace(tstart, tend, npoints)
    y = np.sin(tarr)

    fig = px.line(x=tarr, y=y)



    return fig
