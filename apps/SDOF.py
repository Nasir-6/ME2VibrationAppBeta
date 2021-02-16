import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

import os


from app import app

header = html.H3('Single Degree Of Freedom', className=" mt-1, text-center")

line1_input = dbc.Row([

    dbc.Col(
        html.Img(src=app.get_asset_url('SDOFPic.png'),
                 className="img-fluid"),
        className="col-12 col-sm-5 col-md-3 col-lg-3"),
    dbc.Col([
        dbc.Row(html.H6("System 1")),
        dbc.Row([
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Mass, m", addon_type="prepend"),
                    dbc.Input(id="m", placeholder="kg", type="number", min=0, max=100, step=0.1),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Spring Constant, k", addon_type="prepend"),
                    dbc.Input(id="k", placeholder="N/m", type="number", min=0, max=100, step=0.1),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Coefficient, c", addon_type="prepend"),
                    dbc.Input(id="c", placeholder="Ns/m", type="number", min=0, max=100, step=0.01),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Initial Displacement, X0", addon_type="prepend"),
                    dbc.Input(id="x0", placeholder="kg", type="number", min=0, max=100, step=0.1),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Time Span, t", addon_type="prepend"),
                    dbc.Input(id="tend", placeholder="s", type="number", min=0, max=10000, step=0.1),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Points", addon_type="prepend"),
                    dbc.Input(id="n", placeholder="", type="number", min=10, max=10000, step=1, value=100),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-6"),
            dbc.Button("Submit", color="secondary", id='submit-button-state', size="sm")
        ])
    ]),
    # dbc.Col(,
    #         width=1, className="align-self-end")

], className="jumbotron")



layout = dbc.Container([
    header,
    line1_input,

    dcc.Graph(id='sine_plot', figure={}),

], fluid=True)


@app.callback(Output('sine_plot', 'figure'),
              Input('submit-button-state', 'n_clicks'),
              State('m', 'value'),
              State('k', 'value'),
              State('c', 'value'),
              State('x0', 'value'),
              State('tend', 'value'),
              State('n', 'value'))
def update_output(n_clicks, m, k, c, x0, tend, n):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    dampRatio = c / (2 * np.sqrt(k * m))
    tlim = 30
    if tend < tlim:  # 30 is limit (Change this once I have a value)
        t = np.linspace(0, tend, n)
    else:
        t = np.linspace(0, tlim, n)
    x = t.copy()

    if dampRatio == 0:
        x = x0 * np.cos(wn * t)
        solutionType = "Undamped Solution"
    elif 1 > dampRatio > 0:
        solutionType = "Under Damped Solution"
        wd = wn * np.sqrt(1 - dampRatio ** 2)
        A = x0
        B = dampRatio * A / wd
        x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
    elif dampRatio == 1:
        solutionType = "Critically Damped Solution"
        A = x0
        B = A * wn
        x = (A + B * t) * np.exp(-wn * t)
    elif dampRatio > 1:
        solutionType = "Over Damped Solution"
        A = x0 * (dampRatio + np.sqrt(dampRatio ** 2 - 1)) / (2 * np.sqrt(dampRatio ** 2 - 1))
        B = x0 - A
        x = A * np.exp((-dampRatio + np.sqrt(dampRatio ** 2 - 1)) * wn * t) + B * np.exp(
            (-dampRatio - np.sqrt(dampRatio ** 2 - 1)) * wn * t)
    else:
        solutionType = "Unaccounted for Solution"


# return x, t, solvable, solutionType
    fig = px.line(x=t, y=x,
                  labels=dict(x="Time (sec)", y="Displacement, x (m)"))



    return fig
