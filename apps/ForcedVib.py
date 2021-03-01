import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from app import app

damp_switch = dbc.FormGroup(
    [
        dbc.Label("Toggle a bunch"),
        dbc.Checklist(
            options=[
                {"label": "Damping Ratio Or Coefficient", "value": 1}
            ],
            value=[],
            id="damping-switch",
            switch=True,
        ),
    ]
)

layout = dbc.Container([
    html.H3('Forced Vibration'),
    html.Div(id='SDOF-display-value'),
    dcc.Link('Go to SDOF', href='/apps/SDOF'),
    damp_switch,
    dbc.Col(dbc.InputGroup(
        [
            dbc.InputGroupAddon("Damping Ratio", addon_type="prepend"),
            dbc.Input(id="dampRatio", placeholder="", type="number", disabled=False, min=0, max=2, step=0.01, value=0.1),
        ],
    ), className="mb-1 col-12 col-sm-12 col-md-6"),
    dbc.Col(dbc.InputGroup(
        [
            dbc.InputGroupAddon("Damping Coefficient, c (Ns/m)", addon_type="prepend"),
            dbc.Input(id="c", placeholder="Ns/m", type="number", disabled=True, min=0.01, step=0.01, value=1.5),
        ],
    ), className="mb-1 col-12 col-sm-12 col-md-6"),
    html.H1(id="testOutput")
])






