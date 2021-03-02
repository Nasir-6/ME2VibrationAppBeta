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
    html.H3('Forced Vibration', className=" mt-1, text-center"),
    html.H4("This module is under development. Do visit at a later date.", className=" mt-1, text-center")
])






