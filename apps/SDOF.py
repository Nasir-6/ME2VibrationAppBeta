import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

import os

from validator import *
from app import app

header = html.H3('Single Degree Of Freedom', className=" mt-1, text-center")

damp_switch = dbc.FormGroup(
    [
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

# ================== ALL POPOVER COMPONENTS
mass_popover = html.Div(
    [
        dbc.Button(
            "Mass, m (kg)", id="mass-popover-target", color="info"
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Mass Input"),
                dbc.PopoverBody(["Your mass input is valid.", html.Br(), "It is greater than 0 and has a minimum increment greater than 0.001"], id="mass_validation_message"),
            ],
            id="mass_popover",
            is_open=False,
            target="mass-popover-target",
        ),
    ]
)


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
                    # dbc.InputGroupAddon("Mass, m (kg)", addon_type="prepend"),
                    dbc.InputGroupAddon(
                        mass_popover,
                        addon_type="prepend",
                    ),
                    dbc.Input(id="m", placeholder="kg", debounce=True, type="number", value=1, min=0.001, step=0.001),
                ],
            ), className="mb-1 col-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Spring Constant, k (N/m)", addon_type="prepend"),
                    dbc.Input(id="k", placeholder="N/m", debounce=True,  type="number", value=1000, min=0.001, step=0.001),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),

            dbc.Col(damp_switch, width=12),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Ratio", addon_type="prepend"),
                    dbc.Input(id="dampRatio", placeholder="", debounce=True,  type="number", value=0.1, min=0, max=2, step=0.001),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Coefficient, c (Ns/m)", addon_type="prepend"),
                    dbc.Input(id="c", placeholder="Ns/m", debounce=True, type="number", value=1, min=0.001, step=0.001),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Initial Displacement, X0 (m)", addon_type="prepend"),
                    dbc.Input(id="x0", placeholder="m", debounce=True,  type="number", value=0.1, min=0, max=10, step=0.01),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Time Span, t (s)", addon_type="prepend"),
                    dbc.Input(id="tend", placeholder="s", debounce=True,  type="number", value=2, min=0.01, max=360, step=0.01),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Points", addon_type="prepend"),
                    dbc.Input(id="n", placeholder="", debounce=True,  type="number", min=10, max=10000, step=1, value=1000),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Button("Submit", color="secondary", id='submit-button-state', size="sm")
        ]),
        dbc.Row(html.P(id="solution_string")),
        dbc.Row(html.P(id="validation_string"))

    ]),

], className="jumbotron")



layout = dbc.Container([
    header,
    line1_input,
    dcc.Graph(id='SDOF_plot', figure={}),

], fluid=True)




# ALL APP CALLBACKS

# INPUT VALIDATORS
@app.callback(
    Output("mass_validation_message", "children"),
    Output("mass-popover-target", "n_clicks"),
    Input("m", "value")
)
def mass_input_validator(mass_input):
    err_string, is_invalid = validate_input("mass", mass_input, step=0.001, min=0)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle mass popover with button (or validator callback above!!)
@app.callback(
    Output("mass_popover", "is_open"),
    [Input("mass-popover-target", "n_clicks")],
    [State("mass_popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open





# This function disables the damping ratio or damping coefficient input using the toggle
@app.callback(
    Output("dampRatio", "disabled"),
    Output("c", "disabled"),
    Input("damping-switch", "value")
)
def damping_toggle(switch):
    switch_state = len(switch)
    return switch_state, not switch_state



# This function updates damping coefficient c when it is disabled and other values are inputted
@app.callback(
    Output(component_id='c', component_property='value'),
    Input(component_id='c', component_property='disabled'),
    Input(component_id='c', component_property='value'),
    Input(component_id='dampRatio', component_property='value'),
    Input(component_id='k', component_property='value'),
    Input(component_id='m', component_property='value')
)
def update_c(c_disabled, c, dampRatio, k, m):
    if c_disabled and m!=None and k!=None and dampRatio!=None:
        c = np.round((dampRatio * 2 * np.sqrt(k * m)),3)
    return c


# This function updates damping ratio when it is disabled and other values are inputted
@app.callback(
    Output(component_id='dampRatio', component_property='value'),
    Input(component_id='dampRatio', component_property='disabled'),
    Input(component_id='dampRatio', component_property='value'),
    Input(component_id='c', component_property='value'),
    State(component_id='k', component_property='value'),
    State(component_id='m', component_property='value')
)
def update_damping_ratio(dampRatio_disabled, dampRatio, c, k, m):
    if dampRatio_disabled and m!=None and k!=None and c!=None:
        dampRatio = np.round((c / (2 * np.sqrt(k * m))),3)
    return dampRatio




# This Function plots the graph
@app.callback(Output('SDOF_plot', 'figure'),
              Output('solution_string', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('m', 'value'),
              State('k', 'value'),
              State('dampRatio', 'value'),
              State('x0', 'value'),
              State('tend', 'value'),
              State('n', 'value'))
def update_output(n_clicks, m, k, dampRatio, x0, tend, n):

    x, t, solutionType = SDOF_solver(m, k, dampRatio,  x0, tend, n)
    fig = px.line(x=t, y=x, labels=dict(x="Time (sec)",
                                        y="Displacement, x (m)"))
    solutionTypeString = "This is " + solutionType
    return fig, solutionTypeString

def SDOF_solver(m, k, dampRatio, x0, tend, n):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    tlim = 1000
    if tend < tlim:  # 30 is limit (Change this once I have a value)
        t = np.linspace(0, tend, n)
    else:
        t = np.linspace(0, tlim, n)
    x = t.copy()

    if dampRatio == 0:
        x = x0 * np.cos(wn * t)
        solutionType = "an Undamped Solution"
    elif 1 > dampRatio > 0:
        solutionType = "an Under Damped Solution"
        wd = wn * np.sqrt(1 - dampRatio ** 2)
        A = x0
        B = dampRatio * A / wd
        x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
    elif dampRatio == 1:
        solutionType = "a Critically Damped Solution"
        A = x0
        B = A * wn
        x = (A + B * t) * np.exp(-wn * t)
    elif dampRatio > 1:
        solutionType = "an Over Damped Solution"
        A = x0 * (dampRatio + np.sqrt(dampRatio ** 2 - 1)) / (2 * np.sqrt(dampRatio ** 2 - 1))
        B = x0 - A
        x = A * np.exp((-dampRatio + np.sqrt(dampRatio ** 2 - 1)) * wn * t) + B * np.exp(
            (-dampRatio - np.sqrt(dampRatio ** 2 - 1)) * wn * t)
    else:
        solutionType = " an unaccounted for Solution"

    return x, t,  solutionType