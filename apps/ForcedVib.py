import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

from validator import *
from app import app

header = html.H3('Forced Vibration', className=" mt-2, text-center")
about_Text = html.P(["This Forced Vibrations solver takes in your parameters and then produces an FRF response. You can then choose a frequency to view the time history plot at that specific frequency."
                    "Try it out by changing the input parameters and pressing submit to view your solution at the bottom of the page.)"])


# ================== ALL POPOVER COMPONENTS
mass_popover = html.Div(
    [
        dbc.Button(
            "?", id="mass-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Mass Input"),
                dbc.PopoverBody([], id="mass_validation_message-FV"),
            ],
            id="mass_popover-FV",
            is_open=False,
            target="mass-popover-target-FV",
        ),
    ],
)

springConst_popover = html.Div(
    [
        dbc.Button(
            "?", id="springConst-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Sprint Constant Input"),
                dbc.PopoverBody([], id="springConst_validation_message-FV"),
            ],
            id="springConst_popover-FV",
            is_open=False,
            target="springConst-popover-target-FV",
        ),
    ],
)

dampRatio_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampRatio-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Ratio Input"),
                dbc.PopoverBody([], id="dampRatio_validation_message-FV"),
            ],
            id="dampRatio_popover-FV",
            is_open=False,
            target="dampRatio-popover-target-FV",
        ),
    ],
)

dampCoeff_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampCoeff-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Coefficient Input-FV"),
                dbc.PopoverBody([], id="dampCoeff_validation_message-FV"),
            ],
            id="dampCoeff_popover-FV",
            is_open=False,
            target="dampCoeff-popover-target-FV",
        ),
    ],
)

initDisp_popover = html.Div(
    [
        dbc.Button(
            "?", id="initDisp-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Initial Displacement Input-FV"),
                dbc.PopoverBody([], id="initDisp_validation_message-FV"),
            ],
            id="initDisp_popover-FV",
            is_open=False,
            target="initDisp-popover-target-FV",
        ),
    ],
)


tSpan_popover = html.Div(
    [
        dbc.Button(
            "?", id="tSpan-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Time Span Input"),
                dbc.PopoverBody([], id="tSpan_validation_message-FV"),
            ],
            id="tSpan_popover-FV",
            is_open=False,
            target="tSpan-popover-target-FV",
        ),
    ],
)


numPts_popover = html.Div(
    [
        dbc.Button(
            "?", id="numPts-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Number of Points Input-FV"),
                dbc.PopoverBody([], id="numPts_validation_message-FV"),
            ],
            id="numPts_popover-FV",
            is_open=False,
            target="numPts-popover-target-FV",
        ),
    ],
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
                    dbc.InputGroupAddon(
                        "Mass, m (kg)",
                        addon_type="prepend"
                    ),
                    dbc.Input(
                        id="m-FV",
                        placeholder="kg",
                        debounce=True, type="number",
                        value=1, min=0.001, step=0.001),
                    dbc.InputGroupAddon(
                        mass_popover,
                        addon_type="append"
                    ),
                ],
            ), className="mb-1 col-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Spring Constant, k (N/m)", addon_type="prepend"),
                    dbc.Input(id="k-FV", placeholder="N/m", debounce=True, type="number",
                              value=1000, min=0.001, step=0.001),
                    dbc.InputGroupAddon(
                        springConst_popover,
                        addon_type="append"
                    ),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),

            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Ratio", addon_type="prepend"),
                    dbc.Input(id="dampRatio-FV", placeholder="", debounce=True, type="number", value=0.1, min=0, max=2,
                              step=0.001),
                    dbc.InputGroupAddon(
                        dampRatio_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Initial Displacement, X0 (m)", addon_type="prepend"),
                    dbc.Input(id="x0-FV", placeholder="m", debounce=True, type="number", value=0.1, min=-10, max=10,
                              step=0.01),
                    dbc.InputGroupAddon(
                        initDisp_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.H6("Computational Parameters"),
                className="mb-1 mt-1 col-12 col-sm-12 col-md-12 col-lg-12"
            ),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Time Span, t (s)", addon_type="prepend"),
                    dbc.Input(id="tend-FV", placeholder="s", debounce=True,  type="number", value=2, min=0.01, max=360, step=0.01),
                    dbc.InputGroupAddon(
                        tSpan_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Points", addon_type="prepend"),
                    dbc.Input(id="n-FV", placeholder="", debounce=True,  type="number", min=10, step=1, value=1000),
                    dbc.InputGroupAddon(
                        numPts_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.P(id="aliasing_Warning-FV", className="text-danger"),
                width=12
            ),
            dbc.Button("Submit", color="secondary", id='submit-button-state-FV', size="sm")
        ]),
        dbc.Row(html.P(id="solution_string-FV")),


    ]),

], className="jumbotron")



# layout = dbc.Container([
#     header,
#     about_Text,
#     line1_input,
#     html.H3("Frequency Response Function (FRF) of your solution", className=" mt-1 mb-1 text-center"),
#     dcc.Graph(id='FRF_plot', figure={}),
#
# ], fluid=True)

layout = dbc.Container([
    html.H3('Forced Vibration', className=" mt-2, text-center"),
    html.H6("This module is currently under development. Do come back at a later date.",className=" mt-2, text-center" ),
], fluid=True)



# ALL APP CALLBACKS

# INPUT VALIDATORS
@app.callback(
    Output("mass_validation_message-FV", "children"),
    Output("mass-popover-target-FV", "n_clicks"),
    Input("m-FV", "value")
)
def mass_input_validator(mass_input):
    err_string, is_invalid = validate_input("mass-FV", mass_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle mass popover with button (or validator callback above!!)
@app.callback(
    Output("mass_popover-FV", "is_open"),
    [Input("mass-popover-target-FV", "n_clicks")],
    [State("mass_popover-FV", "is_open")],
)
def mass_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("springConst_validation_message-FV", "children"),
    Output("springConst-popover-target-FV", "n_clicks"),
    Input("k", "value")
)
def springConst_input_validator(springConst_input):
    err_string, is_invalid = validate_input("spring constant-FV", springConst_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle springConst popover with button (or validator callback above!!)
@app.callback(
    Output("springConst_popover-FV", "is_open"),
    [Input("springConst-popover-target-FV", "n_clicks")],
    [State("springConst_popover-FV", "is_open")],
)
def springConst_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("dampRatio_validation_message-FV", "children"),
    Output("dampRatio-popover-target-FV", "n_clicks"),
    Input("dampRatio-FV", "value")
)
def dampRatio_input_validator(dampRatio_input):
    err_string, is_invalid = validate_input("damping ratio-FV", dampRatio_input, step=0.001, min=0, max=2)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle dampRatio popover with button (or validator callback above!!)
@app.callback(
    Output("dampRatio_popover-FV", "is_open"),
    [Input("dampRatio-popover-target-FV", "n_clicks")],
    [State("dampRatio_popover-FV", "is_open")],
)
def dampRatio_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open




@app.callback(
    Output("initDisp_validation_message-FV", "children"),
    Output("initDisp-popover-target-FV", "n_clicks"),
    Input("x0-FV", "value")
)
def initDisp_input_validator(initDisp_input):
    err_string, is_invalid = validate_input("initial displacement", initDisp_input, step=0.1, min=-10, max=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle initDisp popover with button (or validator callback above!!)
@app.callback(
    Output("initDisp_popover-FV", "is_open"),
    [Input("initDisp-popover-target-FV", "n_clicks")],
    [State("initDisp_popover-FV", "is_open")],
)
def initDisp_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("tSpan_validation_message-FV", "children"),
    Output("tSpan-popover-target-FV", "n_clicks"),
    Input("tend-FV", "value")
)
def tSpan_input_validator(tSpan_input):
    err_string, is_invalid = validate_input("time span", tSpan_input, step=0.01, min=0.01, max=360)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle tSpan popover with button (or validator callback above!!)
@app.callback(
    Output("tSpan_popover-FV", "is_open"),
    [Input("tSpan-popover-target-FV", "n_clicks")],
    [State("tSpan_popover-FV", "is_open")],
)
def tSpan_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("numPts_validation_message-FV", "children"),
    Output("numPts-popover-target-FV", "n_clicks"),
    Input("n-FV", "value")
)
def numPts_input_validator(numPts_input):
    err_string, is_invalid = validate_input("number of points", numPts_input, step=1, min=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle numPts popover with button (or validator callback above!!)
@app.callback(
    Output("numPts_popover-FV", "is_open"),
    [Input("numPts-popover-target-FV", "n_clicks")],
    [State("numPts_popover-FV", "is_open")],
)
def numPts_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


# ======= ALIASING WARNING ==================
@app.callback(
    Output("aliasing_Warning-FV", "children"),
    [Input("m-FV", "value"),
     Input("k-FV", "value"),
     Input("tend-FV", "value"),
     Input("n-FV", "value"),
     ]
)
def aliasing_check(m, k, tSpan, nPts):
    aliasing_warning = validate_aliasing(m, k, tSpan, nPts)
    return aliasing_warning









# ============ Plotting Graph ========================

# This Function plots the FRF graph
@app.callback(Output('FRF_plot', 'figure'),
              Output('solution_string-FV', 'children'),
              Input('submit-button-state-FV', 'n_clicks'),
              State('m-FV', 'value'),
              State('k-FV', 'value'),
              State('dampRatio-FV', 'value'),
              State('x0-FV', 'value'),
              State('tend-FV', 'value'),
              State('n-FV', 'value'))
def update_output(n_clicks, m, k, dampRatio, x0, tend, n):

    dampCoeff=1 # THIS IS TO USE VALIDATION (NEED TO SORT THIS OUT!!!!!)
    # First validate inputs
    is_invalid = validate_all_inputs(m,k,dampRatio, dampCoeff, x0, tend, n)
    if(is_invalid):
        solutionTypeString = "Graph was not Updated. Please check your inputs before Submitting"
        return dash.no_update, solutionTypeString

    x, t, solutionType = SDOF_solver(m, k, dampRatio,  x0, tend, n)
    fig = px.line(x=t, y=x, labels=dict(x="Time (sec)",
                                        y="Displacement, x (m)"))
    solutionTypeString = "This is " + solutionType + ". Please scroll down to see your solution."
    return fig, solutionTypeString

def FRFSolver(m=10, k=10, dampRatios=np.array([0.25,0.15,0.5]), wantNormalised = False):

    solvable = True
    # INSERT VALIDATOR HERE
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    w = np.linspace(0, 10, 10000)
    r = w / wn

    amp = np.zeros((len(dampRatios), len(w)))
    phase = np.zeros((len(dampRatios), len(w)))
    if wantNormalised:
        row = 0
        for dampRat in dampRatios:
            print(dampRat)
            amp[row, :] = 1 / np.sqrt((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2)
            phase[row, :] = np.arctan(-2 * dampRat * r / (1 - r ** 2))
            phase[phase > 0] = phase[phase > 0] - np.pi
            row = row + 1
    else:
        row = 0
        for dampRat in dampRatios:
            c = dampRat * 2 * np.sqrt(k * m)
            print(dampRat)
            amp[row, :] = 1 / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
            phase[row, :] = np.arctan(-c * w / (k - m * w ** 2))
            phase[phase > 0] = phase[phase > 0] - np.pi
            row = row + 1


    return amp, phase, r, w, solvable