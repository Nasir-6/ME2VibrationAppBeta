import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

from validator import *
from app import app

header = html.H3('Single Degree Of Freedom', className=" mt-1, text-center")
about_Text = html.P("This SDOF solver takes in your parameters and then produces a time history plot of your system. "
                    "Try it out by changing the input parameters and pressing submit to view your solution at the bottom of the page.")

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
            "?", id="mass-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Mass Input"),
                dbc.PopoverBody([], id="mass_validation_message"),
            ],
            id="mass_popover",
            is_open=False,
            target="mass-popover-target",
        ),
    ],
)

springConst_popover = html.Div(
    [
        dbc.Button(
            "?", id="springConst-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Sprint Constant Input"),
                dbc.PopoverBody([], id="springConst_validation_message"),
            ],
            id="springConst_popover",
            is_open=False,
            target="springConst-popover-target",
        ),
    ],
)

dampRatio_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampRatio-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Ratio Input"),
                dbc.PopoverBody([], id="dampRatio_validation_message"),
            ],
            id="dampRatio_popover",
            is_open=False,
            target="dampRatio-popover-target",
        ),
    ],
)

dampCoeff_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampCoeff-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Coefficient Input"),
                dbc.PopoverBody([], id="dampCoeff_validation_message"),
            ],
            id="dampCoeff_popover",
            is_open=False,
            target="dampCoeff-popover-target",
        ),
    ],
)

initDisp_popover = html.Div(
    [
        dbc.Button(
            "?", id="initDisp-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Initial Displacement Input"),
                dbc.PopoverBody([], id="initDisp_validation_message"),
            ],
            id="initDisp_popover",
            is_open=False,
            target="initDisp-popover-target",
        ),
    ],
)


tSpan_popover = html.Div(
    [
        dbc.Button(
            "?", id="tSpan-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Time Span Input"),
                dbc.PopoverBody([], id="tSpan_validation_message"),
            ],
            id="tSpan_popover",
            is_open=False,
            target="tSpan-popover-target",
        ),
    ],
)


numPts_popover = html.Div(
    [
        dbc.Button(
            "?", id="numPts-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Number of Points Input"),
                dbc.PopoverBody([], id="numPts_validation_message"),
            ],
            id="numPts_popover",
            is_open=False,
            target="numPts-popover-target",
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
                        id="m",
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
                    dbc.Input(id="k", placeholder="N/m", debounce=True, type="number",
                              value=1000, min=0.001, step=0.001),
                    dbc.InputGroupAddon(
                        springConst_popover,
                        addon_type="append"
                    ),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),

            dbc.Col(damp_switch, width=12),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Ratio", addon_type="prepend"),
                    dbc.Input(id="dampRatio", placeholder="", debounce=True, type="number", value=0.1, min=0, max=2,
                              step=0.001),
                    dbc.InputGroupAddon(
                        dampRatio_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Coefficient, c (Ns/m)", addon_type="prepend"),
                    dbc.Input(id="c", placeholder="Ns/m", debounce=True, type="number", value=1, min=0, step=0.001),
                    dbc.InputGroupAddon(
                        dampCoeff_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Initial Displacement, X0 (m)", addon_type="prepend"),
                    dbc.Input(id="x0", placeholder="m", debounce=True, type="number", value=0.1, min=-10, max=10,
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
                    dbc.Input(id="tend", placeholder="s", debounce=True,  type="number", value=2, min=0.01, max=360, step=0.01),
                    dbc.InputGroupAddon(
                        tSpan_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Points", addon_type="prepend"),
                    dbc.Input(id="n", placeholder="", debounce=True,  type="number", min=10, step=1, value=1000),
                    dbc.InputGroupAddon(
                        numPts_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.P(id="aliasing_Warning", className="text-danger"),
                width=12
            ),
            dbc.Button("Submit", color="secondary", id='submit-button-state', size="sm")
        ]),
        dbc.Row(html.P(id="solution_string")),


    ]),

], className="jumbotron")



layout = dbc.Container([
    header,
    about_Text,
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
def mass_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("springConst_validation_message", "children"),
    Output("springConst-popover-target", "n_clicks"),
    Input("k", "value")
)
def springConst_input_validator(springConst_input):
    err_string, is_invalid = validate_input("spring constant", springConst_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle springConst popover with button (or validator callback above!!)
@app.callback(
    Output("springConst_popover", "is_open"),
    [Input("springConst-popover-target", "n_clicks")],
    [State("springConst_popover", "is_open")],
)
def springConst_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("dampRatio_validation_message", "children"),
    Output("dampRatio-popover-target", "n_clicks"),
    Input("dampRatio", "value")
)
def dampRatio_input_validator(dampRatio_input):
    err_string, is_invalid = validate_input("damping ratio", dampRatio_input, step=0.001, min=0, max=2)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle dampRatio popover with button (or validator callback above!!)
@app.callback(
    Output("dampRatio_popover", "is_open"),
    [Input("dampRatio-popover-target", "n_clicks")],
    [State("dampRatio_popover", "is_open")],
)
def dampRatio_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("dampCoeff_validation_message", "children"),
    Output("dampCoeff-popover-target", "n_clicks"),
    Input("c", "value")
)
def dampCoeff_input_validator(dampCoeff_input):
    err_string, is_invalid = validate_input("damping coefficient", dampCoeff_input, step=0.001, min=0)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle dampCoeff popover with button (or validator callback above!!)
@app.callback(
    Output("dampCoeff_popover", "is_open"),
    [Input("dampCoeff-popover-target", "n_clicks")],
    [State("dampCoeff_popover", "is_open")],
)
def dampCoeff_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("initDisp_validation_message", "children"),
    Output("initDisp-popover-target", "n_clicks"),
    Input("x0", "value")
)
def initDisp_input_validator(initDisp_input):
    err_string, is_invalid = validate_input("initial displacement", initDisp_input, step=0.1, min=-10, max=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle initDisp popover with button (or validator callback above!!)
@app.callback(
    Output("initDisp_popover", "is_open"),
    [Input("initDisp-popover-target", "n_clicks")],
    [State("initDisp_popover", "is_open")],
)
def initDisp_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("tSpan_validation_message", "children"),
    Output("tSpan-popover-target", "n_clicks"),
    Input("tend", "value")
)
def tSpan_input_validator(tSpan_input):
    err_string, is_invalid = validate_input("time span", tSpan_input, step=0.01, min=0.01, max=360)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle tSpan popover with button (or validator callback above!!)
@app.callback(
    Output("tSpan_popover", "is_open"),
    [Input("tSpan-popover-target", "n_clicks")],
    [State("tSpan_popover", "is_open")],
)
def tSpan_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("numPts_validation_message", "children"),
    Output("numPts-popover-target", "n_clicks"),
    Input("n", "value")
)
def numPts_input_validator(numPts_input):
    err_string, is_invalid = validate_input("number of points", numPts_input, step=1, min=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle numPts popover with button (or validator callback above!!)
@app.callback(
    Output("numPts_popover", "is_open"),
    [Input("numPts-popover-target", "n_clicks")],
    [State("numPts_popover", "is_open")],
)
def numPts_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


# ======= ALIASING WARNING ==================
@app.callback(
    Output("aliasing_Warning", "children"),
    [Input("m", "value"),
     Input("k", "value"),
     Input("tend", "value"),
     Input("n", "value"),
     ]
)
def aliasing_check(m, k, tSpan, nPts):
    aliasing_warning = validate_aliasing(m, k, tSpan, nPts)
    return aliasing_warning





# ======== Damping Ratio & Coefficient Updates =============

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






# ============ Plotting Graph ========================

# This Function plots the graph
@app.callback(Output('SDOF_plot', 'figure'),
              Output('solution_string', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('m', 'value'),
              State('k', 'value'),
              State('dampRatio', 'value'),
              State('c', 'value'),
              State('x0', 'value'),
              State('tend', 'value'),
              State('n', 'value'))
def update_output(n_clicks, m, k, dampRatio, dampCoeff, x0, tend, n):

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