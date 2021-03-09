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


damp_switch = dbc.FormGroup(
    [
        dbc.Checklist(
            options=[
                {"label": "Use Damping Coefficient", "value": 1}
            ],
            value=[],
            id="damping-switch-FV",
            switch=True,
        ),
    ]
)


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
                dbc.PopoverHeader("Damping Coefficient Input"),
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
                dbc.PopoverHeader("Initial Displacement Input"),
                dbc.PopoverBody([], id="initDisp_validation_message-FV"),
            ],
            id="initDisp_popover-FV",
            is_open=False,
            target="initDisp-popover-target-FV",
        ),
    ],
)


forceAmp_popover = html.Div(
    [
        dbc.Button(
            "?", id="forceAmp-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Forcing Amplitude"),
                dbc.PopoverBody([], id="forceAmp_validation_message-FV"),
            ],
            id="forceAmp_popover-FV",
            is_open=False,
            target="forceAmp-popover-target-FV",
        ),
    ],
)






wlim_popover = html.Div(
    [
        dbc.Button(
            "?", id="wlim-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("ω axis limit Input"),
                dbc.PopoverBody([], id="wlim_validation_message-FV"),
            ],
            id="wlim_popover-FV",
            is_open=False,
            target="wlim-popover-target-FV",
        ),
    ],
)









line1_input = dbc.Row([

    dbc.Col(
        html.Img(src=app.get_asset_url('ForcedVib.png'),
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

            dbc.Col(damp_switch, width=12),
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
                    dbc.InputGroupAddon("Damping Coefficient, c (Ns/m)", addon_type="prepend"),
                    dbc.Input(id="c-FV", placeholder="Ns/m", debounce=True, type="number", value=1, min=0, step=0.001),
                    dbc.InputGroupAddon(
                        dampCoeff_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.H6("Initial Conditions"),
                className="mb-1 mt-1 col-12 col-sm-12 col-md-12 col-lg-12"
            ),
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
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Forcing Amplitude, F0 (N)", addon_type="prepend"),
                    dbc.Input(id="F0-FV", placeholder="N", debounce=True, type="number", value=0.1, min=-10, max=10,
                              step=0.01),
                    dbc.InputGroupAddon(
                        forceAmp_popover,
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
                    dbc.InputGroupAddon("ω axis limit , ω (rad/s)", addon_type="prepend"),
                    dbc.Input(id="wlim-FV", placeholder="s", debounce=True,  type="number", value=100, min=0.1, max=100, step=0.1),
                    dbc.InputGroupAddon(
                        wlim_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Button("Submit", color="secondary", id='submit-button-state-FV', size="sm")
        ]),
        dbc.Row(html.P(id="input_warning_string-FV", className="text-danger")),
        dbc.Row(html.P(id="system_params-FV")),

    ]),

], className="jumbotron")



layout = dbc.Container([
    header,
    about_Text,
    line1_input,
    html.H3("Frequency Response Function (FRF) of your solution", className=" mt-1 mb-1 text-center"),
    dcc.Graph(id='FRFAmp_plot', figure={}),
    dcc.Graph(id='FRFPhase_plot', figure={}),
    html.H3("Force and Displacement time history", className=" mt-1 mb-1 text-center"),
    html.H4("Please choose a excitation frequency using the slider below", className=" mt-1 mb-1 text-center"),
    dcc.Slider(id="w-slider",
               min=0,
               max=100,
               step=0.01,
               value=50,
               marks={
                   0: '0 rad/s',
                   100: '100 rad/s'
               },
               updatemode='drag'
               ),
    html.P(id="w-slider-output-FV", className=" mt-1 mb-1 text-center"),
    dcc.Graph(id='timeHistory-plot-FV', figure={}),

], fluid=True)

# # THIS IS A PLACEHOLDER FOR BETA RELEASE - COMING SOON PAGE!!! ============================================================
# layout = dbc.Container([
#     html.H3('Forced Vibration', className=" mt-2, text-center"),
#     html.H6("This module is currently under development. Do come back at a later date.",className=" mt-2, text-center" ),
# ], fluid=True)



# ALL APP CALLBACKS

# INPUT VALIDATORS
@app.callback(
    Output("mass_validation_message-FV", "children"),
    Output("mass-popover-target-FV", "n_clicks"),
    Input("m-FV", "value")
)
def mass_input_validator(mass_input):
    err_string, is_invalid = validate_input("mass", mass_input, step=0.001, min=0.001)
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
    Input("k-FV", "value")
)
def springConst_input_validator(springConst_input):
    err_string, is_invalid = validate_input("spring constant", springConst_input, step=0.001, min=0.001)
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
    err_string, is_invalid = validate_input("damping ratio", dampRatio_input, step=0.001, min=0, max=2)
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


######### DAMPING COEFFFICIENT VALIDATOR #########################
@app.callback(
    Output("dampCoeff_validation_message-FV", "children"),
    Output("dampCoeff-popover-target-FV", "n_clicks"),
    Input("c-FV", "value")
)
def dampCoeff_input_validator(dampCoeff_input):
    err_string, is_invalid = validate_input("damping coefficient", dampCoeff_input, step=0.001, min=0)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle dampCoeff popover with button (or validator callback above!!)
@app.callback(
    Output("dampCoeff_popover-FV", "is_open"),
    [Input("dampCoeff-popover-target-FV", "n_clicks")],
    [State("dampCoeff_popover-FV", "is_open")],
)
def dampCoeff_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



######### Initial Dislacement VALIDATOR #########################

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



######### Forcing Amplitude VALIDATOR #########################

@app.callback(
    Output("forceAmp_validation_message-FV", "children"),
    Output("forceAmp-popover-target-FV", "n_clicks"),
    Input("F0-FV", "value")
)
def forceAmp_input_validator(forceAmp_input):
    err_string, is_invalid = validate_input("forcing amplitude", forceAmp_input, step=0.1, min=-10, max=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle initDisp popover with button (or validator callback above!!)
@app.callback(
    Output("forceAmp_popover-FV", "is_open"),
    [Input("forceAmp-popover-target-FV", "n_clicks")],
    [State("forceAmp_popover-FV", "is_open")],
)
def forceAmp_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



######### w x-axis limit VALIDATOR #########################

@app.callback(
    Output("wlim_validation_message-FV", "children"),
    Output("wlim-popover-target-FV", "n_clicks"),
    Input("wlim-FV", "value")
)
def wlim_input_validator(wlim_input):
    err_string, is_invalid = validate_input("ω axis limit ", wlim_input, step=0.1, min=0.1, max=100)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle wlim popover with button (or validator callback above!!)
@app.callback(
    Output("wlim_popover-FV", "is_open"),
    [Input("wlim-popover-target-FV", "n_clicks")],
    [State("wlim_popover-FV", "is_open")],
)
def wlim_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open






# ======== Damping Ratio & Coefficient Updates =============

# This function disables the damping ratio or damping coefficient input using the toggle
@app.callback(
    Output("dampRatio-FV", "disabled"),
    Output("c-FV", "disabled"),
    Input("damping-switch-FV", "value")
)
def damping_toggle(switch):
    switch_state = len(switch)
    return switch_state, not switch_state


# This function updates damping coefficient c when it is disabled and other values are inputted
@app.callback(
    Output(component_id='c-FV', component_property='value'),
    Input(component_id='c-FV', component_property='disabled'),
    Input(component_id='c-FV', component_property='value'),
    Input(component_id='dampRatio-FV', component_property='value'),
    Input(component_id='k-FV', component_property='value'),
    Input(component_id='m-FV', component_property='value')
)
def update_c(c_disabled, c, dampRatio, k, m):
    if c_disabled and m!=None and k!=None and dampRatio!=None:
        c = np.round((dampRatio * 2 * np.sqrt(k * m)),3)
    return c


# This function updates damping ratio when it is disabled and other values are inputted
@app.callback(
    Output(component_id='dampRatio-FV', component_property='value'),
    Input(component_id='dampRatio-FV', component_property='disabled'),
    Input(component_id='dampRatio-FV', component_property='value'),
    Input(component_id='c-FV', component_property='value'),
    State(component_id='k-FV', component_property='value'),
    State(component_id='m-FV', component_property='value')
)
def update_damping_ratio(dampRatio_disabled, dampRatio, c, k, m):
    if dampRatio_disabled and m!=None and k!=None and c!=None:
        dampRatio = np.round((c / (2 * np.sqrt(k * m))),3)
    return dampRatio












# ============ Plotting Graph ========================

# This Function plots the FRF graph
@app.callback(Output('FRFAmp_plot', 'figure'),
              Output('FRFPhase_plot', 'figure'),
              Output('input_warning_string-FV', 'children'),
              Output('system_params-FV', 'children'),
              Output('w-slider', 'max'),
              Output('w-slider', 'marks'),
              Input('submit-button-state-FV', 'n_clicks'),
              State('m-FV', 'value'),
              State('k-FV', 'value'),
              State('dampRatio-FV', 'value'),
              State('c-FV', 'value'),
              State('x0-FV', 'value'),
              State('F0-FV', 'value'),
              State('wlim-FV', 'value'),
              )
def update_output(n_clicks, m, k, dampRatio, dampCoeff, x0, F0, wlim):

    dampRatios = [dampRatio]
    tend = 1    # So doesn't flag validator

    # This is to change slider limits according to wlim
    slider_marks = {
                       0: '0 rad/s',
                       wlim: str(wlim) + ' rad/s',
                   },

    is_invalid = validate_all_inputsFV(m,k,dampRatio, dampCoeff, x0, F0, wlim)
    print(is_invalid)
    if(is_invalid):
        solutionTypeString = "Graph was not Updated. Please check your inputs before Submitting"
        FRFAmp_fig = px.line(x=[0], y=[0],
                      labels=dict(
                          x="Excitation frequency (rad/s)",
                          y="x/F (m/N)"
                      )
                      )
        FRFPhase_fig = px.line(x=[0], y=[0],
                      labels=dict(
                          x="Excitation frequency (rad/s)",
                          y="Phase (rad)"
                      )
                      )
        input_warning_string = ["Graph was cleared!", html.Br(),
                                "Please check your inputs before Submitting again!"]
        system_params = [""]
        return FRFAmp_fig, FRFPhase_fig,  input_warning_string, system_params, wlim, slider_marks[0]
    else:
        amp, phase, r, w, wn, wnHz, wd, wdHz = FRF_Solver(m, k, dampRatios, wlim, wantNormalised=False)
        # print(w)
        # print(amp[0])
        FRFAmp_fig = px.line(x=w, y=amp[0],
                      labels=dict(
                          x="Excitation frequency (rad/s)",
                          y="x/F (m/N)"
                      )
                      )
        FRFPhase_fig = px.line(x=w, y=phase[0],
                      labels=dict(
                          x="Excitation frequency (rad/s)",
                          y="Phase (rad)"
                      )
                      )

        input_warning_string = ""
        # Create a string here!!!!!! Make solver function spit out the frequency, Hz and rad/s and max amplitude!!! ====================================
        if dampRatio > 0 and dampRatio < 1:
            # If the system is underdamped there will be damped natural freq
            dampedNatFreq_string = ["Damped Natural Frequency, ωd (rad/s): " + str(wd) + " rad/s", html.Br(),
                                    "Damped Natural Frequency, ωd (Hz): " + str(wdHz) + " Hz", html.Br(), ]
        else:
            # Otherwise no damped nat frequ
            dampedNatFreq_string = [""]

        system_params = ["Please scroll down to see your solution.", html.Br(), html.Br(),
                         "System Parameters:", html.Br(),
                         "Natural Frequency, ωn (rad/s): " + str(wn) + " rad/s", html.Br(),
                         "Natural Frequency, ωn (Hz): " + str(wnHz) + " Hz", html.Br(),
                         ] + dampedNatFreq_string


    return FRFAmp_fig, FRFPhase_fig, input_warning_string, system_params, wlim, slider_marks[0]

def FRF_Solver(m=10, k=10, dampRatios=[0.25], wlim=100, wantNormalised = False):


    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    wnHz = wn / (2 * np.pi)  # Natural freq in Hz
    if 0 < dampRatios[0] < 1:
        wd = wn * np.sqrt(1 - dampRatios[0] ** 2)  # Damped nat freq (rad/s)
        wdHz = wd / (2 * np.pi)  # Damped Natural freq in Hz
    else:
        wd = 0
        wdHz = 0

    w = np.linspace(0, wlim, 10000)   #SET LIMIT HERE FOR X AXIS!!!
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
            # print(dampRat)
            amp[row, :] = 1 / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
            phase[row, :] = np.arctan(-c * w / (k - m * w ** 2))
            phase[phase > 0] = phase[phase > 0] - np.pi
            row = row + 1


    return amp, phase, r, w, np.round(wn,decimals=2), np.round(wnHz,decimals=2), np.round(wd,decimals=2), np.round(wdHz,decimals=2)



## SLIDER OUTPUT AND F/X Time history plots
@app.callback(
    Output('w-slider-output-FV', 'children'),
    Output('timeHistory-plot-FV', 'figure'),
    Input('w-slider', 'value'),
    State('m-FV', 'value'),
    State('k-FV', 'value'),
    State('dampRatio-FV', 'value'),
    State('c-FV', 'value'),
    State('x0-FV', 'value'),
    State('F0-FV', 'value'),
)
def update_output(w_slider_value, m, k, dampRatio, c, x0, F0):
    slider_output_string = 'You have selected "{}" rad/s'.format(w_slider_value)
    wHz = w_slider_value/(2*np.pi)
    x, t, F, xf, tf = forcedSolver(m, k, dampRatio, c, x0, F0, wHz)

    # timeHistory_plot = px.line(df, x="year", y="lifeExp", color='country')
    df = {
        'Time': t,
        'Force, F': F,
        'Forced Response, x': x
    }
    timeHistory_plot = px.line(df,
                               x="Time",
                               y=['Force, F', 'Forced Response, x'],
                               labels={
                                   "Time": "Time (s)",
                                   "value": "Force(N)/Displacement(m) Amplitude",
                                   "variable": "Response"
                               }
                               )

    # print(timeHistory_plot.layout)

    return slider_output_string, timeHistory_plot



def forcedSolver(m=10, k=10 ** 6, dampRatio=0.1, c=100, x0=0, Famp=10, wHz=5):


    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    dampRatio = c / (2 * np.sqrt(k * m))
    wd = wn * np.sqrt(1 - dampRatio ** 2)  # Damped frequency
    w = 2 * np.pi * wHz  # Conv Forced freq from Hz into rad/s

    # Work out Nice time frame using decay to 1%
    t_decay = 1 / (dampRatio*wn) * np.log(1 / 0.01)
    tend = np.ceil(t_decay * 1.5)
    t = np.linspace(0, tend, 10000)
    x = t.copy()

    # Solving for Complete Forced Solution
    # Displacement amplitdue from force ONLY
    x0f = Famp / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
    phasef = np.arctan(c * w / (k - m * w ** 2))

    A = x0 - x0f * np.sin(-phasef)
    B = (dampRatio * wn * A - x0f * w * np.cos(-phasef)) / wd

    x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t)) + x0f * np.sin(w * t - phasef)

    # Only the Forcing amplitude and it's relevant displacment
    # Shorter time scale, tf so can see phase shift
    tf = np.linspace(0, 3, 1000)
    F = Famp * np.sin(w * t)
    xf = x0f * np.sin(w * tf - phasef)

    print(x)


    return x, t, F, xf, tf
