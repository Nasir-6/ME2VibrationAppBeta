import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

# For DUAL AXIS
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validator import *
from app import app

header = html.H3('Base Excitation', className=" mt-2, text-center")
about_Text = html.P([
                        "This Base Excitation solver takes in your parameters and then produces a motion transmissibility curve. You can then choose a frequency to view the time history plot at that specific frequency."
                        "Try it out by changing the input parameters and pressing submit to view your solution at the bottom of the page.)"])

damp_switch = dbc.FormGroup(
    [
        dbc.Checklist(
            options=[
                {"label": "Use Damping Coefficient", "value": 1}
            ],
            value=[],
            id="damping-switch-BE",
            switch=True,
        ),
    ]
)

# ================== ALL POPOVER COMPONENTS
mass_popover = html.Div(
    [
        dbc.Button(
            "?", id="mass-popover-target-BE", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Mass Input"),
                dbc.PopoverBody([], id="mass_validation_message-BE"),
            ],
            id="mass_popover-BE",
            is_open=False,
            target="mass-popover-target-BE",
        ),
    ],
)

springConst_popover = html.Div(
    [
        dbc.Button(
            "?", id="springConst-popover-target-BE", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Sprint Constant Input"),
                dbc.PopoverBody([], id="springConst_validation_message-BE"),
            ],
            id="springConst_popover-BE",
            is_open=False,
            target="springConst-popover-target-BE",
        ),
    ],
)

dampRatio_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampRatio-popover-target-BE", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Ratio Input"),
                dbc.PopoverBody([], id="dampRatio_validation_message-BE"),
            ],
            id="dampRatio_popover-BE",
            is_open=False,
            target="dampRatio-popover-target-BE",
        ),
    ],
)

dampCoeff_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampCoeff-popover-target-BE", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Coefficient Input"),
                dbc.PopoverBody([], id="dampCoeff_validation_message-BE"),
            ],
            id="dampCoeff_popover-BE",
            is_open=False,
            target="dampCoeff-popover-target-BE",
        ),
    ],
)


baseAmp_popover = html.Div(
    [
        dbc.Button(
            "?", id="baseAmp-popover-target-BE", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Base Excitation Amplitude"),
                dbc.PopoverBody([], id="baseAmp_validation_message-BE"),
            ],
            id="baseAmp_popover-BE",
            is_open=False,
            target="baseAmp-popover-target-BE",
        ),
    ],
)

wlim_popover = html.Div(
    [
        dbc.Button(
            "?", id="wlim-popover-target-BE", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("ω axis limit Input"),
                dbc.PopoverBody([], id="wlim_validation_message-BE"),
            ],
            id="wlim_popover-BE",
            is_open=False,
            target="wlim-popover-target-BE",
        ),
    ],
)

line1_input = dbc.Row([

    dbc.Col(
        html.Img(src=app.get_asset_url('BaseExcitation.png'),
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
                        id="m-BE",
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
                    dbc.Input(id="k-BE", placeholder="N/m", debounce=True, type="number",
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
                    dbc.InputGroupAddon("Damping Ratio, ζ", addon_type="prepend"),
                    dbc.Input(id="dampRatio-BE", placeholder="", debounce=True, type="number", value=0.1, min=0, max=2,
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
                    dbc.Input(id="c-BE", placeholder="Ns/m", debounce=True, type="number", value=1, min=0, step=0.001),
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
                    dbc.InputGroupAddon("Base Amplitude, y0 (m)", addon_type="prepend"),
                    dbc.Input(id="y0-BE", placeholder="m", debounce=True, type="number", value=0.1, min=-10, max=10,
                              step=0.001),
                    dbc.InputGroupAddon(
                        baseAmp_popover,
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
                    dbc.InputGroupAddon("ω axis limit , ω (Hz)", addon_type="prepend"),
                    dbc.Input(id="wlim-BE", placeholder="s", debounce=True, type="number", value=40, min=0.1, max=10000,
                              step=0.1),
                    dbc.InputGroupAddon(
                        wlim_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Button("Submit", color="secondary", id='submit-button-state-BE', size="sm")
        ]),
        dbc.Row(html.P(id="input_warning_string-BE", className="text-danger")),
        dbc.Row(html.P(id="system_params-BE")),

    ]),

], className="jumbotron")

layout = dbc.Container([
    header,
    about_Text,
    line1_input,
    html.H3("FRF and Time history plot of your desired base excitation frequency", className=" mt-1 mb-1 text-center"),
    html.H4("Please choose a excitation frequency using the slider below", className=" mt-1 mb-1 text-center"),
    dbc.Row([
        dbc.Col(
            [
                dcc.Slider(id="w-slider-BE",
                           min=0,
                           max=40,
                           step=0.01,
                           value=10,
                           marks={
                               0: '0 Hz',
                               100: '40 Hz'
                           },
                           updatemode='mouseup'
                           ),
            ],
            className="mb-3 col-12 col-sm-12 col-md-12 col-lg-4"
        )
    ]),
    dbc.Row([
        dbc.Col(
            [
                dcc.Graph(id='MotionTransmissibility_plot', figure={}),
            ],
            className="mb-1 p-0 col-12 col-sm-12 col-md-12 col-lg-4"
        ),
        dbc.Col(
            [
                dcc.Graph(id='timeHistory-plot-BE', figure={}),
            ],
            className="mb-1 p-0 col-12 col-sm-12 col-md-12 col-lg-8"
        ),
    ]),

], fluid=True)


# # THIS IS A PLACEHOLDER FOR BETA RELEASE - COMING SOON PAGE!!! ============================================================
# layout = dbc.Container([
#     html.H3('base Vibration', className=" mt-2, text-center"),
#     html.H6("This module is currently under development. Do come back at a later date.",className=" mt-2, text-center" ),
# ], fluid=True)


# ALL APP CALLBACKS

# INPUT VALIDATORS
@app.callback(
    Output("mass_validation_message-BE", "children"),
    Output("mass-popover-target-BE", "n_clicks"),
    Input("m-BE", "value")
)
def mass_input_validator(mass_input):
    err_string, is_invalid = validate_input("mass", mass_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle mass popover with button (or validator callback above!!)
@app.callback(
    Output("mass_popover-BE", "is_open"),
    [Input("mass-popover-target-BE", "n_clicks")],
    [State("mass_popover-BE", "is_open")],
)
def mass_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("springConst_validation_message-BE", "children"),
    Output("springConst-popover-target-BE", "n_clicks"),
    Input("k-BE", "value")
)
def springConst_input_validator(springConst_input):
    err_string, is_invalid = validate_input("spring constant", springConst_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle springConst popover with button (or validator callback above!!)
@app.callback(
    Output("springConst_popover-BE", "is_open"),
    [Input("springConst-popover-target-BE", "n_clicks")],
    [State("springConst_popover-BE", "is_open")],
)
def springConst_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("dampRatio_validation_message-BE", "children"),
    Output("dampRatio-popover-target-BE", "n_clicks"),
    Input("dampRatio-BE", "value")
)
def dampRatio_input_validator(dampRatio_input):
    err_string, is_invalid = validate_input("damping ratio", dampRatio_input, step=0.001, min=0, max=2)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle dampRatio popover with button (or validator callback above!!)
@app.callback(
    Output("dampRatio_popover-BE", "is_open"),
    [Input("dampRatio-popover-target-BE", "n_clicks")],
    [State("dampRatio_popover-BE", "is_open")],
)
def dampRatio_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


######### DAMPING COEFFFICIENT VALIDATOR #########################
@app.callback(
    Output("dampCoeff_validation_message-BE", "children"),
    Output("dampCoeff-popover-target-BE", "n_clicks"),
    Input("c-BE", "value")
)
def dampCoeff_input_validator(dampCoeff_input):
    err_string, is_invalid = validate_input("damping coefficient", dampCoeff_input, step=0.001, min=0)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle dampCoeff popover with button (or validator callback above!!)
@app.callback(
    Output("dampCoeff_popover-BE", "is_open"),
    [Input("dampCoeff-popover-target-BE", "n_clicks")],
    [State("dampCoeff_popover-BE", "is_open")],
)
def dampCoeff_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


######### Base Amplitude VALIDATOR #########################

@app.callback(
    Output("baseAmp_validation_message-BE", "children"),
    Output("baseAmp-popover-target-BE", "n_clicks"),
    Input("y0-BE", "value")
)
def baseAmp_input_validator(baseAmp_input):
    err_string, is_invalid = validate_input("base amplitude", baseAmp_input, step=0.001, min=-10, max=10)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle initDisp popover with button (or validator callback above!!)
@app.callback(
    Output("baseAmp_popover-BE", "is_open"),
    [Input("baseAmp-popover-target-BE", "n_clicks")],
    [State("baseAmp_popover-BE", "is_open")],
)
def baseAmp_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


######### w x-axis limit VALIDATOR #########################

@app.callback(
    Output("wlim_validation_message-BE", "children"),
    Output("wlim-popover-target-BE", "n_clicks"),
    Input("wlim-BE", "value")
)
def wlim_input_validator(wlim_input):
    err_string, is_invalid = validate_input("ω axis limit ", wlim_input, step=0.1, min=0.1, max=10000)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle wlim popover with button (or validator callback above!!)
@app.callback(
    Output("wlim_popover-BE", "is_open"),
    [Input("wlim-popover-target-BE", "n_clicks")],
    [State("wlim_popover-BE", "is_open")],
)
def wlim_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


# ======== Damping Ratio & Coefficient Updates =============

# This function disables the damping ratio or damping coefficient input using the toggle
@app.callback(
    Output("dampRatio-BE", "disabled"),
    Output("c-BE", "disabled"),
    Input("damping-switch-BE", "value")
)
def damping_toggle(switch):
    switch_state = len(switch)
    return switch_state, not switch_state


# This function updates damping coefficient c when it is disabled and other values are inputted
@app.callback(
    Output(component_id='c-BE', component_property='value'),
    Input(component_id='c-BE', component_property='disabled'),
    Input(component_id='c-BE', component_property='value'),
    Input(component_id='dampRatio-BE', component_property='value'),
    Input(component_id='k-BE', component_property='value'),
    Input(component_id='m-BE', component_property='value')
)
def update_c(c_disabled, c, dampRatio, k, m):
    if c_disabled and m != None and k != None and dampRatio != None:
        c = np.round((dampRatio * 2 * np.sqrt(k * m)), 3)
    return c


# This function updates damping ratio when it is disabled and other values are inputted
@app.callback(
    Output(component_id='dampRatio-BE', component_property='value'),
    Input(component_id='dampRatio-BE', component_property='disabled'),
    Input(component_id='dampRatio-BE', component_property='value'),
    Input(component_id='c-BE', component_property='value'),
    State(component_id='k-BE', component_property='value'),
    State(component_id='m-BE', component_property='value')
)
def update_damping_ratio(dampRatio_disabled, dampRatio, c, k, m):
    if dampRatio_disabled and m != None and k != None and c != None:
        dampRatio = np.round((c / (2 * np.sqrt(k * m))), 3)
    return dampRatio


# ============ Plotting Graph ========================

# This Function plots the MotionTransmissibility graph
@app.callback(Output('MotionTransmissibility_plot', 'figure'),
              Output('input_warning_string-BE', 'children'),
              Output('system_params-BE', 'children'),
              Output('w-slider-BE', 'max'),
              Output('w-slider-BE', 'marks'),
              Output('w-slider-BE', 'value'),
              Input('submit-button-state-BE', 'n_clicks'),
              Input('w-slider-BE', 'value'),
              State('m-BE', 'value'),
              State('k-BE', 'value'),
              State('dampRatio-BE', 'value'),
              State('c-BE', 'value'),
              State('y0-BE', 'value'),
              State('wlim-BE', 'value'),
              )
def update_output(n_clicks, w_slider_value, m, k, dampRatio, dampCoeff, y0, wlim):
    dampRatios = [dampRatio]
    tend = 1  # So doesn't flag validator

    # This is to change slider limits according to wlim
    slider_marks = {
                       0: '0 Hz',
                       wlim: str(wlim) + ' Hz',
                   },

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add figure title
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.92
        ),
        margin=dict(
            t=30,
            b=10,
            r=10,
        ),
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Excitation frequency (Hz)")

    # Set y-axes titles
    fig.update_yaxes(title_text="T = x0/y0 (-)", secondary_y=False)
    fig.update_yaxes(title_text="Phase (Degrees)", secondary_y=True, showgrid=False)

    is_invalid = validate_all_inputsBE(m, k, dampRatio, dampCoeff, y0, wlim)

    if (is_invalid):
        w_slider_value = 0 # Set to 0 so can empty time history plot!
        fig.add_trace(
            go.Scatter(x=[0], y=[0], name="Amplitude"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=[0], y=[0], name= "Phase"),
            secondary_y=True,
        )

        input_warning_string = ["Graph was cleared!", html.Br(),
                                "Please check your inputs before Submitting again!"]
        system_params = [""]
        return fig, input_warning_string, system_params, wlim, slider_marks[0], w_slider_value
    else:
        Tamp, phase, r, w, wn, wnHz, wd, wdHz = MotionTransmissibility_Solver(m, k, dampRatios, wlim, wantNormalised=False)
        # print(w)
        # print(amp[0])

        # THIS IS DUAL AXIS PLOT
        # Create figure with secondary y-axis

        # Add traces
        fig.add_trace(
            go.Scatter(x=w, y=Tamp[0], name="Amplitude"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=w, y=-phase[0]*180/np.pi, name="Phase"),
            secondary_y=True,
        )
        # Adding vertical line indicating chosen w using slider
        fig.add_vline(x=w_slider_value, line_width=2, line_dash="dash", line_color="red", annotation_text='{} Hz'.format(w_slider_value),  annotation_position="right")


        input_warning_string = ""
        # Create a string here!!!!!! Make solver function spit out the frequency, Hz and rad/s and max amplitude!!! ====================================
        if 0 < dampRatio < 1:
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

    return fig, input_warning_string, system_params, wlim, slider_marks[0], w_slider_value


def MotionTransmissibility_Solver(m=10, k=10, dampRatios=[0.25], wlim=50, wantNormalised=False):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    wnHz = wn / (2 * np.pi)  # Natural freq in Hz
    if 0 < dampRatios[0] < 1:
        wd = wn * np.sqrt(1 - dampRatios[0] ** 2)  # Damped nat freq (rad/s)
        wdHz = wd / (2 * np.pi)  # Damped Natural freq in Hz
    else:
        wd = 0
        wdHz = 0

    wHz_axis = np.linspace(0, wlim, 1000)  # SET LIMIT HERE FOR X AXIS!!!
    w = wHz_axis*2*np.pi
    r = w / wn

    Tamp = np.zeros((len(dampRatios), len(w)))
    phase = np.zeros((len(dampRatios), len(w)))
    if wantNormalised:
        row = 0
        for dampRat in dampRatios:
            # print(dampRat)
            Tamp[row, :] = 1 / np.sqrt((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2)

            # NEED TO SORT OUT NORMALISED PHASE!!
            gamma = np.arctan(2*dampRat*r / 1)
            if m * w ** 2 > k:
                beta = np.arctan(2*dampRat*r / (1 - r ** 2)) + np.pi
            else:
                beta = np.arctan(2*dampRat*r / (1 - r ** 2))
            phase[row, :] = beta - gamma
            phase[phase > 0] = phase[phase > 0] - np.pi
            row = row + 1
    else:
        row = 0
        for dampRat in dampRatios:
            c = dampRat * 2 * np.sqrt(k * m)
            # print(dampRat)
            Tamp[row, :] = np.sqrt((k**2 + (c*w)**2)/((k - m * w ** 2) ** 2 + (c * w) ** 2))
            beta = np.arctan(c * w / (k - m * w ** 2)) + np.pi
            i=0
            for wval in w:
                if m * wval ** 2 > k:
                    beta[i] = np.arctan(c * wval / (k - m * wval ** 2)) + np.pi
                else:
                    beta[i] = np.arctan(c * wval / (k - m * wval ** 2))
                i=i+1

        gamma = np.arctan(c * w / k)

        # Working out Phase
        # theta = phi + alpha  # + alpha as Phi is negative
        phase[row, :] = beta - gamma
        # phase[phase >= 0] = phase[phase >= 0] - np.pi
        row = row + 1

    return Tamp, phase, r, wHz_axis, np.round(wn, decimals=2), np.round(wnHz, decimals=2), np.round(wd, decimals=2), np.round(
        wdHz, decimals=2)


## SLIDER OUTPUT AND F/X Time history plots
@app.callback(
    Output('timeHistory-plot-BE', 'figure'),
    Input('w-slider-BE', 'value'),
    State('m-BE', 'value'),
    State('k-BE', 'value'),
    State('dampRatio-BE', 'value'),
    State('c-BE', 'value'),
    State('y0-BE', 'value'),
)
def update_output_time_hist(w_slider_value, m, k, dampRatio, c, y0):


    # THIS IS DUAL AXIS PLOT
    # Create figure with secondary y-axis
    timeHistory_plot = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces
    if(w_slider_value==0):
        # Empty time history plot
        timeHistory_plot.add_trace(
            go.Scatter(x=[0], y=[0], name="Mass displacement, x"),
            secondary_y=False,
        )
        timeHistory_plot.add_trace(
            go.Scatter(x=[0], y=[0], name="Base displacement, y"),
            secondary_y=False,
        )
    else:
        wHz = w_slider_value
        t, y, x = MotionTransmissibilityTimeHistorySolver(m, k, dampRatio, c, y0, wHz)
        timeHistory_plot.add_trace(
            go.Scatter(x=t, y=x, name="Mass displacement, x"),
            secondary_y=False,
        )
        timeHistory_plot.add_trace(
            go.Scatter(x=t, y=y, name="Base displacement, y"),
            secondary_y=False,
        )

    # Add figure title
    timeHistory_plot.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.92
        ),
        margin=dict(
            t=30,
            b=10,
            r=10,
        ),
    )

    # timeHistory_plot.update_yaxes(range=[-1.1 * max(abs(Ft)), 1.1 * max(abs(Ft))], secondary_y=False)

    # Set x-axis title
    timeHistory_plot.update_xaxes(title_text="Time (s)")

    # Set y-axes titles
    timeHistory_plot.update_yaxes(title_text="Displacement Amplitude (m)", secondary_y=False)

    # print(timeHistory_plot.layout)

    return timeHistory_plot


def MotionTransmissibilityTimeHistorySolver(m=10, k=10 ** 6, dampRatio=0.1, c=100, yAmp=10, wHz=5):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    wd = wn * np.sqrt(1 - dampRatio ** 2)  # Damped frequency
    w = 2 * np.pi * wHz  # Conv Forced freq from Hz into rad/s

    # Work out Nice time frame using decay to 1%
    f = wHz
    t_one_wave = 1/f
    tend = t_one_wave * 6
    t = np.linspace(0, tend, 1000)

    # Solving for Complete Forced Solution
    Tamp = np.sqrt((k**2 + (c*w)**2)/((k - m * w ** 2) ** 2 + (c * w) ** 2))

    if m * w ** 2 > k:
        beta = np.arctan(c*w/(k-m*w**2)) + np.pi
    else:
        beta = np.arctan(c*w/(k-m*w**2))

    gamma = np.arctan(c * w / k)
    # Working out Phase
    # theta = phi + alpha         # + alpha as Phi is negative
    phase = beta - gamma

    y = yAmp * np.sin(w * t)
    x = Tamp * yAmp * np.sin(w * t - phase)

    return t, y, x
