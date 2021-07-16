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

header = html.H3('Forced Vibration', className=" mt-2, text-center")
about_Text = html.P([
                        "This Forced Vibrations solver takes in your parameters and then produces an FRF response. You can then choose a frequency to view the time history plot at that specific frequency."
                        "Try it out by changing the input parameters and pressing submit to view your solution at the bottom of the page.To submit feedback for this module please click ",
                     html.A("here",
                            href="https://forms.gle/W4DmoEKuGnu2RkWN6",
                            target="_blank"),
                     "."])

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

''' The Following are the all the popover components. 
They consist of the blue ? button, The popover itself with the header and the validation message 
telling the user why/why not the current input is valid'''

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

initialDisplacement_popover = html.Div(
    [
        dbc.Button(
            "?", id="initialDisplacement-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Initial Displacement Input"),
                dbc.PopoverBody([], id="initialDisplacement_validation_message-FV"),
            ],
            id="initialDisplacement_popover-FV",
            is_open=False,
            target="initialDisplacement-popover-target-FV",
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

wAxisLimit_popover = html.Div(
    [
        dbc.Button(
            "?", id="wAxisLimit-popover-target-FV", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("ω axis limit Input"),
                dbc.PopoverBody([], id="wAxisLimit_validation_message-FV"),
            ],
            id="wAxisLimit_popover-FV",
            is_open=False,
            target="wAxisLimit-popover-target-FV",
        ),
    ],
)


''' This is the set of inputs for the 1st system. It was done like this to accommodate multiple system inputs.'''
system1_input = dbc.Row([

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
                    dbc.Input(id="k-FV", placeholder="N/m", 
                              debounce=True, type="number",
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
                    dbc.Input(id="dampRatio-FV", placeholder="", 
                              debounce=True, type="number", 
                              value=0.1, min=0, max=2, step=0.001),
                    dbc.InputGroupAddon(
                        dampRatio_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Coefficient, c (Ns/m)", addon_type="prepend"),
                    dbc.Input(id="c-FV", placeholder="Ns/m", 
                              debounce=True, type="number", 
                              value=1, min=0, step=0.001),
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
                    dbc.Input(id="x0-FV", placeholder="m", 
                              debounce=True, type="number", 
                              value=0.1, min=-10, max=10, step=0.001),
                    dbc.InputGroupAddon(
                        initialDisplacement_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Forcing Amplitude, F0 (N)", addon_type="prepend"),
                    dbc.Input(id="F0-FV", placeholder="N", 
                              debounce=True, type="number", 
                              value=0.1, min=-10000, max=10000, step=0.01),
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
                    dbc.InputGroupAddon("ω axis limit , ω (Hz)", addon_type="prepend"),
                    dbc.Input(id="wAxisLimit-FV", placeholder="s", 
                              debounce=True, type="number", 
                              value=40, min=0.1, max=10000, step=0.1),
                    dbc.InputGroupAddon(
                        wAxisLimit_popover,
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


'''This is where the layout is formed from the components. Note the plots/graph components are defined here.'''
layout = dbc.Container([
    header,
    about_Text,
    system1_input,
    html.H3("FRF and Time history plot of your desired forcing frequency", className=" mt-1 mb-1 text-center"),
    html.H4("Please choose a excitation frequency using the slider below", className=" mt-1 mb-1 text-center"),
    dbc.Row([
        dbc.Col(
            [
                dcc.Slider(id="w-slider",
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
                dcc.Graph(id='FRF_plot', figure={}),
            ],
            className="mb-1 p-0 col-12 col-sm-12 col-md-12 col-lg-4"
        ),
        dbc.Col(
            [
                dcc.Graph(id='timeHistory-plot-FV', figure={}),
            ],
            className="mb-1 p-0 col-12 col-sm-12 col-md-12 col-lg-8"
        ),
    ]),

], fluid=True)



# ALL APP CALLBACKS
'''Here are all the app callbacks used. Note it is important that the id matches up to the Output/Input name'''


# ALL INPUT VALIDATORS and popover functions
# Each input has it's own validator passing in it's desired values to be checked against.
# They also have their own popover toggle function
@app.callback(
    Output("mass_validation_message-FV", "children"),
    Output("mass-popover-target-FV", "n_clicks"),
    Input("m-FV", "value")
)
def mass_input_validator(mass_input):
    err_string, is_invalid = validate_input("mass", mass_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle mass popover with button (or validator callback above!!)
# Note the n_clicks is set as an Input to ensure updating if changed
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
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


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
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


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
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


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
    Output("initialDisplacement_validation_message-FV", "children"),
    Output("initialDisplacement-popover-target-FV", "n_clicks"),
    Input("x0-FV", "value")
)
def initialDisplacement_input_validator(initialDisplacement_input):
    err_string, is_invalid = validate_input("initial displacement", initialDisplacement_input, step=0.001, min=-10, max=10)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle initialDisplacement popover with button (or validator callback above!!)
@app.callback(
    Output("initialDisplacement_popover-FV", "is_open"),
    [Input("initialDisplacement-popover-target-FV", "n_clicks")],
    [State("initialDisplacement_popover-FV", "is_open")],
)
def initialDisplacement_toggle_popover(n, is_open):
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
    err_string, is_invalid = validate_input("forcing amplitude", forceAmp_input, step=0.1, min=-10000, max=10000)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle initialDisplacement popover with button (or validator callback above!!)
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
    Output("wAxisLimit_validation_message-FV", "children"),
    Output("wAxisLimit-popover-target-FV", "n_clicks"),
    Input("wAxisLimit-FV", "value")
)
def wAxisLimit_input_validator(wAxisLimit_input):
    err_string, is_invalid = validate_input("ω axis limit ", wAxisLimit_input, step=0.1, min=0.1, max=10000)
    if is_invalid:
        return err_string, 1  # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0  # Set nclicks to 0 to prevent popover


# Toggle wAxisLimit popover with button (or validator callback above!!)
@app.callback(
    Output("wAxisLimit_popover-FV", "is_open"),
    [Input("wAxisLimit-popover-target-FV", "n_clicks")],
    [State("wAxisLimit_popover-FV", "is_open")],
)
def wAxisLimit_toggle_popover(n, is_open):
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
    if c_disabled and m != None and k != None and dampRatio != None:
        c = np.round((dampRatio * 2 * np.sqrt(k * m)), 3)
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
    if dampRatio_disabled and m != None and k != None and c != None:
        dampRatio = np.round((c / (2 * np.sqrt(k * m))), 3)
    return dampRatio


# ============ Plotting Graph ========================


'''
This update output function is for the FRF plot ONLY and takes in the inputs, passes it to the validator and solver and 
then updates the graphs/text using the solutions/values it gets back.
  Inputs
  n_clicks, w_slider_value, m, k, dampRatio, dampCoeff, x0, F0, wAxisLimit
      n_clicks (num) = number of clicks of the submit button (NOTE: This is used as the only Input in order to update 
                       the outputs when the submit button is clicked.)  
      w_slider_value (num) = The chosen frequency value of slider (Hz)
      m (num) = mass (kg)
      k (num) = Stiffness constant (N/m)
      dampRatio (num) = The Damping Ratio
      dampCoeff (num) = The Damping Coefficient (Ns/m) 
      x0 (num) = The initial displacement of the mass (m)
      F0 (num) = The Forcing Amplitude of the forcing function (N)
      wAxisLimit (num) = The chosen limit of the x axis which is used to recalibrate the axis and slider limits (Hz)
  Output
     return fig, input_warning_string, system_params, wAxisLimit, slider_marks[0], w_slider_value
     fig (Figure object) = The object that holds all the parameters to produce the figure
     input_warning_string (string) = A string that returns any warning messages if any in red 
     system_params (string array) = An array which concatenates any important parameters of the system to be 
                                    displayed as text.
     wAxisLimit (num) = The updated limit of the x axis which is used to recalibrate the slider limit (Hz)
     slider_marks[0] (object) = An object used to update the labels of the slider 
     w_slider_value (num) = This is the chosen w value which was Outputted to update the Time history plot 
    
'''
# IMPORTANT NOTE: Although similar inputs are used a new version must be used for the Forced Vibration module to avoid
# clashes and so the -FV suffix is added to any reused components from the SDOF module
@app.callback(Output('FRF_plot', 'figure'),
              Output('input_warning_string-FV', 'children'),
              Output('system_params-FV', 'children'),
              Output('w-slider', 'max'),
              Output('w-slider', 'marks'),
              Output('w-slider', 'value'),
              Input('submit-button-state-FV', 'n_clicks'),
              Input('w-slider', 'value'),
              State('m-FV', 'value'),
              State('k-FV', 'value'),
              State('dampRatio-FV', 'value'),
              State('c-FV', 'value'),
              State('x0-FV', 'value'),
              State('F0-FV', 'value'),
              State('wAxisLimit-FV', 'value'),
              )
def update_output(n_clicks, w_slider_value, m, k, dampRatio, dampCoeff, x0, F0, wAxisLimit):
    dampRatios = [dampRatio]
    tend = 1  # So doesn't flag validator

    # This is to change slider limits according to wAxisLimit
    slider_marks = {
                       0: '0 Hz',
                       wAxisLimit: str(wAxisLimit) + ' Hz',
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
    fig.update_yaxes(title_text="x/F (m/N)", secondary_y=False)
    fig.update_yaxes(title_text="Phase (Degrees)", secondary_y=True, showgrid=False)

    is_invalid = validate_all_inputsFV(m, k, dampRatio, dampCoeff, x0, F0, wAxisLimit)

    if (is_invalid):
        w_slider_value = 0  # Set to 0 so can empty time history plot!
        fig.add_trace(
            go.Scatter(x=[0], y=[0], name="FRF Amplitude"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=[0], y=[0], name="FRF Phase"),
            secondary_y=True,
        )

        input_warning_string = ["Graph was cleared!", html.Br(),
                                "Please check your inputs before Submitting again!"]
        system_params = [""]
        return fig, input_warning_string, system_params, wAxisLimit, slider_marks[0], w_slider_value
    else:
        # FIGURE OUT HOW TO SET NORMALISED VARIABLE!!!!!
        amp, phase, r, wHz_axis, wn, wnHz, wd, wdHz = FRF_Solver(m, k, dampRatios, wAxisLimit, wantNormalised=False)

        # THIS IS DUAL AXIS PLOT
        # Create figure with secondary y-axis

        # Add traces
        fig.add_trace(
            go.Scatter(x=wHz_axis, y=amp[0], name="FRF Amplitude"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=wHz_axis, y=phase[0]*180/np.pi, name="FRF Phase"),
            secondary_y=True,
        )
        # Adding vertical line indicating chosen w using slider
        fig.add_vline(x=w_slider_value, line_width=2, line_dash="dash", line_color="red",annotation_text='{} Hz'.format(w_slider_value),  annotation_position="right")

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

    return fig, input_warning_string, system_params, wAxisLimit, slider_marks[0], w_slider_value


def FRF_Solver(m=10, k=10, dampRatios=[0.25], wAxisLimit=100, wantNormalised=False):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    wnHz = wn / (2 * np.pi)  # Natural freq in Hz
    if 0 < dampRatios[0] < 1:
        wd = wn * np.sqrt(1 - dampRatios[0] ** 2)  # Damped nat freq (rad/s)
        wdHz = wd / (2 * np.pi)  # Damped Natural freq in Hz
    else:
        wd = 0
        wdHz = 0

    wHz_axis = np.linspace(0, wAxisLimit, 1000)  # SET LIMIT HERE FOR X AXIS!!!
    r = wHz_axis / wn
    w = 2*np.pi*wHz_axis

    amp = np.zeros((len(dampRatios), len(w)))
    phase = np.zeros((len(dampRatios), len(w)))
    if wantNormalised:
        row = 0
        for dampRat in dampRatios:

            amp[row, :] = 1 / np.sqrt((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2)
            phase[row, :] = np.arctan(-2 * dampRat * r / (1 - r ** 2))
            phase[phase > 0] = phase[phase > 0] - np.pi
            row = row + 1
    else:
        row = 0
        for dampRat in dampRatios:
            c = dampRat * 2 * np.sqrt(k * m)

            amp[row, :] = 1 / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
            phase[row, :] = np.arctan(-c * w / (k - m * w ** 2))
            phase[phase > 0] = phase[phase > 0] - np.pi
            row = row + 1

    return amp, phase, r, wHz_axis, np.round(wn, decimals=2), np.round(wnHz, decimals=2), np.round(wd, decimals=2), np.round(
        wdHz, decimals=2)


## SLIDER OUTPUT AND F/X Time history plots
@app.callback(
    Output('timeHistory-plot-FV', 'figure'),
    Input('w-slider', 'value'),
    State('m-FV', 'value'),
    State('k-FV', 'value'),
    State('dampRatio-FV', 'value'),
    State('c-FV', 'value'),
    State('x0-FV', 'value'),
    State('F0-FV', 'value'),
)
def update_output_time_hist(w_slider_value, m, k, dampRatio, c, x0, F0):

    # THIS IS DUAL AXIS PLOT
    # Create figure with secondary y-axis
    timeHistory_plot = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces

    if(w_slider_value==0):
        # Empty time history plot
        timeHistory_plot.add_trace(
            go.Scatter(x=[0], y=[0], name="Displacement Response, x"),
            secondary_y=False,
        )
        timeHistory_plot.add_trace(
            go.Scatter(x=[0], y=[0], name="Force, F"),
            secondary_y=True,
        )
    else:
        wHz = w_slider_value
        x, t, F = forcedSolver(m, k, dampRatio, c, x0, F0, wHz)
        timeHistory_plot.add_trace(
            go.Scatter(x=t, y=x, name="Displacement Response, x"),
            secondary_y=False,
        )
        timeHistory_plot.add_trace(
            go.Scatter(x=t, y=F, name="Force, F"),
            secondary_y=True,
        )
        timeHistory_plot.update_yaxes(range=[-1.1 * max(abs(x)), 1.1 * max(abs(x))], secondary_y=False)

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



    # Set x-axis title
    timeHistory_plot.update_xaxes(title_text="Time (s)")

    # Set y-axes titles
    timeHistory_plot.update_yaxes(title_text="Force Amplitude (N)", secondary_y=True)
    timeHistory_plot.update_yaxes(title_text="Displacement Response (m)", secondary_y=False)

    # print(timeHistory_plot.layout)

    return timeHistory_plot




def forcedSolver(m=1, k=1000, dampRatio=0.1, c=6.325, x0=0.1, Famp=0.1, wHz=10):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    if 0 < dampRatio < 1:
        wd = wn * np.sqrt(1 - dampRatio ** 2)  # Damped frequency
    else:
        wd = 0
    w = 2 * np.pi * wHz  # Conv Forced freq from Hz into rad/s

    # Work out Nice time frame using decay to 10%
    t_decay = 1 / (dampRatio * wn) * np.log(1 / 0.01)
    tend = t_decay * 1.25
    t = np.linspace(0, tend, 10000)
    x = t.copy()

    # Solving for Complete Forced Solution
    # Displacement amplitdue from force ONLY
    x0f = Famp / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
    phasef = np.arctan(c * w / (k - m * w ** 2))

    A = x0 - x0f * np.sin(-phasef)
    if 0 < dampRatio < 1:
        B = (dampRatio * wn * A - x0f * w * np.cos(-phasef)) / wd
    else:
        B = 0
    x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t)) + x0f * np.sin(w * t - phasef)

    # Only the Forcing amplitude and it's relevant displacment
    F = Famp * np.sin(w * t)


    return x, t, F

