import numpy as np
from decimal import *
import dash_html_components as html

# This Function Takes in the properties of an input
# and returns a string highlighting the issue with the input
#   Variables
#       Name (string) = Used to print out which input is invalid
#       Value (num) = The value passed in via Input
#       Step (num) = The increments the input can go up via
#       Min (num) = The Minimum value must be
#       Max (num) = Maximum limit
#   Output
#       err (string) = String explaining the input issue
#       is_invalid (bool) = Flag to
def validate_input(name, value, step, min, max=None):

    is_invalid = False
    err = ""
    steperr = ""
    minerr = ""
    maxerr = ""
    if(value==None):
        err = "You did not input a value for the " + str(name)
        is_invalid = True
    else:
        remainder = Decimal(str(value)) % Decimal(str(step))
        if(remainder != 0 and value!=0 and min!=0):
            steperr = "\n    • The minimum increment is " + str(step)
            is_invalid = True
        else:
            steperr = "\n    • The minimum increment is greater than " + str(step)
        if(value < min):
            minerr = "\n    • The value is not smaller than " + str(min)
            is_invalid = True
        else:
            minerr = "\n    • The value is greater than or equal to " + str(min)
        if(max!=None):
            if(value>max):
                maxerr = "\n    • The value is not larger than " + str(max)
                is_invalid = True
            else:
                maxerr = "\n    •The value is smaller than or equal to the maximum limit of " + str(max)
        if(is_invalid):
            err = ["Your " + name + " input is invalid please ensure:", html.Br(), steperr, html.Br(), minerr, html.Br(), maxerr]
        else:
            err = ["Your "+ name + " input is valid as: ", html.Br(), steperr, html.Br(), minerr, html.Br(), maxerr]
    return err, is_invalid


def validate_all_inputs(mass_input, springConst_input, dampRatio_input, dampCoeff_input, initDisp_input, tSpan_input, numPts_input):

    is_invalid = False

    err_string, mass_is_invalid = validate_input("mass", mass_input, step=0.001, min=0)
    err_string, k_is_invalid = validate_input("spring constant", springConst_input, step=0.001, min=0.001)
    err_string, dampRatio_is_invalid = validate_input("damping ratio", dampRatio_input, step=0.001, min=0, max=2)
    err_string, dampCoeff_is_invalid = validate_input("damping coefficient", dampCoeff_input, step=0.001, min=0)
    err_string, x0_is_invalid = validate_input("initial displacement", initDisp_input, step=0.1, min=-10, max=10)
    err_string, tSpan_is_invalid = validate_input("time span", tSpan_input, step=0.01, min=0.01, max=360)
    err_string, n_is_invalid = validate_input("number of points", numPts_input, step=1, min=10)

    if(mass_is_invalid or k_is_invalid or dampRatio_is_invalid or dampCoeff_is_invalid or x0_is_invalid or tSpan_is_invalid or n_is_invalid):
        is_invalid = True;

    return is_invalid









def validate_aliasing(m, k, tSpan, nPts):
    wn = np.sqrt(k / m)
    naturalFreqHz = wn/(2*np.pi)

    sampFreq = nPts/tSpan
    # print("Natural frequency is ", naturalFreqHz, "Hz")
    # print("1 wave time ", 1/naturalFreqHz, "s")
    # print("Sampling Frequency is ", sampFreq, "samples per sec")

    nPts_required = 2 * naturalFreqHz * tSpan

    if sampFreq< 2*naturalFreqHz:
        aliasing_warning = ["Please Ensure your sampling frequency is more than double the natural frequency.",
                            html.Br(),"You need well above "+ str(np.ceil(nPts_required)) + " points to prevent aliasing"]
    else:
        aliasing_warning = ""

    print(aliasing_warning)

    return aliasing_warning






# Testing Aliasing
# validate_aliasing(m=1, k=100, tSpan=1, nPts=1)

# # Testing validator outputs
# err, is_invalid = validate_input("mass", value=0, step=0.01, min=0, max=100)
# print("Is it Invalid?",is_invalid)
# print(err)
