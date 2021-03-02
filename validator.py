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
#       err (string) = Contacatenated String explaining the input issues
#       is_invalid (bool) = Flag to raise error - REQUIRED FOR PREVENT SUBMIT
def validate_input(name, value, step, min, max=None):

    # Initialise is_invalid flag & error string
    is_invalid = False
    err = ""
    steperr = ""
    minerr = ""
    maxerr = ""

    if(value==None):    # If input box is empty
        err = "You did not input a value for the " + str(name)
        is_invalid = True
    else:
        # Use modulus to find if it has a min increment smaller or not
        # IMPORTANT: Decimal was used to deal with floating point issues!!!
        remainder = Decimal(str(value)) % Decimal(str(step))
        if(remainder != 0 and value!=0 and min!=0):
            steperr = "\n    • The minimum increment is " + str(step)
            is_invalid = True

        if(value < min):
            minerr = "\n    • The value is not smaller than " + str(min)
            is_invalid = True

        if(max!=None):      # Case of a max limit
            if(value>max):
                maxerr = "\n    • The value is not larger than " + str(max)
                is_invalid = True

        if(is_invalid):
            err = ["Your " + name + " input is invalid please ensure:", html.Br(), steperr, html.Br(), minerr, html.Br(), maxerr]
        else:
            steperr = "\n    • The minimum increment is greater than " + str(step)
            minerr = "\n    • The value is greater than or equal to " + str(min)
            if (max != None):  # Case of a max limit
                maxerr = "\n    •The value is smaller than or equal to the maximum limit of " + str(max)
            err = ["Your "+ name + " input is valid as: ", html.Br(), steperr, html.Br(), minerr, html.Br(), maxerr]
    return err, is_invalid


# Function to validate all inputs - THIS IS KEY FOR PREVENT UPDATE
#   Inputs
#       All inputs (THIS IS CURRENTLY FOR SDOF - NEED TO ALTER FOR FORCED VIB ETC.!!!)

#   Outputs
#       is_invalid (bool) = A variable to return whether a single input is invalid (TRUE) or if all inputs are valid (FALSE)
#                           This is required to prevent submit!!!
def validate_all_inputs(mass_input, springConst_input, dampRatio_input, dampCoeff_input, initDisp_input, tSpan_input, numPts_input):

    is_invalid = False

    # PLEASE ENSURE THAT THE "step", "min", and "max" are correct for their respective inputs!!!
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







# Function to check for aliasing - raises error message if there is aliasing
#   Input Variables
#       m = mass input (kg)
#       k = spring constant input (N/m)
#       tSpan = Time span input (sec)
#       nPts = Number of points input

#   Output
#       alaising_warning = A warning message that tells user the nyquist limit and a recommended number of points

def validate_aliasing(m, k, tSpan, nPts):
    wn = np.sqrt(k / m)
    naturalFreqHz = wn/(2*np.pi)

    sampFreq = nPts/tSpan
    # print("Natural frequency is ", naturalFreqHz, "Hz")
    # print("1 wave time ", 1/naturalFreqHz, "s")
    # print("Sampling Frequency is ", sampFreq, "samples per sec")

    # Nyquist-Shannon law to prevent aliasing (Will be the limit but still some issues)
    nPts_required = 2 * naturalFreqHz * tSpan
    # Give recommended amount to ensure wave is captured properly (double required)
    nPts_recommended = 4 * naturalFreqHz * tSpan

    if sampFreq < 2*naturalFreqHz:   # If Aliasing occurs with current input
        aliasing_warning = [
            "Please ensure your sampling frequency (Number of Points/Time Span), " + str(np.round(sampFreq, decimals=2)) +" Hz, is well above double the natural frequency of the system, "+ str(np.round(naturalFreqHz,decimals=2)) + " Hz.",
            html.Br(),"This is to prevent aliasing from occuring.",
            html.Br(),"We recommend increasing the number of points beyond " + str(np.ceil(nPts_recommended)) + " points."
                            ]
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
