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
        if(remainder != 0):
            steperr = "\n    • The minimum increment is " + str(step)
            is_invalid = True
        else:
            steperr = "\n    • The minimum increment is greater than " + str(step)
        if(value < min):
            minerr = "\n    • The value is not smaller than " + str(min)
            is_invalid = True
        else:
            minerr = "\n    • The value is greater than " + str(min)
        if(max!=None):
            if(value>max):
                maxerr = "\n    • The value is not larger than " + str(max)
                is_invalid = True
            else:
                maxerr = "\n    •The value is smaller than the maximum limit of " + str(max)
        if(is_invalid):
            err = ["Your " + name + " input is invalid please ensure:", html.Br(), steperr, html.Br(), minerr, html.Br(), maxerr]
        else:
            err = ["Your "+ name + " input is valid as: ", html.Br(), steperr, html.Br(), minerr, html.Br(), maxerr]
    return err, is_invalid

err, is_invalid = validate_input("mass", value=None, step=0.01, min=0, max=100)
print("Is it Invalid?",is_invalid)
print(err)
