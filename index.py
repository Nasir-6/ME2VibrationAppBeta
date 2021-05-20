'''
The Index file is the startup file. It contains the common layout which at this point in time is the navbar only.
It Contains all the page paths and is the place to add other pages if required. Note how the different layouts for each
page is defined in it's own file and is only called here.
'''


import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app
from app import server
# Import all the diff app pages
from apps import SDOF, ForcedVib, VibrationIsolation, BaseExcitation

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# Setting up Navigation Bar component

nav_bar_links = dbc.Row(
    [
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("SDOF", active="exact", href="/apps/SDOF")),
                dbc.NavItem(dbc.NavLink("Forced Vibration", active="exact", href="/apps/ForcedVib")),
                dbc.NavItem(dbc.NavLink("Vibration Isolation", active="exact", href="/apps/VibrationIsolation")),
                dbc.NavItem(dbc.NavLink("Base Excitation", active="exact", href="/apps/BaseExcitation")),
            ],
            pills=True,
        )
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)


navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("ME2 Vibrations App", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",   #Change LINK TO MAIN PAGE!!!
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        # These are all the links to each app page, which collapses at small screens
        dbc.Collapse(nav_bar_links, id="navbar-collapse", navbar=True),
    ],
    color="light",
    dark=False,
)



# Structure of the App
app.layout = dbc.Container([
    navbar,
    # dcc location is the way we change pages
    dcc.Location(id='url', pathname='', refresh=False),
    # All the content of each app goes in this DIV!!!!
    html.Div(id='page-content')
], fluid="True")



# Functions/Controller

# Callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),    # Input needed for callback!!!
    State("navbar-collapse", "is_open"),
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



# Callback to change pages
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '':
        return SDOF.layout      #Default First Page is SDOF
    elif pathname == '/apps/SDOF':
        return SDOF.layout
    elif pathname == '/apps/ForcedVib':
        return ForcedVib.layout
    elif pathname == '/apps/VibrationIsolation':
        return VibrationIsolation.layout
    elif pathname == '/apps/BaseExcitation':
        return BaseExcitation.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)