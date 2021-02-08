import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import server
# Import all the diff app pages
from apps import SDOF, ForcedVib, VibrationIsolation, BaseExcitation    #


app.layout = html.Div([
    html.H1("ME2 Vibrations App", className="title"),
    html.Div([
        dcc.Link('SDOF | ', href='/apps/SDOF'),
        dcc.Link('Forced Vibrations', href='/apps/ForcedVib'),
    ], className="pageLinks"),

    # dcc location is the way we change pages. so
    dcc.Location(id='url', pathname='', refresh=False),
    # All the content of each app goes in this DIV!!!!
    html.Div(id='page-content')
])


# This is to change pages
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '':
        return SDOF.layout      #Default First Page is SDOF
    elif pathname == '/apps/SDOF':
        return SDOF.layout
    elif pathname == '/apps/ForcedVib':
        return ForcedVib.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)