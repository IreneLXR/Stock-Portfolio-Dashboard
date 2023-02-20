import dash
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc
from dash.dependencies import Input, Output

import pandas as pd
import yfinance as yf

# Set available tickers
ticker_list = ['MSFT', 'GOOG', 'AAPL']

# Set up color theme: white and grey
colors = {
    'background': '#F3F3F3',
    'boxBackground': '#FFFFFF',
    'text': '#000000'
}

# Dash app
app = Dash()

app.layout = html.Div([
    # header
    html.Div([
        html.Center(html.H1('FRE6191 Project Part I')),
        html.Center(html.H5('Team 6: Layla Li (xl4432), Helen Zhang (jz5582), Sofia Guo (fg2283), Rita Wang (rw3167)')),
    ], style={
        'width': '100%',
        'background-color': colors['boxBackground'],
        'top': '0px',
        'height': '10%',
        'margin': '0',
        'position': 'absolute',
    }),

    # main body
    html.Div([
        # inputs area
        html.Div([
            html.Label('Available Stocks:'),
            dcc.Dropdown(
                options=ticker_list,
                placeholder='Select stocks',
                multi=True,
                id='select-ticker'),
                ], style={
                    'display': 'inline-block', 
                    'width': '27%',
                    'padding': '2rem',
                    'marginRight': '1.5%',
                    'marginLeft': '1.5%',
                    'border-radius': '10px',
                    'boxShadow': '#EBEBEB 4px 4px 2px',
                    'background-color': colors['boxBackground']
        }),

        # dashboard area
        html.Div([
            html.Div(
                id='intro'
            ),
            html.Br(),
            dcc.Graph(id='time-series-graph')
            ], style={
                'display': 'inline-block',
                'width': '67%', 
                'padding': '2rem',
                'marginRight': '1.5%',
                'marginLeft': '1.5%',
                'border-radius': '10px',
                'boxShadow': '#EBEBEB 5px 5px 3px',
                'background-color': colors['boxBackground']
        }),
    ], style={
        'display':'flex',
        'flex-direction': 'row',
        'height': '85%',
        'width': '100%',
        'margin': '0',
        'position': 'absolute',
        'top': '12%'
    }),

    # footer
    html.Div([
        html.Center(html.Footer("@FRE6191 Team 6")),
    ], style={
        'position': 'absolute',
        'bottom': '0px',
        'width': '100%',
        'margin': '0',
    }),
    
    ],
    style={
        'background-color': colors['background'],
        'color': colors['text'],
        'height': '100vh',
        'width': '100vw',
        'position': 'fixed',
        'padding': '0',
        'margin': '0',
        # eliminate body's 8px margin
        'margin-top':'-8px',
        'margin-left': '-8px',
        'overflow-x': 'hidden',
        'overflow-y': 'hidden'
})

@app.callback(Output('intro', 'children'), [Input('select-ticker', 'value')])
def update_intro(value):
    print(value)
    if value == None:
        return "This web application displays a graph of the historical prices of selected stocks"
    return "This web application displays a graph of the historical prices of " + ", ".join(value)

@app.callback(Output('time-series-graph', 'figure'), [Input('select-ticker', 'value')])
def draw_time_series_graph(value):
    # prepare data
    fig = go.Figure()
    if value == None:
        return fig
    selected_tickers = [yf.Ticker(val) for val in value]
    infos = [selected_ticker.history(period='max') for selected_ticker in selected_tickers]
    ticker_to_info = dict(zip(value, infos))
    for v in value:
        ticker_to_info[v] = ticker_to_info[v].reset_index()

    # draw graph
    for v in value:
        fig.add_trace(go.Scatter(x=ticker_to_info[v]['Date'], y=ticker_to_info[v]['Close'], name=v))
    
    fig.update_layout(
        title_text="Time Series Graph of " + ",".join(value),
        yaxis={'title': 'Price'}
    )
    # add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                    label='1m',
                    step="month",
                    stepmode="backward"),
                    dict(count=6,
                    label='6m',
                    step="month",
                    stepmode="backward"),
                    dict(count=1,
                    label='YTD',
                    step="year",
                    stepmode="todate"),
                    dict(count=1,
                    label='1y',
                    step="year",
                    stepmode="backward"),
                ])
            ),
            rangeslider_visible=True,
            type='date'
        )
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)