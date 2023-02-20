import dash
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

import pandas as pd
import yfinance as yf

# Set available tickers
ticker_list = ['MSFT', 'GOOG', 'AAPL']

# variables
num_tickers = 0
ratios = {}
selected_ticker = []

# Set up color theme: white and grey
colors = {
    'background': '#F3F3F3',
    'boxBackground': '#FFFFFF',
    'text': '#000000'
}

def fig_blank():
    fig = go.Figure()
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig

# Dash app
app = Dash()
app.config.suppress_callback_exceptions=True

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
                #value=[ticker_list[0]],
                placeholder='Select stocks',
                multi=True,
                id='select-ticker'),
            html.Div(
                id='set-ratios',
                children=[]
            )], style={
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
            dcc.Graph(id='time-series-graph'),
            dcc.Graph(id='display-portfolios', figure=fig_blank()),
            #dcc.Graph(id='display-portfolios-for-one', figure=fig_blank()),
            #html.Div(id='display-portfolios-container', children=[dcc.Graph(id='display-portfolios')], style={'visible': 'None'}),
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
    global num_tickers
    num_tickers = len(value)
    global selected_ticker
    selected_ticker = value
    print(selected_ticker)
    return "This web application displays a graph of the historical prices of " + ", ".join(value)

def get_control_id(value):
    return 'ratio-{}'.format(value)


@app.callback(Output('set-ratios', 'children'), [Input('select-ticker', 'value')])
def dynamically_display_set_ratios(value):
    print("in graph")
    print("value： ")
    print(value)
    print(selected_ticker)
    # first clear 
    global ratios
    ratios = {}
    ret = []
    global num_tickers
    num_tickers = len(value)
    if num_tickers == 0:
        print("The portfolio has not been constructed yet.")
        ret.append(html.Label("The portfolio has not been constructed yet."))
        return html.Div(ret)   
    if num_tickers == 1:
        # do not need to set tickers for 1 ticker
        ret.append(html.Label("The portfolio has 100% of " + value[0]))
    for val in ticker_list:
        if val not in value or len(value) == 1:
            ret.append(html.Div([
                html.Br(),
                html.Label(val + ": "),
                dcc.Input(
                    id=get_control_id(val),
                    placeholder='please input ratios',
                    min=0,
                    max=100,
                    type='number'),
                html.Label('%')
            ], style={'display': 'none'})) 
            continue
        ret.append(html.Div([
            html.Br(),
            html.Label(val + ": "),
            dcc.Input(
                id=get_control_id(val),
                placeholder='please input ratios',
                min=0,
                max=100,
                type='number'),
            html.Label('%')
        ])) 
    ret.append(html.Div([
        html.Br(),
        html.Button('Construct', id='submit-ratios', n_clicks=0)
    ]))
    return html.Div(ret)


# update pie chart for one stock
@app.callback(
    Output('display-portfolios', 'figure'),
    [Input('submit-ratios', 'n_clicks'),
    Input('select-ticker', 'value')],
    [State('ratio-{}'.format(val), 'value') for val in ticker_list],
    prevent_initial_call=True
)
def draw_ratios(*args, **kwargs):
    # set ratios
    print(args)
    sum = 0
    if args[0] == 0:
        return fig_blank()
    if len(args[1]) == 0:
        return fig_blank()
    for i in range(2, len(args)):
        if args[i] != None:
            ratios[ticker_list[i - 1]] = args[i]
            sum += args[i]
    if len(selected_ticker) == 1:
        ratios[selected_ticker[0]] = 100
        sum += 100
    print(ratios)
    if sum != 100:
        for key in ratios:
            ratios[key] = 1 / len(ratios) * 100
    print(ratios)
    fig = go.Figure()
    labels = [key for key in ratios.keys()]
    vals = [val for val in ratios.values()]
    fig = px.pie(values=vals, names=labels)
    fig.update_layout(
        title_text="Portfolio Constructions of " + ", ".join(labels),
    )
    return fig
    

@app.callback(Output('time-series-graph', 'figure'), [Input('select-ticker', 'value')])
def draw_time_series_graph(value):
    # prepare data
    fig = go.Figure()
    if value == None:
        return fig_blank()
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