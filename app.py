import dash
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

from dash import Dash, html, dcc, ctx, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import yfinance as yf

import json
# Set available tickers
ticker_list = ['MSFT', 'GOOG', 'AAPL', 'CRM', 'INTC', 'NVDA', 'QCOM', 'INTU', 'META', 'AMZN', 'NFLX', 'TSLA']
ticker_list.sort()
yf_ticker = [yf.Ticker(val) for val in ticker_list]
all_tickers_infos = [ticker.history(period='max') for ticker in yf_ticker]
ticker_to_info = dict(zip(ticker_list, all_tickers_infos))
for ticker in ticker_list:
    ticker_to_info[ticker] = ticker_to_info[ticker].reset_index()

# golbal variables
num_tickers = 0
ratios = {}
selected_ticker = []
memo = {}

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

def get_control_id(value):
    return 'ratio-{}'.format(value)

def build_ratio_input_components():
    ret = []
    print("build")
    global ticker_list
    for val in ticker_list:
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
    print(ret)
    return ret


# Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{'name': 'viewport', 'content': 'width=100%, height=100%, initial_scale=1'}])
app.config.suppress_callback_exceptions=True

app.layout = html.Div([
    dcc.Store(id='selected-tickers'),
    dcc.Store(id='status-check'),
    # header
    html.Div([
        html.Center(html.H1('FRE6191 Project Part III', style={'fontSize': '2.5vh', 'margin': '1.4vh'})),
        html.Center(
            html.H5('Team 6: Layla Li (xl4432), Helen Zhang (jz5582), Sofia Guo (fg2283), Rita Wang (rw3167)', 
            style={'fontSize': '1.5vh', 'margin': '1.2vh'})),
    ], style={
        'width': '100%',
        'backgroundColor': colors['boxBackground'],
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
                searchable=True,
                id='select-ticker'),
            html.Div(
                id='set-ratios',
                children=build_ratio_input_components()
            ),
            html.Div([
                    html.Br(),
                    html.Center(dbc.Button('Construct', color='primary', id='submit-ratios', n_clicks=0)),
                    html.Br(),
                    html.Center(dbc.Alert(children=[], id="warning-msg", color="danger", style={'display':'none', 'margin': '1vh'}))
                ],
                id='submit-ratios-parent',
                style={'display':'none'}
            ),
            html.Div([
                    html.Br(),
                    html.Label('Run Monte Carlo Optimization To Obtain Optimal Portfolio: '),
                    html.Br(),
                    dbc.Input(
                            id='monte-caro-iters-nums',
                            placeholder='please number of iterations',
                            min=1,
                            max=100,
                            type='number'
                        ),
                    html.Br(),
                    html.Center(dbc.Button('Obtain Optimal Portfolio', id='run-monte-carlo', color='primary', n_clicks=0)),
                    html.Br(),
                    html.Center(dbc.Alert(children=[], id="warning-msg-mc", color="danger", style={'display':'none', 'margin': '1vh'}))
                ],
                id='run-monte-carlo-parent',
                style={'display':'none'}
            )
        ], style={
                    'display': 'inline-block', 
                    'width': '20%',
                    'padding': '2rem',
                    'marginRight': '1.5%',
                    'marginLeft': '1.5%',
                    'borderRadius': '10px',
                    'boxShadow': '#EBEBEB 4px 4px 2px',
                    'backgroundColor': colors['boxBackground']
        }),
        # dashboard area
        html.Div([
            html.Div(
                id='intro'
            ),
            html.Br(),
            html.Div([
                html.Div([dcc.Graph(
                                id='time-series-graph', 
                                style={
                                    'width': '60%',
                                    'height': '100%',
                                    'display': 'inline-block'
                                }),
                        html.Div( 
                            id='stock-dash-table',
                            style={
                                    #'position': 'flex',
                                    'top': '14%',
                                    'left': '70%',
                                    'overflowX': 'auto',
                                    'overflowY': 'hidden',
                                    'width': '40%',
                                    'height': '100%',
                                    'display': 'inline-block',
                                    'margin': 'auto',
                                    'padding': '4%',
                            })], 
                style={
                    'display': 'inline-block',
                    'margin': 'auto',
                    'width': '100%',
                    'height': '50%'
                }),
                html.Div([dcc.Graph(
                                id='display-portfolios', 
                                figure=fig_blank(), 
                                style={
                                    'width': '40%',
                                    'height': '100%',
                                    'display': 'inline-block'
                                }),
                            
                            dcc.Graph(
                                id='monte-carlo-result',
                                figure=fig_blank(),
                                style={
                                    'width': '60%',
                                    'height': '100%',
                                    'display': 'inline-block'
                                })
                        ],
                id='display-portfolio-parent',
                style={
                    'display': 'inline-block',
                    'width': '100%',
                    'height': '50%'
                })
            ], style={
                'display': 'inline-block',
                'width': '100%',
                'height': '95%',
            }, id='plot-container')
            ], style={
                'display': 'inline-block',
                'width': '74%', 
                'bottom': '3%',
                'padding': '2rem',
                'marginRight': '1.5%',
                'marginLeft': '1.5%',
                'borderRadius': '10px',
                'boxShadow': '#EBEBEB 5px 5px 3px',
                'backgroundColor': colors['boxBackground']
        }),
    ], style={
        'display':'flex',
        'flexDirection': 'row',
        'height': '85%',
        'width': '100%',
        'margin': '0',
        'position': 'absolute',
        'top': '12%',
        'bottom' :'3%'
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
        'backgroundColor': colors['background'],
        'color': colors['text'],
        'height': '100vh',
        'width': '100vw',
        'top': '0',
        'bottom': '0',
        'left' : '0',
        'right': '0',
        'position': 'fixed',
        'padding': '0',
        'margin': '0',
        'overflowX': 'hidden',
        'overflowY': 'hidden'
})


# Get input stocks
@app.callback(
        Output('selected-tickers', 'data'),
        Output('intro', 'children'), 
        [Input('select-ticker', 'value')]
)
def update_intro(value):
    print(value)
    intro_sentence = "This web application helps you to construct an optimal stock portfolio."
    if value == None:
        return [], intro_sentence
    global num_tickers
    num_tickers = len(value)
    global selected_ticker
    selected_ticker = value
    print(selected_ticker)
    return selected_ticker, intro_sentence

# disable if select options if larger than 8
@app.callback(
        
        Output('select-ticker', 'options'),  
        [Input('select-ticker', 'value')])
def set_options(value):
    global ticker_list
    if value == None or len(value) < 8:
        return ([{'label': i, 'value': i, 'disabled': False} for i in ticker_list])
    ret = [{'label': i, 'value': i, 'disabled': True} for i in ticker_list]
    return (ret)
    
# Draw time series plot
@app.callback(Output('time-series-graph', 'figure'), [Input('select-ticker', 'value')])
def draw_time_series_graph(value):
    # prepare data
    fig = go.Figure()
    if value == None:
        return fig_blank()
    global ticker_to_info
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

# draw dash table
@app.callback(Output('stock-dash-table', 'children'), [Input('select-ticker', 'value')])
def draw_stock_dash_table(value):
    # prepare data
    if value == None:
        return []
    global ticker_to_info
    df_stocks = pd.DataFrame(columns=['Symbol', 'Last Price', 'Change', '%Change (%)'])
    for v in value:
        #ticker_to_info[v] = ticker_to_info[v].reset_index()
        cur_price = ticker_to_info[v]['Close'].iloc[-1]
        prev_price = ticker_to_info[v]['Close'].iloc[-2]
        diff_price = round(cur_price - prev_price, 2)
        pct_change = (cur_price - prev_price) / prev_price
        df_tmp = {'Symbol' : v, 'Last Price': round(cur_price, 2), 'Change': diff_price, '%Change (%)': round(pct_change*100, 2)}
        df_stocks.loc[len(df_stocks.index)] = (df_tmp)
    return [
        html.Label('Stocks Statistics:'),
        dash_table.DataTable(
        data=df_stocks.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df_stocks.columns],
        editable=True,
        style_data_conditional=
            [
                {
                'if': {
                    'column_id': 'Change',
                    'filter_query': '{Change} < 0'
                    
                },
                'color': 'tomato'
                },
                {
                'if': {
                    'column_id': 'Change',
                    'filter_query': '{Change} >= 0',
                },
                'color': 'green'
                },
                {
                
                'if': {
                    'column_id': '%Change (%)',
                    'filter_query': '{%Change (%)} < 0',
                    
                    },
                'color': 'tomato'
                
                },
                {
                'if': {
                    'column_id': '%Change (%)',
                    'filter_query': '{%Change (%)} >= 0',
                    
                },
                'color': 'green'
                },
            ],
        style_cell_conditional=
            [
                {
                    'if':{'column_id': d},
                    'textAlign': 'left'
                } for d in ['Symbol']
            ],     
        
        style_header={
            'backgroundColor': 'rgb(239, 239, 239)',
            'color': 'black',
            'minWidth': '150px', 
            'width': '150px', 
            'maxWidth': '150px',
            'wightSpace': 'normal',
            'fontWeight': 'bold'
        },
        style_data={
            'backgroundColor': 'white',
            'color': 'black',
            'wightSpace': 'normal',
            'height': 'auto',
            'minWidth': '150px', 
            'width': '150px', 
            'maxWidth': '150px'
        },
        fill_width=False
        #style_as_list_view=True,
    )]

# Handle display of input components
def get_control_id(value):
    return 'ratio-{}'.format(value)

@app.callback(
        Output('submit-ratios-parent', 'style'),
        Output('run-monte-carlo-parent', 'style'),
        Output('set-ratios', 'children'), 
        [Input('select-ticker', 'value')])
def dynamically_display_input_components(value):
    print("in graph")
    print("value： ")
    print(value)
    print(selected_ticker)
    # first clear
    global ratios
    ratios = {}
    ret = []
    if value == None:
        print("The portfolio has not been constructed yet.")
        ret.append(html.Label("The portfolio has not been constructed yet."))
    global num_tickers
    num_tickers = -1
    if value != None:
        num_tickers = len(value) 
    if num_tickers == 0:
        print("The portfolio has not been constructed yet.")
        ret.append(html.Label("The portfolio has not been constructed yet."))
    elif num_tickers == 1:
        # do not need to set tickers for 1 ticker
        ret.append(html.Label("The portfolio has 100% of " + value[0]))
    for val in ticker_list:
        if value == None or num_tickers == 0 or val not in value or len(value) == 1:
            ret.append(html.Div([
                html.Br(),
                #html.Label(val + ": "),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(val),
                        dbc.Input(
                            id=get_control_id(val),
                            placeholder='please input ratios',
                            min=0,
                            max=100,
                            type='number'
                        ),
                        dbc.InputGroupText("%")
                    ]
                )
            ], style={'display': 'none'})) 
            continue
        ret.append(html.Div([
            html.Br(),
            dbc.InputGroup(
                    [
                        dbc.InputGroupText(val),
                        dbc.Input(
                            id=get_control_id(val),
                            placeholder='please input ratios',
                            min=0,
                            max=100,
                            type='number'
                        ),
                        dbc.InputGroupText("%")
                    ]
                )
        ])) 
    if value == None or num_tickers == 0:
        return {'display':'none'}, {'display':'none'}, html.Div(ret)
    if num_tickers == 1:
        return {'display':'block'}, {'display':'none'}, html.Div(ret)
    return {'display':'block'}, {'display':'block'}, html.Div(ret)


# memoization
def memoize(f):
    def wrapper(*args):
        global m
        print(args)
        print(memo)
        # hash map
        # check if the arguments have been called before, if not, store them to the hash map
        if args not in memo.keys():
            memo[args] = f(*args)
            print('args not in', args)
            #print('results are: ', memo[args])
        print('args exist : ', args )
        return memo[args]
    return wrapper

# Monte Carlo Simulation
@memoize
def get_stocks_returns(tickers, begin_date, end_date):
    df = pd.DataFrame([])
    for ticker in tickers:
        stock_price = yf.download(ticker, begin_date, end_date).Close
        df[ticker] = stock_price
    return df

def get_ret_vol_sharpe(weights, log_ret):
    weights = np.array(weights)
    ret = np.sum((log_ret.mean() * weights) * 252)
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sharpe = ret / vol
    return np.array([ret, vol, sharpe])

def check_sum(weights):
    return np.sum(weights) - 1

def minimize_vol(weights, log_ret):
    return get_ret_vol_sharpe(weights, log_ret)[1]

@memoize
def run_monte_carlo_and_efficient_frontier(num_iter, selected_ticker):
    # get stock returns
    begin_date = "2017-01-01"
    end_date = "2022-12-31"
    df = get_stocks_returns(selected_ticker, begin_date, end_date)
    # log ret
    df = np.log(df / df.shift(1))
    df = df[1:]
    # run monte carlo simulations
    n = len(selected_ticker)
    weights = np.array(np.random.random(n))
    ret = np.zeros(num_iter)
    vol = np.zeros(num_iter)
    sharpe = np.zeros(num_iter)
    all_weights = []
    for i in range(num_iter):
        weights = np.array(np.random.random(n))
        weights = weights / np.sum(weights)
        all_weights.append(weights)
        ret[i] = np.sum((df.mean() * weights) * 252)
        vol[i] = np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))
        sharpe[i] = ret[i] / vol[i]

    # run efficient frontier
    N_PTS = 100
    frontier_rets = np.linspace(ret.min(), ret.max(), N_PTS)
    frontier_vols = []
    initial_guess = np.array(np.random.random(len(selected_ticker)))
    initial_guess = initial_guess / np.sum(initial_guess)
    bounds = []
    for i in range(len(selected_ticker)):
        bounds.append((0, 1))
    bounds = tuple(bounds)
    for ret_possible in frontier_rets:
        constraints = ({'type': 'eq', 'fun': check_sum},
                        {'type': 'eq', 'fun': lambda w: get_ret_vol_sharpe(w, df)[0] - ret_possible})
        minimized_res = minimize(minimize_vol, initial_guess, (df,), method='SLSQP', bounds=bounds, constraints=constraints)
        frontier_vols.append(minimized_res['fun'])
    
    return ret, vol, sharpe, all_weights, frontier_rets, frontier_vols

# draw ratios (pie chart) and monte carlo results with efficient frontier
@app.callback(
    [
        Output('warning-msg', 'children'),
        Output('warning-msg', 'style'),
        Output('warning-msg-mc', 'children'),
        Output('warning-msg-mc', 'style'),
        Output('display-portfolios', 'figure'),
        Output('monte-carlo-result', 'figure'),
        [Output('ratio-{}'.format(val), 'value') for val in ticker_list]
    ],
    Input('submit-ratios', 'n_clicks'),
    Input('run-monte-carlo', 'n_clicks'),
    [Input('select-ticker', 'value')],
    State('monte-caro-iters-nums', 'value'),
    [State('ratio-{}'.format(val), 'value') for val in ticker_list],
    
    prevent_initial_call=True
)
def draw_ratios(*args, **kwargs):
    # set ratios
    print('args')
    print(args)
    prop_id = ctx.triggered[0]['prop_id']
    prop_event_val = ctx.triggered[0]['value']
    sum = 0
    global ratios
    ratios = {}
    input_vals = []
    global selected_ticker
    fig_mc = go.Figure()
    # used triggered event to regulate behaviors
    if prop_id == 'select-ticker.value':
        print(prop_id)
        return [], {'display':'none'}, [], {'display':'none'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
    elif prop_id == 'submit-ratios.n_clicks':
        if prop_event_val == 0:
            return [], {'display':'none'}, [], {'display':'none'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
        else:
                for i in range(4, len(args)):
                    input_vals.append(args[i])
                    print(args[i])
                    if args[i] != None:
                        if args[i] < 0:
                            ratios = {}
                            return  "There's at least one selected stock receive invalid values(negative values), cannot calculate portfolio", {'display':'block'}, [], {'display':'none'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
                        ratios[ticker_list[i - 4]] = args[i]
                        sum += int(args[i])
                    else:
                        if ticker_list[i - 4] in selected_ticker and len(selected_ticker) != 1:
                            ratios = {}
                            return  "There's at least one selected stock not receive ratio, cannot calculate portfolio", {'display':'block'}, [], {'display':'none'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
                if len(selected_ticker) == 1:
                    idx = ticker_list.index(selected_ticker[0])
                    input_vals[idx] = 100
                    ratios[selected_ticker[0]] = 100
                    sum = 100
                print(ratios)
                if sum > 100:
                    ratios = {}
                    return  "Total ratios exceed 100! Please reinput", {'display':'block'}, [], {'display':'none'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
                if sum < 100:
                    ratios = {}
                    return "Total ratios are less than 100! Please reinput", {'display':'block'}, [], {'display':'none'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
    else:
        # run monte carlo
        for i in range(4, len(args)):
            input_vals.append(args[i])
        # can be set for users
        #begin_date = "2017-01-01"
        #end_date = "2022-12-31"
        #df = get_stocks_returns(selected_ticker, begin_date, end_date)
        #log_ret = np.log(df / df.shift(1))
        num_iters = args[3]
        if num_iters == None or num_iters <= 0 or num_iters > 100:
            return [], {'display':'none'}, "Invalid Number of Iterations!", {'display':'block'}, fig_blank(), fig_blank(), [None for i in range(len(ticker_list))]
        selected_ticker.sort()
        selected_ticker_tuple = tuple(selected_ticker)
        # use memoization!
        ret, vol, sharpe, weights, frontier_rets, frontier_vols = run_monte_carlo_and_efficient_frontier(num_iters, selected_ticker_tuple)
        res = pd.DataFrame([])
        res['ret'] = ret
        res['vol'] = vol
        res['sharpe'] = sharpe
        best_sharpe = sharpe.max()
        idx = sharpe.argmax()
        # draw scatter plot
        fig_mc.add_trace(go.Scatter(x=res['vol'], y=res['ret'], mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=res['sharpe'],
                                        colorscale='sunsetdark',
                                        showscale=True
                                    ),
                                    name='Sharpe Ratios'))
        # draw efficient froniter
        fig_mc.add_trace(go.Scatter(x=frontier_vols, y=frontier_rets, name='Efficient Frontier', mode='lines', line=dict(color="#4c00b0")))
        #fig_mc.update_layout(showlegend=False)
        # draw best sharpe ratio
        fig_mc.add_trace(go.Scatter(x=[res['vol'][idx]], y=[res['ret'][idx]], mode='markers+text',
                                    marker=dict(
                                        size=18,
                                        color=[best_sharpe],
                                        symbol='star',
                                    ),
                                    name=f'<b>best Sharpe Ratio</b>',
                                    text=f'<b>best Sharpe Ratio</b>',
                                    textposition='top center',
                                    textfont=dict(
                                        size=18,
                                        color='black',
                                        family='Times New Roman'
                                    )))
        fig_mc.update_layout(
                            title_text="Volatility vs Sharpe Ratio of " + str(len(selected_ticker)) + " differents portfolio weights using MC simulation over " 
                                    + str(num_iters) + " iterations",
                            xaxis_title="Volatility",
                            yaxis_title="Return",
                            legend=dict(
                                orientation="h",
                                yanchor='bottom',
                                y=1.02,
                                xanchor='right',
                                x=1
                            ))
        for i in range(len(selected_ticker)):
            ratios[selected_ticker[i]] = weights[idx][i]

    # check if the user indeed input something or not?

    
    print(ratios)
    fig = go.Figure()
    labels = [key for key in ratios.keys()]
    vals = [val for val in ratios.values()]
    fig = px.pie(values=vals, names=labels)
    fig.update_layout(
        title_text="Portfolio Constructions of " + ", ".join(labels),
    )
    print(input_vals)
    if prop_id == 'run-monte-carlo.n_clicks':
        #print(res)
        return [], {'display':'none'}, [], {'display':'none'}, fig, fig_mc, input_vals
    return [], {'display':'none'}, [], {'display':'none'}, fig, fig_blank(), input_vals



if __name__ == "__main__":
    app.run_server(debug=True)