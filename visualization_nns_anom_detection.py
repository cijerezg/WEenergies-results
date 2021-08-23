import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import numpy as np
import utils

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = np.load('results/data_nns.npy')
errs = np.load('results/errs_nns.npy')
errs = errs+1
errs = np.log(errs)


aa = np.arange(data.shape[0])

data = data[:,-errs.shape[1]:,:]

# Scatter points
errs_list = []
ranks = []
for i in range(errs.shape[2]):
    err = np.max(errs[:,:,i],1)
    ranks.append(np.argsort(np.argsort(err)))
    err_norm = utils.normalization_0_1(err) 
    errs_list.append(err_norm)

df_scat = pd.DataFrame({
    'ids': aa,
    'er1': errs_list[0],
    'er2': errs_list[1],
    'ranking1': ranks[0],
    'ranking2': ranks[1]
    })

# Predictions
time_p = np.arange(data.shape[1])
time_p = np.repeat(time_p[np.newaxis,:],data.shape[0], axis=0)
time_p = np.repeat(time_p[:,:,np.newaxis],data.shape[2], axis=2)

ids_pred = np.repeat(aa[:,np.newaxis],data.shape[1], axis=1)
ids_pred = np.repeat(ids_pred[:,:,np.newaxis], data.shape[2], axis=2)

cols = np.asarray(['NN 3','NN 5','Original'])
cols = np.repeat(cols[np.newaxis,:], data.shape[1], axis=0)
cols = np.repeat(cols[np.newaxis,:,:], data.shape[0], axis=0)

df_preds = pd.DataFrame({
    'ids': ids_pred.flatten(),
    'vals': data.flatten(),
    'time': time_p.flatten(),
    'colors': cols.flatten()
    })

# Errs
time_e = np.arange(errs.shape[1])
time_e = np.repeat(time_e[np.newaxis,:],errs.shape[0], axis=0)
time_e = np.repeat(time_e[:,:,np.newaxis],errs.shape[2], axis=2)

ids_e = np.repeat(aa[:,np.newaxis],errs.shape[1], axis=1)
ids_e = np.repeat(ids_e[:,:,np.newaxis], errs.shape[2], axis=2)

cols_e = np.asarray(['NN 3','NN 5'])
cols_e = np.repeat(cols_e[np.newaxis,:], errs.shape[1], axis=0)
cols_e = np.repeat(cols_e[np.newaxis,:,:], errs.shape[0], axis=0)

df_errs = pd.DataFrame({
    'ids': ids_e.flatten(),
    'vals': errs.flatten(),
    'time': time_e.flatten(),
    'colors': cols_e.flatten()
    })


app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='crossfilter-axis',
            options=[{'label':i,'value':i} for i in ['Errors','Rankings']],
            value='Errors'
        )], style = {'width':'99%', 'display': 'inline-blocks'}),
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 0}]}
        )
    ], style={'width':'99%', 'display':'inline-block', 'padding':'0.20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='x-time-series1')
    ], style={'display':'inline-block','width':'99%','height':'49%'})
])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-axis', 'value')])
def update_graph(axis_name):
    if axis_name == 'Errors':
        fig = px.scatter(df_scat, x='er1', y='er2',
                         hover_name = 'ids')
    else:
        fig = px.scatter(df_scat, x='ranking1',y='ranking2',
                         hover_name='ids')

    fig.update_layout(font= dict(size=20),
                      hovermode='closest')
    fig.update_traces(customdata= df_scat['ids'])

    return fig


def create_time_series(dff, title):
    fig = px.scatter(dff, x='time', y='vals', color='colors')
    fig.update_traces(marker=dict(size=1), mode='lines+markers')
    fig.update_layout(title=title, font=dict(size=20))
    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_x_timeseries(hoverData):
    id_m = hoverData['points'][0]['customdata']
    dff = df_preds[df_preds['ids']==id_m]
    return create_time_series(dff, 'KWH curves')


@app.callback(
    dash.dependencies.Output('x-time-series1', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_x_timeseries1(hoverData):
    id_m = hoverData['points'][0]['customdata']
    dff = df_errs[df_errs['ids']==id_m]
    return create_time_series(dff, 'Error curves')


if __name__ == '__main__':
    app.run_server(debug=True)
