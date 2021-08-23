import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Filename that contains KWH', nargs="?")
args = parser.parse_args()
path = 'results/'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = np.load(path+'data.npy')
SDs = np.load(path+args.filename)

aa = np.arange(data.shape[0])


#Scatter points
ranks = []
for i in range(SDs.shape[1]):
    ranks.append(np.argsort(np.argsort(SDs[:, i])))

df_scat = pd.DataFrame({
    'ids': aa,
    'Score 1': SDs[:,0],
    'Score 2': SDs[:,1],
    'Ranking 1': ranks[0],
    'Ranking 2': ranks[1]
    })

#Data
time = np.arange(data.shape[1])
time = np.repeat(time[np.newaxis, :], data.shape[0], axis=0)

ids_pred = np.repeat(aa[:, np.newaxis], data.shape[1], axis=1)

df_data = pd.DataFrame({
    'ids': ids_pred.flatten(),
    'vals': data.flatten(),
    'time': time.flatten()
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
    ], style={'display':'inline-block','width':'99%'})
])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-axis','value')])
def update_graph(axis_name):
    if axis_name == 'Errors':
        fig = px.scatter(df_scat, x='SD 7' , y='SD 14',
                         hover_name ='ids')
    else:
        fig = px.scatter(df_scat, x='Ranking SD 7', y='Ranking SD 14',
                         hover_name ='ids')

    fig.update_layout(font= dict(size=20),hovermode='closest')
    fig.update_traces(customdata = df_scat['ids'])
    
    return fig


def create_time_series(dff, title):
    fig = px.scatter(dff, x='time',y='vals')
    fig.update_traces(marker=dict(size=1), mode='lines+markers')
    fig.update_layout(title=title,font=dict(size=20))

    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_x_timeseries(hoverData):
    id_m = hoverData['points'][0]['customdata']
    dff = df_data[df_data['ids']==id_m]
    return create_time_series(dff, 'KWH curves')


if __name__ == '__main__':
    app.run_server(debug=True)