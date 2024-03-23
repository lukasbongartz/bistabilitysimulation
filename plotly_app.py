import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for Render.com

# Define the layout of the app
app.layout = html.Div([
    html.H1("Interactive Trigonometric Plot"),
    dcc.Dropdown(
        id='function-dropdown',
        options=[
            {'label': 'Sine', 'value': 'sin'},
            {'label': 'Cosine', 'value': 'cos'},
            {'label': 'Tangent', 'value': 'tan'}
        ],
        value='sin'  # Default value
    ),
    dcc.Graph(id='trig-plot')
])

# Define callback to update graph based on dropdown selection
@app.callback(
    Output('trig-plot', 'figure'),
    [Input('function-dropdown', 'value')]
)
def update_graph(selected_function):
    x_values = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    if selected_function == 'sin':
        y_values = np.sin(x_values)
    elif selected_function == 'cos':
        y_values = np.cos(x_values)
    elif selected_function == 'tan':
        y_values = np.tan(x_values)
    else:
        y_values = np.sin(x_values)  # Default to sine

    figure = {
        'data': [go.Scatter(x=x_values, y=y_values, mode='lines')],
        'layout': go.Layout(title=f'Plot of {selected_function}', xaxis_title='x', yaxis_title=f'{selected_function}(x)')
    }
    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
