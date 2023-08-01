import streamlit as st
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go

kB=1.380e-23
e=1.602e-19
phi = np.arange(0,1.001,0.001)

def H(h1, h2, h3):
    return kB*300*(h1*phi**2 + h2*(1-phi)**2 + h3*(1-phi)*phi)/e*1000

def TS(T):
    return -kB*(phi*np.log(phi) + (1-phi)*np.log(1-phi))*T/e*1000

def G(T, h1, h2, h3):
    return H(h1, h2, h3) - TS(T)

def mu(T, h1, h2, h3):
    return np.diff(G(T, h1, h2, h3)) / np.diff(phi)

def main():
    st.title('Interactive Equation Visualization')

    st.sidebar.header('Equation Parameters')

    h1 = st.sidebar.slider('h1', 0.0, 5.0, 0.0)
    h2 = st.sidebar.slider('h2', 0.0, 5.0, 0.0)
    h3 = st.sidebar.slider('h3', 0.0, 5.0, 0.0)
    T = st.sidebar.slider('T', 200.0, 400.0, 300.0)

    fig = sp.make_subplots(rows=2, cols=3, subplot_titles=("H(h1, h2, h3)", "TS(T)", "G(T, h1, h2, h3)", "mu(T, h1, h2, h3)", "1 - phi"))

    y_H = H(h1, h2, h3)
    fig.add_trace(go.Scatter(x=phi, y=y_H, mode='lines', line=dict(color='blue')), row=1, col=1)

    y_TS = TS(T)
    fig.add_trace(go.Scatter(x=phi, y=y_TS, mode='lines', line=dict(color='blue')), row=1, col=2)

    y_G = G(T, h1, h2, h3)
    fig.add_trace(go.Scatter(x=phi, y=y_G, mode='lines', line=dict(color='blue')), row=1, col=3)

    y_mu = mu(T, h1, h2, h3)
    fig.add_trace(go.Scatter(x=phi[:-1], y=y_mu, mode='lines', line=dict(color='blue')), row=2, col=1)

    fig.add_trace(go.Scatter(x=y_mu, y=1 - phi[:-1], mode='lines', line=dict(color='blue')), row=2, col=2)

    # Update layout properties
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="white",
        plot_bgcolor='white',
        showlegend=False
    )

    # Update xaxis properties
    fig.update_xaxes(linecolor='black', linewidth=2, mirror=True, title_standoff=25)

    # Update yaxis properties
    fig.update_yaxes(linecolor='black', linewidth=2, mirror=True, scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
