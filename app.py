import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
    #st.title('Interactive Equation Visualization')

    st.sidebar.header('Equation Parameters')

    h1 = st.sidebar.slider('h1', 0.0, 5.0, 0.0)
    h2 = st.sidebar.slider('h2', 0.0, 5.0, 0.0)
    h3 = st.sidebar.slider('h3', 0.0, 5.0, 0.0)
    T = st.sidebar.slider('T', 200.0, 400.0, 300.0)

    font = {'size' : 14} 
    plt.rc('font', **font)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    y_H = H(h1, h2, h3)
    axs[0, 0].plot(phi, y_H, linewidth=3, color = plt.cm.tab20b(0))
    axs[0, 0].set_title(r'Enthalpy', fontsize=16)
    axs[0, 0].set_xlabel(r'$\phi$', fontsize=14)
    axs[0, 0].set_ylabel(r'$H_\mathrm{mix}$ (meV)', fontsize=14)

    y_TS = TS(T)
    axs[0, 1].plot(phi, y_TS, linewidth=3, color = plt.cm.tab20b(0))
    axs[0, 1].set_title(r'Entropy', fontsize=16)
    axs[0, 1].set_xlabel(r'$\phi$', fontsize=14)
    axs[0, 1].set_ylabel(r'$TS_\mathrm{mix}$ (meV)', fontsize=14)

    y_G = G(T, h1, h2, h3)
    axs[0, 2].plot(phi, y_G, linewidth=3, color = plt.cm.tab20b(0))
    axs[0, 2].set_title(r'Gibbs Free Energy', fontsize=16)
    axs[0, 2].set_xlabel('phi', fontsize=14)
    axs[0, 2].set_ylabel(r'$G$ (meV)', fontsize=14)

    y_mu = mu(T, h1, h2, h3)
    axs[1, 0].plot(phi[:-1], y_mu, linewidth=3, color = plt.cm.tab20b(0))
    axs[1, 0].set_title('Chem. Potential', fontsize=16)
    axs[1, 0].set_xlabel(r'$\phi$', fontsize=14)
    axs[1, 0].set_ylabel(r'$\mu$ (meV)', fontsize=14)

    axs[1, 1].plot(y_mu, 1 - phi[:-1], linewidth=3, color = plt.cm.tab20b(0)) # adjusted for the number of points after diff
    axs[1, 1].set_title("Theor. Transfer Curve", fontsize=16)
    axs[1, 1].set_xlabel(r'$V_\mathrm{GS}$ (mV)', fontsize=14)
    axs[1, 1].set_ylabel(r'$-I_\mathrm{D}$ (norm.)', fontsize=14)

    # Remove the sixth plot
    fig.delaxes(axs[1,2])

    # Format all axes (even if empty)
    for i in range(2):
        for j in range(3):
            if axs[i, j].has_data():
                axs[i, j].set_aspect(1./axs[i, j].get_data_ratio())
            for axis in ['top','bottom','left','right']:
                axs[i, j].spines[axis].set_linewidth(1.5)
            axs[i, j].tick_params(axis = 'both', width = 1.5, length = 5, grid_linewidth = 1.5, direction = 'in')
            axs[i, j].grid(True,'major',alpha=0.2)

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
