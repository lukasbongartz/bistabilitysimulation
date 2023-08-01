import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

    st.sidebar.markdown("Lukas Bongartz, 2023")

    st.sidebar.header('Parameters')

    h1 = st.sidebar.slider(r'$h_1\(k_\mathrm{B}\300\mathrm{K})$', 0.0, 5.0, 0.0)
    h2 = st.sidebar.slider(r'$h_2\(k_\mathrm{B}\300\mathrm{K})$', 0.0, 5.0, 0.0)
    h3 = st.sidebar.slider(r'$h_3\(k_\mathrm{B}\300\mathrm{K})$', 0.0, 5.0, 0.0)
    T = st.sidebar.slider('$T\(\mathrm{K})$', 200.0, 400.0, 300.0)

    font = {'size' : 14} 
    plt.rc('font', **font)
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    y_H = H(h1, h2, h3)
    axs[0].plot(phi, y_H, linewidth=3, color = plt.cm.tab20b(0))
    axs[0].set_title(r'Enthalpy', fontsize=16)
    axs[0].set_xlabel(r'$\phi$', fontsize=14)
    axs[0].set_ylabel(r'$H_\mathrm{mix}$ (meV)', fontsize=14)

    y_TS = TS(T)
    axs[1].plot(phi, y_TS, linewidth=3, color = plt.cm.tab20b(0))
    axs[1].set_title(r'Entropy', fontsize=16)
    axs[1].set_xlabel(r'$\phi$', fontsize=14)
    axs[1].set_ylabel(r'$TS_\mathrm{mix}$ (meV)', fontsize=14)

    y_G = G(T, h1, h2, h3)
    axs[2].plot(phi, y_G, linewidth=3, color = plt.cm.tab20b(0))
    axs[2].set_title(r'Gibbs Free Energy', fontsize=16)
    axs[2].set_xlabel('phi', fontsize=14)
    axs[2].set_ylabel(r'$G$ (meV)', fontsize=14)


    y_mu = mu(T, h1, h2, h3)
    slope = np.diff(y_mu) / np.diff(phi[:-1])  # calculate slope of mu vs phi
    slope = np.append(slope, 0)  # append a 0 at the end to match the shape of phi and y_mu

    # Calculate the indices where the slope changes sign
    sign_change_indices = np.where(np.diff(np.sign(slope)))[0]

    # Split the indices based on the above condition
    segments_phi = np.split(phi[:-1], sign_change_indices+1)
    segments_mu = np.split(y_mu, sign_change_indices+1)

    for seg_phi, seg_mu in zip(segments_phi, segments_mu):
        if (np.diff(seg_mu) / np.diff(seg_phi) < 0).any():  # if any slope is negative
            axs[4].plot(seg_phi, seg_mu, '--', linewidth=3, color = plt.cm.tab20b(0))
            axs[5].plot(seg_mu, 1 - seg_phi, '--', linewidth=3, color = plt.cm.tab20b(0))
        else:
            axs[4].plot(seg_phi, seg_mu, linewidth=3, color = plt.cm.tab20b(0))
            axs[5].plot(seg_mu, 1 - seg_phi, linewidth=3, color = plt.cm.tab20b(0))



    axs[4].set_title('Chem. Potential', fontsize=16)
    axs[4].set_xlabel(r'$\phi$', fontsize=14)
    axs[4].set_ylabel(r'${\partial G}/{\partial \phi} = \mu$ (meV)', fontsize=14)

    axs[5].set_title("Theor. Transfer Curve", fontsize=16)
    axs[5].set_xlabel(r'$V_\mathrm{GS}$ (mV)', fontsize=14)
    axs[5].set_ylabel(r'$-I_\mathrm{D}$ (norm.)', fontsize=14)

    # Remove the sixth plot
    fig.delaxes(axs[3])

    # Format all axes (even if empty)
    for i in range(6):
        if axs[i].has_data():
            axs[i].set_aspect(1./axs[i].get_data_ratio())
        for axis in ['top','bottom','left','right']:
            axs[i].spines[axis].set_linewidth(1.5)
        axs[i].tick_params(axis = 'both', width = 1.5, length = 5, grid_linewidth = 1.5, direction = 'in')
        axs[i].grid(True,'major',alpha=0.2)

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
