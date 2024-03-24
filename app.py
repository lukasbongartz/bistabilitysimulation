import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd




kB=1.380e-23
e=1.602e-19
psi_array = np.arange(0,1.01,0.01)

mud = 0
muu = 0

def H(psi, h_dd, h_uu, h_ud, mud, muu):
    return psi*mud + (1-psi)*muu + 0.5*(h_dd*psi**2 + h_uu*(1-psi)**2 + 2*h_ud*(1-psi)*psi)

def TS(psi, T):
    return -kB*(psi*np.log(psi) + (1-psi)*np.log(1-psi))*T/e*1000

def G(psi, T, h_dd, h_uu, h_ud, mud, muu):
    return H(psi, h_dd, h_uu, h_ud, mud, muu) - TS(psi, T)

def mu(psi, T, h_dd, h_uu, h_ud, mud, muu):
    return (np.diff(G(psi, T, h_dd, h_uu, h_ud, mud, muu))/np.diff(psi))

def lambda_value(h_dd, h_uu, h_ud, T, psi_lambda):
    value = (h_dd + h_uu - 2*h_ud)/(kB*T/e*1000) * psi_lambda*(psi_lambda-1)
    value_rn = np.round(value, 2)
    return value_rn

# Parameter ranges
T_range = np.linspace(200, 500, 1000)  # Temperature range
h_ud_range = np.linspace(-50, 250, 1000)  # hud range
h_dd = h_uu = 0  # Fixed for simplicity

lambda_vals = []
psi_min_vals = []

for T in T_range:
    for h_ud in h_ud_range:
        lambda_val = lambda_value(h_dd, h_uu, h_ud, T, 0.5)
        psi_range = np.linspace(0.01, 0.99, 100)
        G_vals = G(psi_range, T, h_dd, h_uu, h_ud, mud, muu)
        
        # Find minima of G (simplified approach)
        min_indices = np.argwhere(np.diff(np.sign(np.diff(G_vals))) > 0).flatten()
        for index in min_indices:
            lambda_vals.append(lambda_val)
            psi_min_vals.append(psi_range[index]) 

def main():
    st.set_page_config(page_title='Bistability', page_icon = "🧠", initial_sidebar_state = 'auto')
    st.sidebar.header('Parameters')



    st.session_state['second_mode'] = False
    st.session_state['h_dd'] = 0.0
    st.session_state['h_uu'] = 0.0
    st.session_state['h_ud'] = 0.0
    st.session_state['mud'] = 0.0
    st.session_state['muu'] = 0.0



    # Now create the sliders with the possibly updated default values from session_state
    h_dd = st.sidebar.slider(r'$h_{dd}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_dd'])
    h_uu = st.sidebar.slider(r'$h_{uu}\,(\mathrm{meV})$',-250.0, 250.0, st.session_state['h_uu'])
    h_ud = st.sidebar.slider(r'$h_{ud}\,(\mathrm{meV})$',-250.0, 250.0, st.session_state['h_ud'])
    #mud = st.sidebar.slider(r'$\mu^0_\mathrm{d}\,(\mathrm{meV}):$', -250.0, 250.0, st.session_state['mud'])
    #muu = st.sidebar.slider(r'$\mu^0_\mathrm{u}\,(\mathrm{meV}):$', -250.0, 250.0, st.session_state['muu'])
    T = st.sidebar.slider(r'$T\,(\mathrm{K})$', 200.0, 500.0, 300.0)  # Enable the temperature slider when not in second mode

    ## make in second mode
    #h_dd = st.sidebar.number_input(r'$h_{dd}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_dd'])
    #h_uu = st.sidebar.number_input(r'$h_{uu}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_uu'])
    #h_ud = st.sidebar.number_input(r'$h_{ud}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_ud'])
    #mud = st.sidebar.number_input(r'$\mu^0_\mathrm{d}\,(\mathrm{meV}):$', -250.0, 250.0, st.session_state['mud'])
    #muu = st.sidebar.number_input(r'$\mu^0_\mathrm{u}\,(\mathrm{meV}):$', -250.0, 250.0, st.session_state['muu'])
    #T = st.sidebar.number_input(r'$T\,(\mathrm{K})$', 200.0, 500.0, 300.0) 

    font = {'size' : 14} 
    plt.rc('font', **font)
    fig = plt.figure(figsize=(15, 8))  # Example: smaller figure size
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)

    gs = gridspec.GridSpec(2, 3, figure=fig)

    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    y_H = H(psi_array, h_dd, h_uu, h_ud, mud, muu)
    axs[0].plot(psi_array, y_H, linewidth=3, color = plt.cm.tab20b(0))
    axs[0].set_title(r'Enthalpy', fontsize=16)
    axs[0].set_xlabel(r'$\psi$', fontsize=14)
    axs[0].set_ylabel(r'$H_0 + H_\mathrm{mix}$ (meV)', fontsize=14)

    y_TS = TS(psi_array, T)
    axs[1].plot(psi_array, -y_TS, linewidth=3, color = plt.cm.tab20b(0))
    axs[1].set_title(r'Entropy', fontsize=16)
    axs[1].set_xlabel(r'$\psi$', fontsize=14)
    axs[1].set_ylabel(r'$-TS_\mathrm{mix}$ (meV)', fontsize=14)

    y_G = G(psi_array, T, h_dd, h_uu, h_ud, mud, muu)
    axs[2].plot(psi_array, y_G, linewidth=3, color = plt.cm.tab20b(0))
    axs[2].set_title(r'Gibbs Free Energy', fontsize=16)
    axs[2].set_xlabel(r'$\psi$', fontsize=14)
    axs[2].set_ylabel(r'$G$ (meV)', fontsize=14)


    y_mu = mu(psi_array, T, h_dd, h_uu, h_ud, mud, muu)
    slope = np.diff(y_mu) / np.diff(psi_array[:-1])  
    slope = np.append(slope, 0)  

    sign_change_indices = np.where(np.diff(np.sign(slope)))[0]

    segments_psi = np.split(psi_array[:-1], sign_change_indices+1)
    segments_mu = np.split(y_mu, sign_change_indices+1)

    for seg_psi, seg_mu in zip(segments_psi, segments_mu):
        if (np.diff(seg_mu) / np.diff(seg_psi) < 0).any(): 
            axs[3].plot(seg_psi, seg_mu, '--', linewidth=3, color = plt.cm.tab20b(0))
            axs[4].plot(seg_mu, 1 - seg_psi, '--', linewidth=3, color = plt.cm.tab20b(0))
        else:
            axs[3].plot(seg_psi, seg_mu, linewidth=3, color = plt.cm.tab20b(0))
            axs[4].plot(seg_mu, 1 - seg_psi, linewidth=3, color = plt.cm.tab20b(0))



    
    psi_lambda = 0.5
    lambda_val = lambda_value(h_dd, h_uu, h_ud, T, psi_lambda)
    #textstr = f'Degree of Bistability: \n$\lambda = {lambda_val}$'    
    #axs[3].text(0.5, 0.5, textstr, transform=axs[3].transAxes, fontsize=16,
    #            verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', 
    #            facecolor='wheat', alpha=0.5))
    #axs[3].axis('off')    


    axs[3].set_title('Chem. Potential', fontsize=16)
    axs[3].set_xlabel(r'$\psi$', fontsize=14)
    axs[3].set_ylabel(r'$\mu$ (meV)', fontsize=14)

    axs[4].set_title("Transfer Curve", fontsize=16)
    axs[4].set_xlabel(r'$V_\mathrm{GS}$ (mV)', fontsize=14)
    axs[4].set_ylabel(r'$-I_\mathrm{D}$ (norm.)', fontsize=14)


#    axs[5].axvline(x=lambda_val, color=plt.cm.tab20c(4), linestyle='-', lw = 3)
#    axs[5].scatter(lambda_vals, psi_min_vals, s=5, color = plt.cm.tab20b(0))
#    axs[5].text(lambda_val, -axs[5].get_ylim()[1]*0.35, f'$\lambda = {lambda_val}$', color='k', ha='center', va = 'center', fontsize=16,
#                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#    axs[5].set_xlabel(r'$\lambda$')
#    axs[5].set_ylabel(r'$\psi_0$')
#    axs[5].set_title('Bifurcation Diagram')

    # Read bifurcation simulation data from a CSV file
    bifurcation_data = pd.read_csv('BistabilitySimulation/bifurcation_data.csv')  # Update the path as necessary
    lambda_vals = bifurcation_data['lambda'].values
    psi_min_vals = bifurcation_data['psi_min'].values

    #axs[5].axvline(x=lambda_val, color=plt.cm.tab20c(4), linestyle='-', lw=3)
    axs[5].scatter(lambda_vals, psi_min_vals, s=5, color=plt.cm.tab20b(0))
    axs[5].set_xlabel(r'$\lambda$')
    axs[5].set_ylabel(r'$\psi_0$')
    axs[5].set_title('Bifurcation Diagram')




    # Format all axes (even if empty)
    for i in range(6):
        if axs[i].has_data():
            axs[i].set_aspect(1./axs[i].get_data_ratio())
        for axis in ['top','bottom','left','right']:
            axs[i].spines[axis].set_linewidth(1.5)
        axs[i].tick_params(axis = 'both', width = 1.5, length = 5, grid_linewidth = 1.5, direction = 'in')
        axs[i].grid(True,'major',alpha=0.2)



   
    st.sidebar.markdown("Lukas Bongartz, 2024")

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
