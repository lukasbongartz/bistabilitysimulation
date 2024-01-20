import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd



#V_ex = np.array([-1500., -1450., -1400., -1350., -1300., -1250., -1200., -1150., -1100., -1050.,
#     -1000., -950., -900., -850., -800., -750., -700., -650., -600., -550., -500.,
#     -450., -400., -350., -300., -250., -200., -150., -100., -50., 0., 50., 100.,
#     150., 200., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750.,
#     800., 850., 900., 950., 1000., 1050., 1100., 1150., 1200., 1250., 1300., 1350.,
#     1400., 1450., 1500., 1500., 1450., 1400., 1350., 1300., 1250., 1200., 1150.,
#     1100., 1050., 1000., 950., 900., 850., 800., 750., 700., 650., 600., 550., 500.,
#     450., 400., 350., 300., 250., 200., 150., 100., 50., 0., -50., -100., -150., -200.,
#     -250., -300., -350., -400., -450., -500., -550., -600., -650., -700., -750., -800.,
#     -850., -900., -950., -1000., -1050., -1100., -1150., -1200., -1250., -1300., -1350.,
#     -1400., -1450., -1500.])
#
#I_ex = np.array([9.84324987e-01, 9.91697477e-01, 9.95895805e-01, 9.98411036e-01,
# 9.99691244e-01, 1.00000000e+00, 9.99634764e-01, 9.98554118e-01,
# 9.96995277e-01, 9.94724791e-01, 9.92239682e-01, 9.89046694e-01,
# 9.85300204e-01, 9.81154590e-01, 9.75822902e-01, 9.69700497e-01,
# 9.63314520e-01, 9.56085112e-01, 9.47236618e-01, 9.38101960e-01,
# 9.27039460e-01, 9.12757614e-01, 8.97907205e-01, 8.79773440e-01,
# 8.60946857e-01, 8.34732723e-01, 8.05645652e-01, 7.71042392e-01,
# 7.32372590e-01, 6.86872504e-01, 6.37061131e-01, 5.77230249e-01,
# 5.12402793e-01, 4.49947486e-01, 3.81358479e-01, 3.10117559e-01,
# 2.57860235e-01, 2.01910266e-01, 1.56066783e-01, 1.12767525e-01,
# 8.03835424e-02, 5.44634792e-02, 3.41971897e-02, 2.19377706e-02,
# 1.23891159e-02, 5.58865284e-03, 2.23856780e-03, 1.04144939e-03,
# 4.28470903e-04, 1.61025735e-04, 6.61288363e-05, 2.59990331e-05,
# 1.31222527e-05, 1.01645965e-05, 9.68892403e-06, 9.88392978e-06,
# 1.04056521e-05, 1.01894099e-05, 1.02510858e-05, 1.04777579e-05,
# 1.07304106e-05, 1.05532525e-05, 9.82873024e-06, 9.30648082e-06,
# 8.87185033e-06, 8.52483874e-06, 8.16208814e-06, 7.81650738e-06,
# 7.46618232e-06, 7.06995805e-06, 6.67222766e-06, 6.26161988e-06,
# 5.93015906e-06, 5.53246632e-06, 5.17585318e-06, 4.82793792e-06,
# 4.49015137e-06, 4.03981950e-06, 3.52705868e-06, 3.09408869e-06,
# 2.69253650e-06, 2.49400641e-06, 2.25902807e-06, 1.90052097e-06,
# 1.36664803e-06, 7.01749590e-07, 1.26623833e-07, 0.00000000e+00,
# 1.47630158e-06, 7.86812987e-06, 2.69495496e-05, 7.12259455e-05,
# 1.80559444e-04, 4.46194248e-04, 8.55672431e-04, 1.73592439e-03,
# 3.49044521e-03, 7.07904046e-03, 1.17842178e-02, 2.01285736e-02,
# 3.37324366e-02, 5.68390175e-02, 8.68463318e-02, 1.29032939e-01,
# 1.83364199e-01, 2.44669578e-01, 3.15628853e-01, 3.96687083e-01,
# 4.63171278e-01, 5.59066362e-01, 6.34237143e-01, 6.88736336e-01,
# 7.37621442e-01, 7.80493337e-01, 8.17634420e-01, 8.53965960e-01,
# 8.81644803e-01, 9.03325496e-01, 9.23206367e-01, 9.39031993e-01,
# 9.52997553e-01, 9.64327390e-01])
#
#
#I_ex = abs(I_ex)-min(abs(I_ex))
#I_ex = -I_ex/max(I_ex)


kB=1.380e-23
e=1.602e-19
psi_array = np.arange(0,1.001,0.001)

def H(psi, h_dd, h_uu, h_ud, mud, muu):
    return psi*mud + (1-psi)*muu + 0.5*(h_dd*psi**2 + h_uu*(1-psi)**2 + 2*h_ud*(1-psi)*psi)

def TS(psi, T):
    return -kB*(psi*np.log(psi) + (1-psi)*np.log(1-psi))*T/e*1000

def G(psi, T, h_dd, h_uu, h_ud, mud, muu):
    return H(psi, h_dd, h_uu, h_ud, mud, muu) - TS(psi, T)

def mu(psi, T, h_dd, h_uu, h_ud, mud, muu):
    return (np.diff(G(psi, T, h_dd, h_uu, h_ud, mud, muu))/np.diff(psi))

#def Id_ex(alpha):
#    Vg_T = (alpha*V_ex) 
#    Id_T = -I_ex
#    return np.array(Vg_T).T, np.array(Id_T)

def main():
    st.set_page_config(page_title='Bistability', page_icon = "🧠", initial_sidebar_state = 'auto')
    st.sidebar.header('Parameters')

    ## Initialize session state for the sliders if they don't exist
    #if 'second_mode' not in st.session_state:
    #    st.session_state['second_mode'] = False
    #    st.session_state['h_dd'] = 0.0
    #    st.session_state['h_uu'] = 0.0
    #    st.session_state['h_ud'] = 0.0
    #    st.session_state['mud'] = 0.0
    #    st.session_state['muu'] = 0.0


    st.session_state['second_mode'] = False
    st.session_state['h_dd'] = 0.0
    st.session_state['h_uu'] = 0.0
    st.session_state['h_ud'] = 0.0
    st.session_state['mud'] = 0.0
    st.session_state['muu'] = 0.0


    #T_slider_placeholder = st.sidebar.empty()

#    if st.session_state['second_mode']:
#        # Update default values when second mode is activated
#        st.session_state['h_dd'] = 1.0
#        st.session_state['h_uu'] = 4.0
#        st.session_state['h_ud'] = 75.0
#        st.session_state['mud'] = 0.0
#        st.session_state['muu'] = 0.0 #20.0

    # Now create the sliders with the possibly updated default values from session_state
    h_dd = st.sidebar.slider(r'$h_{dd}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_dd'])
    h_uu = st.sidebar.slider(r'$h_{uu}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_uu'])
    h_ud = st.sidebar.slider(r'$h_{ud}\,(\mathrm{meV})$', -250.0, 250.0, st.session_state['h_ud'])
    mud = st.sidebar.slider(r'$\mu^0_\mathrm{d}\,(\mathrm{meV}):$', -250.0, 250.0, st.session_state['mud'])
    muu = st.sidebar.slider(r'$\mu^0_\mathrm{u}\,(\mathrm{meV}):$', -250.0, 250.0, st.session_state['muu'])
    T = st.sidebar.slider(r'$T\,(\mathrm{K})$', 200.0, 500.0, 300.0)  # Enable the temperature slider when not in second mode


#    if st.session_state['second_mode']:
#        alpha_init = 0.05
#        alpha = st.sidebar.slider(r'Gate Efficiency $\alpha$', 0.0, 1.0, alpha_init)
#        T = T_slider_placeholder.slider(r'$T\,(\mathrm{K})$', 200.0, 500.0, 263.15, disabled=True)  # Disable the temperature slider
#    else:
#        T = T_slider_placeholder.slider(r'$T\,(\mathrm{K})$', 200.0, 500.0, 300.0)  # Enable the temperature slider when not in second mode

    font = {'size' : 14} 
    plt.rc('font', **font)
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    cmap = plt.cm.coolwarm.reversed()
    line_colors = cmap(np.linspace(0,1,9))

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
            axs[4].plot(seg_psi, seg_mu, '--', linewidth=3, color = plt.cm.tab20b(0))
            axs[5].plot(seg_mu, 1 - seg_psi, '--', linewidth=3, color = plt.cm.tab20b(0))
        else:
            axs[4].plot(seg_psi, seg_mu, linewidth=3, color = plt.cm.tab20b(0))
            axs[5].plot(seg_mu, 1 - seg_psi, linewidth=3, color = plt.cm.tab20b(0))



    axs[4].set_title('Chem. Potential', fontsize=16)
    axs[4].set_xlabel(r'$\psi$', fontsize=14)
    axs[4].set_ylabel(r'${\partial G}/{\partial \psi} = \mu$ (meV)', fontsize=14)

    axs[5].set_title("Transfer Curve", fontsize=16)
    axs[5].set_xlabel(r'$V_\mathrm{GS}$ (mV)', fontsize=14)
    axs[5].set_ylabel(r'$-I_\mathrm{D}$ (norm.)', fontsize=14)

    if st.session_state['second_mode']:
        axs[5].plot(Id_ex(alpha)[0], Id_ex(alpha)[1], linestyle='-',
                    linewidth=2, marker='o', markersize=4, color=plt.cm.tab20b(0), alpha=0.6)
        

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


    st.sidebar.markdown("Lukas Bongartz, 2023")

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
