import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd



File = pd.read_csv('In Nitrogen/Data/01122021/Batlogg.txt', delimiter='\t',skiprows=[1]).to_numpy()
temperature = np.arange(303.15,258.15,-5)
temperature_C = temperature[:]-273.15
temperature_str = np.arange(303,258,-5).astype(str)
length = 122

VD_list=[-0.1]   
VD_list_str=['01']

for i in np.arange(len(temperature)):
    exec('T_{0}_all = File[i*length:(i+1)*length,:]'.format(temperature_str[i]))
V_G_data = np.empty([len(T_303_all),len(temperature)])
for i in np.arange(len(temperature)):
    exec('V_G_data[:,i] = T_{0}_all[:,7]'.format(temperature_str[i]))    
I_D_data = np.empty([len(T_303_all),len(temperature)])
for i in np.arange(len(temperature)):
    exec('I_D_data[:,i] = -abs(T_{0}_all[:,4])'.format(temperature_str[i]))
I_G_data = np.empty([len(T_303_all),len(temperature)])
for i in np.arange(len(temperature)):
    exec('I_G_data[:,i] = -abs(T_{0}_all[:,8])'.format(temperature_str[i])) 


# Normalise drain current
for i in np.arange(len(temperature)):
    I_D_data[:,i] = abs(I_D_data[:,i])-min(abs(I_D_data[:,i]))
    I_D_data[:,i] = -I_D_data[:,i]/max(I_D_data[:,i])




kB=1.380e-23
e=1.602e-19
phi_array = np.arange(0,1.001,0.001)

Id_ex_T = 263.15

def H(phi, h1, h2, h3, mu0, mup):
    return phi*mu0 + (1-phi)*mup + (h1*phi**2 + h2*(1-phi)**2 + h3*(1-phi)*phi)

def TS(phi, T):
    return -kB*(phi*np.log(phi) + (1-phi)*np.log(1-phi))*T/e*1000

def G(phi, T, h1, h2, h3, mu0, mup):
    return H(phi, h1, h2, h3, mu0, mup) - TS(phi, T)

def mu(phi, T, h1, h2, h3, mu0, mup):
    return (np.diff(G(phi, T, h1, h2, h3, mu0, mup))/np.diff(phi))

def Id_ex(alpha, T, V_shift):
    T_counter = int(np.where(T == temperature)[0])
    Vg_T = (alpha*V_G_data[:,T_counter]+V_shift/1000)*1000, 
    Id_T = -I_D_data[:,T_counter]
    return np.array(Vg_T).T, np.array(Id_T), T_counter

def main():
    st.set_page_config(page_title='Bistability', page_icon = "🧠", initial_sidebar_state = 'auto')
    st.sidebar.header('Parameters')

    h1 = st.sidebar.slider(r'$h_1\,(\mathrm{meV}): \mathrm{PEDOT}^{0}\leftrightarrow \mathrm{PEDOT}^{0}$', -100.0, 100.0, 0.0)
    h2 = st.sidebar.slider(r'$h_2\,(\mathrm{meV}): \mathrm{PEDOT}^{+}\leftrightarrow \mathrm{PEDOT}^{+}$', -100.0, 100.0, 0.0)
    h3 = st.sidebar.slider(r'$h_3\,(\mathrm{meV}): \mathrm{PEDOT}^{0}\leftrightarrow \mathrm{PEDOT}^{+}$', -100.0, 100.0, 0.0)
    T = st.sidebar.slider(r'$T\,(K)$', 200.0, 400.0, 300.0)
    mu0 = st.sidebar.slider(r'$\mu^0_\mathrm{PEDOT^0}\,(\mathrm{meV}):$', 0.0, 500.0, 0.0)
    mup = st.sidebar.slider(r'$\mu^0_\mathrm{PEDOT^+}\,(\mathrm{meV}):$', 0.0, 500.0, 0.0)
    second_mode = st.sidebar.button('Show Experimental Data')

    alpha_init = 0  # Default value
    if second_mode:
        alpha = st.sidebar.slider('alpha', 0.0, 1.0, 0.5)  # Adjust min, max, default values as needed

    font = {'size' : 14} 
    plt.rc('font', **font)
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    cmap = plt.cm.coolwarm.reversed()
    line_colors = cmap(np.linspace(0,1,9))

    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    y_H = H(phi_array, h1, h2, h3, mu0, mup)
    axs[0].plot(phi_array, y_H, linewidth=3, color = plt.cm.tab20b(0))
    axs[0].set_title(r'Enthalpy', fontsize=16)
    axs[0].set_xlabel(r'$\phi$', fontsize=14)
    axs[0].set_ylabel(r'$H_\mathrm{mix}$ (meV)', fontsize=14)

    y_TS = TS(phi_array, T)
    axs[1].plot(phi_array, -y_TS, linewidth=3, color = plt.cm.tab20b(0))
    axs[1].set_title(r'Entropy', fontsize=16)
    axs[1].set_xlabel(r'$\phi$', fontsize=14)
    axs[1].set_ylabel(r'$-TS_\mathrm{mix}$ (meV)', fontsize=14)

    y_G = G(phi_array, T, h1, h2, h3, mu0, mup)
    axs[2].plot(phi_array, y_G, linewidth=3, color = plt.cm.tab20b(0))
    axs[2].set_title(r'Gibbs Free Energy', fontsize=16)
    axs[2].set_xlabel(r'$\phi$', fontsize=14)
    axs[2].set_ylabel(r'$G$ (meV)', fontsize=14)

    y_mu = mu(phi_array, T, h1, h2, h3, mu0, mup)
    slope = np.diff(y_mu) / np.diff(phi_array[:-1])  # calculate slope of mu vs phi
    slope = np.append(slope, 0)  # append a 0 at the end to match the shape of phi and y_mu

    # Calculate the indices where the slope changes sign
    sign_change_indices = np.where(np.diff(np.sign(slope)))[0]

    # Split the indices based on the above condition
    segments_phi = np.split(phi_array[:-1], sign_change_indices+1)
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

    if second_mode:
        # Assuming temperature and V_shift_init are defined somewhere in your code
        temperature = 300  # Example value, replace with your actual temperature
        V_shift_init = 0  # Example value, replace with your actual V_shift_init value

        axs[5].plot(Id_ex(alpha_init, Id_ex_T, V_shift_init)[0],Id_ex(alpha_init, Id_ex_T, V_shift_init)[1], linestyle='-',
        linewidth=1.5, marker='o', markersize=3, color = line_colors[Id_ex(alpha_init, Id_ex_T, V_shift_init)[2]], alpha = alpha) 

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
