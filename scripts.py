# from hbar import *
import os
import matplotlib.pyplot as plt
from batch import *
from templates import *
from physics import *
from utils import *
from tqdm import tqdm
from log import log_function_call


@log_function_call
def rashba_vary_lambda(lamd = np.linspace(0,80,2),single_lead_current=False,target_density = 0.01):
    '''Calculate the nonlocal signal in a 4-terminal hbar and spin current in a 6-terminal hbar for a variety of spin-orbital coupling strength \lambda
    Mirror the original configuration from EH's PRB paper.
    Set I3 = Iin , I4 = -Iin, I(the rest) = 0
    Measure V1-V2 - step1
    Measure Gsh = (I5up-I5down)/(V3-V4)
    The scattering energy for each scenario correpsonds to a fixed carrier density.
    '''
    Iin = 10e-9 # A
    deltaV12_inmuV = []
    target_energies = []
    gsh = []
    # grid parameters
    N1,L = 36,90
    # core parameters
    geop = dict(lx_leg=int(N1), ly_leg=int(N1/6), lx_neck=int(N1/6), ly_neck=int(N1/6))

    hamp_sys = dict(ts=tk,ws=0,vs=0,ms=0.05,Wdis=0,a=L/N1)
    hamp_lead = dict(tl=tk,wl=0,vl=0,ml=0.05)
    
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.arange(0,0.05,0.001) # This determines the energy resolution in the integrated DOS. The higher the better but also more expensive.

    target_energies = [density_to_energy(*varyx_idos(mkhbar_4t,geop,hamp_sys,hamp_lead,'vs',xvalue,energy_range),target_density) for xvalue in (l/1e3 for l in lamd)]
    target_energies_6t = [density_to_energy(*varyx_idos(mkhbar_6t,geop,hamp_sys,hamp_lead,'vs',xvalue,energy_range),target_density) for xvalue in (l/1e3 for l in lamd)]
    
    print(target_energies)
    print(target_energies_6t)
    
    # calculate Gsh in a 6 terminal hbar
    for xvalue, energy in zip((l/1e3 for l in lamd),target_energies_6t):
        voltages_6t,Isup,Isdown = varyx_voltage_6t(mkhbar_6t,geop,hamp_sys,hamp_lead,'vs',xvalue,energy,[0,0,Iin,-Iin,0,0])
        gsh.append((Isdown-Isup)/(voltages_6t[2]))
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [varyx_voltage_4t(mkhbar_4t,geop,hamp_sys,hamp_lead,'vs',xvalue,energy,[0,0,Iin,-Iin]) for xvalue, energy in zip((l/1e3 for l in lamd),target_energies)]
    deltaV12_inmuV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    
    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(zip((l/1e3 for l in lamd),target_energies,voltages_list)):
        rho_site,J_site = varyx_rho_j_energy_site(mkhbar_4t,geop,hamp_sys,hamp_lead,'vs',xvalue,energy)
        print(f"hamp_sys:{hamp_sys}")
        sys_EH_4T = mkhbar_4t(geop,hamp_sys,hamp_lead).finalized()
        if rho_site is not None:
            total_modes = len(rho_site[0])
            print(f"At $\lambda$={xvalue*1e3}, the number of modes:{total_modes}, energy is {energy}")
            
            axs = prepare_plot(xlabel='$\\lambda$ [meV nm]',xlim=(min(lamd)-1,max(lamd)+1),figsize=(10,6))
            kwant.plotter.density(sys_EH_4T, np.array(sum(rho_site[0][mode_num]+rho_site[1][mode_num]+
                                                          rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes)),dtype=object), ax=axs[0,1],cmap='jet')
            if single_lead_current:
                kwant.plotter.current(sys_EH_4T, np.array(sum(J_site[3][mode_num] for mode_num in range(total_modes)),dtype=object), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
            else:
                kwant.plotter.current(sys_EH_4T, np.array(sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]
                                                              +J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)),dtype=object), ax=axs[1,1],cmap='jet',linecolor='w')
            axs[0,0].plot(lamd,deltaV12_inmuV)
            axs[0,0].scatter(lamd,deltaV12_inmuV)
            axs[0,0].scatter(xvalue*1e3,deltaV12_inmuV[i],color='red',s=100)

            axs[1,0].plot(lamd,[g/el*8*np.pi for g in gsh])
            axs[1,0].scatter(lamd,[g/el*8*np.pi for g in gsh])
            axs[1,0].scatter(xvalue*1e3,gsh[i]/el*8*np.pi,color='red',s=100)
            plt.savefig(f"rashba_lambda_{xvalue*1e3}_plot.png")    

        else:
            print(f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed")

@log_function_call
def dirac_vary_lambda(lamd = np.linspace(4,80,20),single_lead_current=False,target_density = 0.01,savepath=None):
    '''A copy from rashba_vary_lambda, but instead using a pure Dirac-type Hamiltonian defined by \lambda'''
    if savepath is None:
        savepath = os.getcwd()
    Iin = 10e-9 # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1,L = 36,90
    # core parameters
    geop = dict(lx_leg=int(N1), ly_leg=int(N1/6), lx_neck=int(N1/6), ly_neck=int(N1/6))
    hamp_sys = dict(ts=0,ws=0.1/3,vs=0.1,ms=0.05,Wdis=0,a=L/N1)
    hamp_lead = dict(tl=0,wl=0.1/3,vl=0.1,ml=0.05)
    
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.05,101)
    target_energies = [density_to_energy(*varyx_idos(mkhbar_4t,geop,hamp_sys,hamp_lead,('vs','vl','ws','wl'),(xvalue,xvalue,xvalue/3,xvalue/3),energy_range),target_density) for xvalue in (l/1e3 for l in lamd)]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [varyx_voltage_4t(mkhbar_4t,geop,hamp_sys,hamp_lead,('vs','vl','ws','wl'),(xvalue,xvalue,xvalue/3,xvalue/3),energy,[0,0,Iin,-Iin]) for xvalue, energy in zip((l/1e3 for l in lamd),target_energies)]
    deltaV12_inmuV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2]-volts[3])*1e6 for volts in voltages_list]

    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(zip((l/1e3 for l in lamd),target_energies,voltages_list)):
        rho_site,J_site = varyx_rho_j_energy_site(mkhbar_4t,geop,hamp_sys,hamp_lead,('vs','vl','ws','wl'),(xvalue,xvalue,xvalue/3,xvalue/3),energy)
        sys_dirac = mkhbar_4t(geop,hamp_sys,hamp_lead).finalized()
        print(f"hamp_sys:{hamp_sys}")
        if rho_site is not None:
            total_modes = len(rho_site[0])
            print(f"At $\lambda$={xvalue*1e3}, the number of modes:{total_modes}, the energy is {energy:0.5f}")
            axs = prepare_plot(xlabel='$\\lambda$ [meV nm]',xlim=(min(lamd)-1,max(lamd)+1),
                               ylabel2 = '$\Delta V_{12}(\lambda)$ [$\mu$V]',figsize=(10,6))
            kwant.plotter.density(sys_dirac, np.array(sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[0,1],cmap='jet')
            if single_lead_current:
                kwant.plotter.current(sys_dirac, np.array(sum(J_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
            else:
                kwant.plotter.current(sys_dirac, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
            axs[0,0].plot(lamd,deltaV12_inmuV)
            axs[0,0].scatter(lamd,deltaV12_inmuV)
            axs[0,0].scatter(xvalue*1e3,deltaV12_inmuV[i],color='red',s=100)

            axs[1,0].plot(lamd,deltaV34_inmuV)
            axs[1,0].scatter(lamd,deltaV34_inmuV)
            axs[1,0].scatter(xvalue*1e3,deltaV34_inmuV[i],color='red',s=100)

            plt.savefig(os.path.join(savepath,f"dirac_lambda_{xvalue*1e3}_density_{target_density}.png"))   

        else:
            print(f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed")
         
@log_function_call
def dirac_vary_density(densities = np.arange(0.001,0.01,0.001),lamb = 300,single_lead_current=False,savepath=None):
    if savepath is None:
        savepath = os.getcwd()
    lamb = lamb/1e3
    Iin = 10e-9 # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1,L = 36,90
    # core parameters
    geop = dict(lx_leg=int(N1), ly_leg=int(N1/6), lx_neck=int(N1/6), ly_neck=int(N1/6))
    hamp_sys = dict(ts=0,ws=lamb/3,vs=lamb,ms=0.05,Wdis=0,a=L/N1)
    hamp_lead = dict(tl=0,wl=lamb/3,vl=lamb,ml=0.05)
    syst = mkhbar_4t(geop,hamp_sys,hamp_lead) # This system won't be changed anymore

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.06,121)
    target_energies = [density_to_energy(*get_idos(syst,energy_range),target_density) for target_density in densities]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vvector_4t(syst,energy,[0,0,Iin,-Iin]) for energy in target_energies]
    deltaV12_inmuV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2]-volts[3])*1e6 for volts in voltages_list]

    
    # Calculate more local quantities and plot them separately for each density
    for i, (energy, voltages) in enumerate(zip(target_energies,voltages_list)):
        rho_site,J_site = rho_j_energy_site(syst,energy)
        fsyst = syst.finalized()
        print(f"hamp_sys:{hamp_sys}")
        if rho_site is not None:
            total_modes = len(rho_site[0])
            print(f"At $density$={densities[i]:0.5f}, the number of modes:{total_modes}, the energy is {energy:0.5f}")
            axs = prepare_plot(xlabel='Density [10$^{11}$ cm$^{-2}$]',xlim=(min(densities)*1e3-1,max(densities)*1e3+1),
                               ylabel2 = '$\Delta V_{12}(\lambda)$ [$\mu$V]', figsize=(10,6))
            kwant.plotter.density(fsyst, np.array(sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[0,1],cmap='jet')
            if single_lead_current:
                kwant.plotter.current(fsyst, np.array(sum(J_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
            else:
                kwant.plotter.current(fsyst, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
            x = [density*1e3 for density in densities]
            axs[0,0].plot(x,deltaV12_inmuV)
            axs[0,0].scatter(x,deltaV12_inmuV)
            axs[0,0].scatter(densities[i]*1e3,deltaV12_inmuV[i],color='red',s=100)

            axs[1,0].plot(x,deltaV34_inmuV)
            axs[1,0].scatter(x,deltaV34_inmuV)
            axs[1,0].scatter(densities[i]*1e3,deltaV34_inmuV[i],color='red',s=100)

            plt.savefig(os.path.join(savepath,f"dirac_lambda{lamb*1e3}_density_{densities[i]:0.5f}.png"))   

        else:
            print(f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed")

@log_function_call
def quad_vary_density(densities = np.arange(0.001,0.01,0.001),single_lead_current=False,savepath=None):
    if savepath is None:
        savepath = os.getcwd()
    Iin = 10e-9 # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1,L = 36,90
    # core parameters
    geop = dict(lx_leg=int(N1), ly_leg=int(N1/6), lx_neck=int(N1/6), ly_neck=int(N1/6))
    hamp_sys = dict(ts=tk,ws=0,vs=0,ms=0.05,Wdis=0,a=L/N1)
    hamp_lead = dict(tl=tk,wl=0,vl=0,ml=0.05)
    syst = mkhbar_4t(geop,hamp_sys,hamp_lead) # This system won't be changed anymore

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.03,61)
    target_energies = [density_to_energy(*get_idos(syst,energy_range),target_density) for target_density in densities]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vvector_4t(syst,energy,[0,0,Iin,-Iin]) for energy in target_energies]
    deltaV12_inmuV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2]-volts[3])*1e6 for volts in voltages_list]

    
    # Calculate more local quantities and plot them separately for each density
    for i, (energy, voltages) in enumerate(zip(target_energies,voltages_list)):
        rho_site,J_site = rho_j_energy_site(syst,energy)
        fsyst = syst.finalized()
        print(f"hamp_sys:{hamp_sys}")
        if rho_site is not None:
            total_modes = len(rho_site[0])
            print(f"At $density$={densities[i]:0.5f}, the number of modes:{total_modes}, the energy is {energy:0.5f}")
            axs = prepare_plot(xlabel='Density [10$^{11}$ cm$^{-2}$]',xlim=(min(densities)*1e3-1,max(densities)*1e3+1),
                               ylabel2 = '$\Delta V_{12}(\lambda)$ [$\mu$V]', figsize=(10,6))
            kwant.plotter.density(fsyst, np.array(sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[0,1],cmap='jet')
            if single_lead_current:
                kwant.plotter.current(fsyst, np.array(sum(J_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
            else:
                kwant.plotter.current(fsyst, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
            x = [density*1e3 for density in densities]
            axs[0,0].plot(x,deltaV12_inmuV)
            axs[0,0].scatter(x,deltaV12_inmuV)
            axs[0,0].scatter(densities[i]*1e3,deltaV12_inmuV[i],color='red',s=100)

            axs[1,0].plot(x,deltaV34_inmuV)
            axs[1,0].scatter(x,deltaV34_inmuV)
            axs[1,0].scatter(densities[i]*1e3,deltaV34_inmuV[i],color='red',s=100)

            plt.savefig(os.path.join(savepath,f"quad_density_{densities[i]:0.5f}.png"))   

        else:
            print(f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed")

@log_function_call
def rashba_vary_density(densities = np.arange(0.001,0.01,0.001),lamb=300,single_lead_current=False,savepath=None):
    lamb = lamb/1e3
    if savepath is None:
        savepath = os.getcwd()
    Iin = 10e-9 # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1,L = 36,90
    # core parameters
    geop = dict(lx_leg=int(N1), ly_leg=int(N1/6), lx_neck=int(N1/6), ly_neck=int(N1/6))
    hamp_sys = dict(ts=tk,ws=0,vs=lamb,ms=0.05,Wdis=0,a=L/N1)
    hamp_lead = dict(tl=tk,wl=0,vl=lamb,ml=0.05)
    syst = mkhbar_4t(geop,hamp_sys,hamp_lead) # This system won't be changed anymore

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.03,61)
    target_energies = [density_to_energy(*get_idos(syst,energy_range),target_density) for target_density in densities]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vvector_4t(syst,energy,[0,0,Iin,-Iin]) for energy in target_energies]
    deltaV12_inmuV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2]-volts[3])*1e6 for volts in voltages_list]

    
    # Calculate more local quantities and plot them separately for each density
    for i, (energy, voltages) in enumerate(zip(target_energies,voltages_list)):
        rho_site,J_site = rho_j_energy_site(syst,energy)
        fsyst = syst.finalized()
        print(f"hamp_sys:{hamp_sys}")
        if rho_site is not None:
            total_modes = len(rho_site[0])
            print(f"At $density$={densities[i]:0.5f}, the number of modes:{total_modes}, the energy is {energy:0.5f}")
            axs = prepare_plot(xlabel='Density [10$^{11}$ cm$^{-2}$]',xlim=(min(densities)*1e3-1,max(densities)*1e3+1),
                               ylabel2 = '$\Delta V_{12}(\lambda)$ [$\mu$V]', figsize=(10,6))
            kwant.plotter.density(fsyst, np.array(sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[0,1],cmap='jet')
            if single_lead_current:
                kwant.plotter.current(fsyst, np.array(sum(J_site[3][mode_num] for mode_num in range(total_modes))), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
            else:
                kwant.plotter.current(fsyst, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
            x = [density*1e3 for density in densities]
            axs[0,0].plot(x,deltaV12_inmuV)
            axs[0,0].scatter(x,deltaV12_inmuV)
            axs[0,0].scatter(densities[i]*1e3,deltaV12_inmuV[i],color='red',s=100)

            axs[1,0].plot(x,deltaV34_inmuV)
            axs[1,0].scatter(x,deltaV34_inmuV)
            axs[1,0].scatter(densities[i]*1e3,deltaV34_inmuV[i],color='red',s=100)

            plt.savefig(os.path.join(savepath,f"rashba_lamb_{lamb*1e3}_density_{densities[i]:0.5f}.png"))   

        else:
            print(f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed")



if __name__ == "__main__":
    # rashba_vary_lambda(lamd = np.arange(0,80,2),single_lead_current=True,target_density = 0.01)
    dirac_vary_lambda(lamd = np.arange(20,300,400),single_lead_current=True,target_density = 0.01,savepath='plots/new')
    # dirac_vary_density(densities = np.arange(0.0005,0.015,0.0005),lamb = 300,single_lead_current=True,savepath='plots/dirac_vary_density')
    # quad_vary_density(densities = np.arange(0.0005,0.015,0.0005),single_lead_current=True,savepath='plots/quad_vary_density')
    # rashba_vary_density(densities = np.arange(0.0005,0.015,0.0005),lamb = 300, single_lead_current=True,savepath='plots/rashba_vary_density')