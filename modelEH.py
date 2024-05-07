# Created on 09.04.2024
# Last update on April 29, 2024
from hbar import *


def prepare_plot(xlabel:str,xlim:tuple,figsize=(10,6)):
    ''' prepare axes for complex plots'''
    fig, axs = plt.subplots(2,2,figsize=figsize,tight_layout=True)
    [ax.set_xlabel(xlabel) for ax in axs[:,0]]
    [ax.set_xlim(*xlim) for ax in axs[:,0]]
    axs[0,0].set_ylabel('$\Delta V_{34}(\lambda)$ [$\mu$V]')
    axs[1,0].set_ylabel('$G_{SH}$ [e/8$\pi$]')
    return axs


def rashba_vary_lambda(lamd = np.linspace(0,80,2),single_lead_current=False,target_density = 0.01):
    '''Calculate the nonlocal signal in a 4-terminal hbar and spin current in a 6-terminal hbar for a variety of spin-orbital coupling strength \lambda
    Mirror the original configuration from EH's PRB paper.
    Set I3 = Iin , I4 = -Iin, I(the rest) = 0
    Measure V1-V2 - step1
    Measure Gsh = (I5up-I5down)/(V3-V4)
    The scattering energy for each scenario correpsonds to a fixed carrier density.
    '''
    print('Function rashba_vary_lambda is called.')    
    # restrictions
    Iin = 10e-9 # A
    target_density = 0.01
    
    deltaV12_inmV = []
    gsh = []
    target_energies = []
    # grid parameters
    N1,L = 36,90
    # core parameters (initialization)
    p_EH = copy.deepcopy(p_default)
    p_EH['ts'] = t
    p_EH['tl'] = t
    p_EH['wl']=0
    # p_EH['vs']=lambda_in_meVnm/1e3
    p_EH['ws']=0
    p_EH['vl']=0 # Leads have two independent spin channels (no spin-orbital coupling)
    p_EH['Wdis']=0.09/1e3 # eV
    p_EH['a']=L/N1 # in nm
    p_EH['m']=0.05 # in me
    p_EH['ylen_leg']=int(N1/6)
    p_EH['ylen_neck']=int(N1/6)
    p_EH['half_xlen_leg']=int(N1/2)
    p_EH['half_xlen_neck']=int(N1/12)
    
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.05,51)
    target_energies = [density_to_energy(*vary_xinp_idos(make_system,p_EH,'vs',xvalue,energy_range),target_density) for xvalue in (l/1e3 for l in lamd)]
    target_energies_6T = [density_to_energy(*vary_xinp_idos(make_SHEsystem,p_EH,'vs',xvalue,energy_range),target_density) for xvalue in (l/1e3 for l in lamd)]
    
    # calculate Gsh in a 6 terminal hbar
    for xvalue, energy in zip((l/1e3 for l in lamd),target_energies_6T):
        voltages_6T,Isup,Isdown = vary_xinp_voltage_6T(make_SHEsystem,p_EH,'vs',xvalue,energy,Iin)
        gsh.append((Isdown-Isup)/(voltages_6T[2]))
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vary_xinp_voltage_4T(make_system,p_EH,'vs',xvalue,energy,Iin) for xvalue, energy in zip((l/1e3 for l in lamd),target_energies)]
    deltaV12_inmV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    
    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(zip((l/1e3 for l in lamd),target_energies,voltages_list)):
        rho_site,J_site = vary_xinp_rho_J_energy_site(make_system,p_EH,'vs',xvalue,energy)
        sys_EH_4T = make_system(p_EH)
        total_modes = len(rho_site[0])
        print(f"At $\lambda$={xvalue*1e3}, the number of modes:{total_modes}, energy is {energy}")
        
        axs = prepare_plot(xlabel='$\\lambda$ [meV nm]',xlim=(min(lamd)-1,max(lamd)+1),figsize=(10,6))
        kwant.plotter.density(sys_EH_4T, sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[0,1],cmap='jet')
        if single_lead_current:
            kwant.plotter.current(sys_EH_4T, sum(J_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
        else:
            kwant.plotter.current(sys_EH_4T, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
        axs[0,0].plot(lamd,deltaV12_inmV)
        axs[0,0].scatter(lamd,deltaV12_inmV)
        axs[0,0].scatter(xvalue*1e3,deltaV12_inmV[i],color='red',s=100)

        axs[1,0].plot(lamd,[g/el*8*np.pi for g in gsh])
        axs[1,0].scatter(lamd,[g/el*8*np.pi for g in gsh])
        axs[1,0].scatter(xvalue*1e3,gsh[i]/el*8*np.pi,color='red',s=100)

        plt.savefig(f"EHmodel/rashba_vary_lambda/single_lead_current/lambda_{xvalue*1e3}_plot.png")    



def rashba_vary_ylen_neck(ylen_neck_insite = np.arange(1,20,2,dtype=int),single_lead_current=False,target_density = 0.01,lambda_in_meVnm  = 80):
    '''Calculate the nonlocal signal in a 4-terminal hbar and spin current in a 6-terminal hbar for a variety of connection bar length.
    Mirror the original configuration from EH's PRB paper.
    Set I3 = Iin , I4 = -Iin, I(the rest) = 0
    Measure V1-V2 - step1
    Measure Gsh = (I5up-I5down)/(V3-V4)
    The scattering energy for each scenario correpsonds to a fixed carrier density. 
    '''    
    print('Function rashba_vary_ylen_neck is called.')
    # restrictions
    Iin = 10e-9 # A

    deltaV12_inmV = []
    gsh = []
    target_energies = []

    # grid parameters
    N1,L = 36,90
    
    # core parameters
    p_EH = copy.deepcopy(p_default)
    p_EH['ts'] = t
    p_EH['tl'] = t
    p_EH['wl']=0
    p_EH['vs']=lambda_in_meVnm/1e3
    p_EH['ws']=0
    p_EH['vl']=0 # Leads have two independent spin channels (no spin-orbital coupling)
    p_EH['Wdis']=0.09/1e3 # eV
    p_EH['a']=L/N1 # in nm
    p_EH['m']=0.05 # in me
    p_EH['ylen_leg']=int(N1/6)
    # p_EH['ylen_neck']=int(N1/6)
    p_EH['half_xlen_leg']=int(N1/2)
    p_EH['half_xlen_neck']=int(N1/12)
    
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.05,51)
    target_energies = [density_to_energy(*vary_xinp_idos(make_system,p_EH,'ylen_neck',xvalue,energy_range),target_density) for xvalue in ylen_neck_insite]
    target_energies_6T = [density_to_energy(*vary_xinp_idos(make_SHEsystem,p_EH,'ylen_neck',xvalue,energy_range),target_density) for xvalue in ylen_neck_insite]
    
    # calculate Gsh in a 6 terminal hbar
    for xvalue, energy in zip(ylen_neck_insite,target_energies_6T):
        voltages_6T,Isup,Isdown = vary_xinp_voltage_6T(make_SHEsystem,p_EH,'ylen_neck',xvalue,energy,Iin)
        gsh.append((Isdown-Isup)/(voltages_6T[2]))
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vary_xinp_voltage_4T(make_system,p_EH,'ylen_neck',xvalue,energy,Iin) for xvalue, energy in zip(ylen_neck_insite,target_energies)]
    deltaV12_inmV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    
    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(zip(ylen_neck_insite,target_energies,voltages_list)):
        rho_site,J_site = vary_xinp_rho_J_energy_site(make_system,p_EH,'ylen_neck',xvalue,energy)
        sys_EH_4T = make_system(p_EH)
        total_modes = len(rho_site[0])
        print(f"At ylen_neck={xvalue}, the number of modes:{total_modes}, energy is {energy}")
        
        axs = prepare_plot(xlabel='connection bar length (site)',xlim=(min(ylen_neck_insite)-1,max(ylen_neck_insite)+1), figsize=(10,6))
        kwant.plotter.density(sys_EH_4T, sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[0,1],cmap='jet')
        if single_lead_current:
            kwant.plotter.current(sys_EH_4T, sum(J_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
        else:
            kwant.plotter.current(sys_EH_4T, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
        axs[0,0].plot(ylen_neck_insite,deltaV12_inmV)
        axs[0,0].scatter(ylen_neck_insite,deltaV12_inmV)
        axs[0,0].scatter(xvalue,deltaV12_inmV[i],color='red',s=100)

        axs[1,0].plot(ylen_neck_insite,[g/el*8*np.pi for g in gsh])
        axs[1,0].scatter(ylen_neck_insite,[g/el*8*np.pi for g in gsh])
        axs[1,0].scatter(xvalue,gsh[i]/el*8*np.pi,color='red',s=100)
        plt.savefig(f"EHmodel/rashba_vary_ylen_neck/single_lead_current/nlen_{xvalue}_plot.png")    

        
    
def dirac_vary_lambda(lamd = np.linspace(4,80,20),single_lead_current=False,target_density = 0.01):
    '''A copy from rashba_vary_lambda, but instead using a pure Dirac-type Hamiltonian defined by \lambda'''
    print('Function dirac_vary_lambda is called.')
    # restrictions
    Iin = 10e-9 # A
    
    deltaV12_inmV = []
    target_energies = []
    # grid parameters
    N1,L = 36,90
    # core parameters
    p_dirac = copy.deepcopy(p_default)
    p_dirac['ts'] = 0
    p_dirac['tl'] = 0
    # -------------------
    p_dirac['vs'] = 0.1
    p_dirac['vl']=0.1
    p_dirac['ws']=0.1/3
    p_dirac['wl']=0.1/3
    # -------------------
    p_dirac['Wdis']=0 # eV
    p_dirac['a']=L/N1 # in nm
    p_dirac['m']=0.05 # in me
    p_dirac['ylen_leg']=int(N1/6)
    p_dirac['ylen_neck']=int(N1/6)
    p_dirac['half_xlen_leg']=int(N1/2)
    p_dirac['half_xlen_neck']=int(N1/12)
    
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.05,61)
    print(f"Before calculating target energies:vs={p_dirac['vs']},vl={p_dirac['vl']},ws={p_dirac['ws']},wl={p_dirac['wl']}")
    target_energies = [density_to_energy(*vary_xinp_idos(make_system,p_dirac,('vs','vl','ws','wl'),(xvalue,xvalue,xvalue/3,xvalue/3),energy_range),target_density) for xvalue in (l/1e3 for l in lamd)]
    print(f"After calculating target energies:vs={p_dirac['vs']},vl={p_dirac['vl']},ws={p_dirac['ws']},wl={p_dirac['wl']}")
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vary_xinp_voltage_4T(make_system,p_dirac,('vs','vl','ws','wl'),(xvalue,xvalue,xvalue/3,xvalue/3),energy,Iin) for xvalue, energy in zip((l/1e3 for l in lamd),target_energies)]
    print(f"After calculating voltages_list:vs={p_dirac['vs']},vl={p_dirac['vl']},ws={p_dirac['ws']},wl={p_dirac['wl']}")
    deltaV12_inmV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    
    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(zip((l/1e3 for l in lamd),target_energies,voltages_list)):
        rho_site,J_site = vary_xinp_rho_J_energy_site(make_system,p_dirac,('vs','vl','ws','wl'),(xvalue,xvalue,xvalue/3,xvalue/3),energy)

        sys_dirac = make_system(p_dirac)
        total_modes = len(rho_site[0])
        print(f"At $\lambda$={xvalue*1e3}, the number of modes:{total_modes}, the energy is {energy:0.5f}")
        print(f"After calculating rho and J:vs={p_dirac['vs']},vl={p_dirac['vl']},ws={p_dirac['ws']},wl={p_dirac['wl']}")

        
        axs = prepare_plot(xlabel='$\\lambda$ [meV nm]',xlim=(min(lamd)-1,max(lamd)+1),figsize=(10,6))
        kwant.plotter.density(sys_dirac, sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[0,1],cmap='jet')
        if single_lead_current:
            kwant.plotter.current(sys_dirac, sum(J_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
        else:
            kwant.plotter.current(sys_dirac, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
        axs[0,0].plot(lamd,deltaV12_inmV)
        axs[0,0].scatter(lamd,deltaV12_inmV)
        axs[0,0].scatter(xvalue*1e3,deltaV12_inmV[i],color='red',s=100)
    
        plt.savefig(f"EHmodel/dirac_vary_lambda/highdens/test/lambda_{xvalue*1e3}_plot.png")     
        

def dirac_vary_ylen_neck(ylen_neck_insite = np.arange(1,20,2,dtype=int),single_lead_current=False, target_density = 0.01,lambda_in_meVnm  = 80):
    '''A copy from rashba_vary_ylen_neck, but instead using a pure Dirac-type Hamiltonian defined by \lambda'''
    print('Function dirac_vary_ylen_neck is called.')    
    # restrictions
    Iin = 10e-9 # A

    deltaV12_inmV = []
    target_energies = []

    # grid parameters
    N1,L = 36,90
    # core parameters
    p_dirac = copy.deepcopy(p_default)
    p_dirac['ts'] = 0
    p_dirac['tl'] = 0
    p_dirac['vs'] = lambda_in_meVnm/1e3
    p_dirac['vl'] = lambda_in_meVnm/1e3
    p_dirac['ws'] = lambda_in_meVnm/1e3/3
    p_dirac['wl'] = lambda_in_meVnm/1e3/3
    
    p_dirac['Wdis'] = 0 # eV
    p_dirac['a'] = L/N1 # in nm
    p_dirac['m'] = 0.05 # in me
    p_dirac['ylen_leg'] = int(N1/6)
    p_dirac['ylen_neck'] = int(N1/6)
    p_dirac['half_xlen_leg'] = int(N1/2)
    p_dirac['half_xlen_neck'] = int(N1/12)
    
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.linspace(0,0.05,51)
    target_energies = [density_to_energy(*vary_xinp_idos(make_system,p_dirac,'ylen_neck',xvalue,energy_range),target_density) for xvalue in ylen_neck_insite]

    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [vary_xinp_voltage_4T(make_system,p_dirac,'ylen_neck',xvalue,energy,Iin) for xvalue, energy in zip(ylen_neck_insite,target_energies)]
    deltaV12_inmV = [(volts[0]-volts[1])*1e6 for volts in voltages_list]
    
    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(zip(ylen_neck_insite,target_energies,voltages_list)):
        rho_site,J_site = vary_xinp_rho_J_energy_site(make_system,p_dirac,'ylen_neck',xvalue,energy)
        sys_dirac = make_system(p_dirac)
        total_modes = len(rho_site[0])
        print(f"At ylen_neck={xvalue}, the number of modes:{total_modes}, the energy is {energy}")
        
        axs = prepare_plot(xlabel='connection bar length (site)',xlim=(min(ylen_neck_insite)-1,max(ylen_neck_insite)+1), figsize=(10,6))
        kwant.plotter.density(sys_dirac, sum(rho_site[0][mode_num]+rho_site[1][mode_num]+rho_site[2][mode_num]+rho_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[0,1],cmap='jet')
        if single_lead_current:
            kwant.plotter.current(sys_dirac, sum(J_site[3][mode_num] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w') # electron flow from this lead (grounded) to others
        else:
            kwant.plotter.current(sys_dirac, sum(J_site[0][mode_num]*voltages[0]+J_site[1][mode_num]*voltages[1]+J_site[2][mode_num]*voltages[2] for mode_num in range(total_modes)), ax=axs[1,1],cmap='jet',linecolor='w')
        axs[0,0].plot(ylen_neck_insite,deltaV12_inmV)
        axs[0,0].scatter(ylen_neck_insite,deltaV12_inmV)
        axs[0,0].scatter(xvalue,deltaV12_inmV[i],color='red',s=100)
        
        plt.savefig(f"EHmodel/dirac_vary_ylen_neck/single_lead_current/nlen_{xvalue}_plot.png")     



if __name__ == "__main__":
    # rashba_vary_lambda(np.arange(0,80,5),True)
    # rashba_vary_ylen_neck(np.arange(1,50,1),True)
    dirac_vary_lambda(list(np.arange(1,11,1)),True,0.01)
    # dirac_vary_ylen_neck(np.arange(1,50,1),True)