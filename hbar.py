# Last updated April 29, 2024, Lixian

# All functions can be catergoried into three-level hierachy:
# Bottom level1: Take syst.finalized() as argument to implement calculation defined by a certain operator, return quantities in the entire graph (local quantities, like j,js,rho,rho_s) or in a specific format (global quantities, like voltages, spin current)
# Bottom level2 (generator): Take syst.finalized() as argument and parameters as arguments to yield the same calculation as the above functions but for a varying parameter that is not binded to the system, e.g., energy for scattering, density for system.

# Middle level (generator): Take make_system(), i.e. system template and parameters as arguments to yield. These functions aim to provide calculations for a varying parameter related to the system properties contained in p.

# High level: User interface (can be built in customer script), to directly implement the batch processing, take system template and parameters and which to vary as arguments. Mainly call middle level generator in for loop.

#---------------------------------------------------------------
# TODO:
# 1. Build a function to build TB model from a continuum model
# 2. Implement magnetic field B

import kwant
import tinyarray
import random
import numpy as np
import scipy.sparse.linalg as sla
import copy
import bisect
from matplotlib import pyplot as plt
from types import SimpleNamespace
from tqdm import tqdm
from typing import Iterable

######################## Pauli Matrix Definitions ######################

s_0 =   tinyarray.array([[1, 0], [0, 1]])
s_x =   tinyarray.array([[0, 1], [1, 0]])
s_y =   tinyarray.array([[0, -1j], [1j, 0]])
s_z =   tinyarray.array([[1, 0], [0, -1]])

sigma_0 = s_0 # Alias
sigma_x = s_x # Alias
sigma_y = s_y # Alias
sigma_z = s_z # Alias

####################### Constants #####################################

t   =   38.0998/1e3 # = hbar^2/(2m_e) in eV*nm^2
e2h =   0.0000387405 # = e^2/h in A/V
el  =   1.602176634e-19 # e in C
eh  =   2.41799e14 # e/h in 1/s

################################ Sytem Template ###################################3
def make_system(p,finalized=True):
    ''' Return a hbar-shaped FiniteSystem or Builder() with four terminals'''
    ts = p['ts']
    tl = p['tl']
    ws = p['ws']
    wl = p['wl']
    vs = p['vs']    
    vl = p['vl']
    a = p['a']
    m = p['m']
    Wdis= p['Wdis']
    half_xlen_leg = p['half_xlen_leg']
    ylen_leg = p['ylen_leg']
    half_xlen_neck=p['half_xlen_neck']
    ylen_neck=p['ylen_neck']
    
    syst = kwant.Builder()
    lat = kwant.lattice.square(a,norbs=2)

    def onsite(site):    #  mu is the chem pot of the system
        rand_num= random.uniform(-1,1)
        return (4 * ts/(m*a**2) + Wdis*rand_num) * s_0 + 4 * ws * s_z

    def lead_onsite(site):
        return (4 * tl/(m*a**2)) * s_0 + 4 * wl * s_z

    def hop_x(site1,site2):
        return -ts/(m*a**2) * s_0 + 1j * vs/(2*a) * s_y - ws * s_z

    def hop_y(site1,site2):
        return -ts/(m*a**2) * s_0 - 1j * vs/(2*a) * s_x - ws * s_z

    def lead_hop_x(site1,site2):
        return -tl/(m*a**2)* s_0 + 1j * vl/(2*a) * s_y - wl * s_z

    def lead_hop_y(site1,site2):
        return -tl/(m*a**2) * s_0 - 1j * vl/(2*a) * s_x - wl * s_z
    
    # Define Fundamental Domain
    for i in range(2*half_xlen_leg):
    # bottom horizontal leg:  2*half_xlen_leg in x direction, ylen_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ylen_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j>0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i>0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(2*half_xlen_leg):   # top horizontal leg:  xlen_leg in x direction, ylen_leg in y direction
        for j in range(ylen_leg+ylen_neck, ylen_leg*2+ylen_neck):
            syst[lat(i, j)] = onsite
            if j>ylen_leg+ylen_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i>0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(half_xlen_leg - half_xlen_neck, half_xlen_leg + half_xlen_neck):   # central connecting neck
        for j in range(ylen_leg, ylen_leg+ylen_neck):
            syst[lat(i, j)] = onsite
            if j>=ylen_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i>half_xlen_leg - half_xlen_neck:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ylen_leg+ylen_neck), lat(i, ylen_leg+ylen_neck - 1)] = hop_y
        

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law = -s_z)
    for j in range(ylen_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j>0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law = -s_z)
    for j in range(ylen_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j>0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law = -s_z)
    for j in range(ylen_leg+ylen_neck, ylen_leg*2+ylen_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j>ylen_leg+ylen_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law = -s_z)
    for j in range(ylen_leg+ylen_neck, ylen_leg*2+ylen_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j>ylen_leg+ylen_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)
    
    if finalized:
        return syst.finalized()
    else:
        return syst


def make_SHEsystem(p,finalized=True):
    ''' Return a hbar-shaped FiniteSystem or Builder() with six terminals'''
    syst = make_system(p,finalized=False) # import a normal 4-terminal system to start with 
    tl = p['tl']
    wl = p['wl']
    vl = p['vl']
    a = p['a']
    m = p['m']
    half_xlen_leg = p['half_xlen_leg']
    half_xlen_neck=p['half_xlen_neck']
    
    lat = kwant.lattice.square(a,norbs=2)

    def lead_onsite(site):
        return (4 * tl/(m*a**2)) * s_0 + 4 * wl * s_z

    def lead_hop_x(site1,site2):
        return -tl/(m*a**2)* s_0 + 1j * vl/(2*a) * s_y - wl * s_z

    def lead_hop_y(site1,site2):
        return -tl/(m*a**2) * s_0 - 1j * vl/(2*a) * s_x - wl * s_z
    
    sym_upper_lead = kwant.TranslationalSymmetry((0, a))
    sym_bottom_lead = kwant.TranslationalSymmetry((0, -a))

    # Add top leads to measure spin current
    # lead No.4   # top middle
    top_middle_lead = kwant.Builder(sym_upper_lead, conservation_law = -s_z)
    for i in range(half_xlen_leg - half_xlen_neck, half_xlen_leg + half_xlen_neck):
        top_middle_lead[lat(i, 0)] = lead_onsite
        if i>half_xlen_leg - half_xlen_neck:
            top_middle_lead[lat(i,0),lat(i-1,0)] = lead_hop_x
        top_middle_lead[lat(i,1),lat(i,0)] = lead_hop_y
    syst.attach_lead(top_middle_lead)

    # lead No.5   # bottom middle
    bot_middle_lead = kwant.Builder(sym_bottom_lead, conservation_law = -s_z)
    for i in range(half_xlen_leg - half_xlen_neck, half_xlen_leg + half_xlen_neck):
        bot_middle_lead[lat(i, 0)] = lead_onsite
        if i>half_xlen_leg - half_xlen_neck:
            bot_middle_lead[lat(i,0),lat(i-1,0)] = lead_hop_x
        bot_middle_lead[lat(i,1),lat(i,0)] = lead_hop_y
    syst.attach_lead(bot_middle_lead)
    
    if finalized:
        return syst.finalized()
    else:
        return syst

############### default system-specific parameter settings ###############

p_default = dict(
    ts = 0, # For scattering region, it should be 0 (no quadratic term) or t (containing quadratic term)
    tl = 0, # For leads, it should be 0 (no quadratic term) or t (containing quadratic term)
    ws = 0.1,
    wl = 0.1, # Wilson term
    vs = 0.3,
    vl = 0.3, # spin-orbital interaction strength
    a=1.0, # in nm
    m=0.05, # in me
    Wdis = 0, # disorder in onsite energy
    half_xlen_leg=10, # site number
    ylen_leg=100,
    half_xlen_neck=5,
    ylen_neck=10
    )


############ Core Functions ##############
##########################################
def current_cut_at_terminal(p, which_terminal):
    ''' Return Boolen function to define the position of cuts right at the interface of terminals, which is used as "where" argument in kwant.operator.Current'''
    half_xlen_leg = p['half_xlen_leg']
    ylen_leg = p['ylen_leg']
    ylen_neck = p['ylen_neck']
    half_xlen_neck = p['half_xlen_neck']
    def left_upper_lead(site_to,site_from):
        return site_from.pos[0] <= 0 and site_to.pos[0] >0 and site_from.pos[1] >= ylen_leg+ylen_neck and site_to.pos[1]>=ylen_leg+ylen_neck
    def left_lower_lead(site_to,site_from):
        return site_from.pos[0] <= 0 and site_to.pos[0] >0 and site_from.pos[1] < ylen_leg and site_to.pos[1]<ylen_leg
    def right_upper_lead(site_to,site_from):
        return site_from.pos[0] <= 2*half_xlen_leg-2 and site_to.pos[0] >2*half_xlen_leg-2 and site_from.pos[1] >= ylen_leg+ylen_neck and site_to.pos[1] >= ylen_leg+ylen_neck
    def right_lower_lead(site_to,site_from):
        return site_from.pos[0] <= 2*half_xlen_leg-2 and site_to.pos[0] >2*half_xlen_leg-2 and site_from.pos[1] < ylen_leg and site_to.pos[1]<ylen_leg
    def middle_lower_lead(site_to,site_from):
        return site_from.pos[1] <= 0 and site_to.pos[1] >0 and site_from.pos[0] >= half_xlen_leg - half_xlen_neck and site_from.pos[0]<half_xlen_leg + half_xlen_neck and site_to.pos[0] >= half_xlen_leg - half_xlen_neck and site_to.pos[0]<half_xlen_leg + half_xlen_neck
    def middle_upper_lead(site_to,site_from):
        return site_from.pos[1] < ylen_leg*2+ylen_neck-1 and site_to.pos[1] >=ylen_leg*2+ylen_neck-1 and site_from.pos[0] >= half_xlen_leg - half_xlen_neck and site_from.pos[0]<half_xlen_leg + half_xlen_neck and site_to.pos[0] >= half_xlen_leg - half_xlen_neck and site_to.pos[0]<half_xlen_leg + half_xlen_neck
    
    if which_terminal == 'lu': 
        return left_upper_lead
    elif which_terminal == 'll':
        return left_lower_lead
    elif which_terminal == 'ru':
        return right_upper_lead
    elif which_terminal == 'rl':
        return right_lower_lead
    elif which_terminal == 'ml':
        return middle_lower_lead
    else:
        return middle_upper_lead
current_cut =  current_cut_at_terminal   # Assign an alias to provide backward compatibility


#### Bottom level (Local quantities) #####
def j_at_terminal(fsyst, p, wf_lead, which_terminal):
    ''' Calculate the charge current due to the propagation modes (wf_lead) from a single lead to the terminal defined by which_terminal
    Parameters
    ----------
    fsyst : `FiniteSystem` instance
    p : dict-like
    wf_lead : kwant.wave_function from a single lead
    which_terminal : 'lu','ll','ru','rl','ml','mu'

    Returns
    -------
    Raises
    ------
    Notes
    -----
    '''
    where = current_cut_at_terminal(p, which_terminal)
    j_operator = kwant.operator.Current(fsyst, where=where, sum=True)
    return (j_operator(mode) for mode in wf_lead)

def jz_at_terminal(fsyst, p, wf_lead, which_terminal,jz_op=sigma_z):
    where = current_cut_at_terminal(p, which_terminal)
    jz_operator = kwant.operator.Current(fsyst, jz_op, where=where, sum=True)
    return (jz_operator(mode) for mode in wf_lead)

def rho_at_site(fsyst, wf_lead):
    rho_operator = kwant.operator.Density(fsyst, sum=False)
    return (rho_operator(mode) for mode in wf_lead)

def rhoz_at_site(fsyst, wf_lead,rhoz_op=sigma_z):
    rhoz_operator = kwant.operator.Density(fsyst, rhoz_op, sum=False)
    return (rhoz_operator(mode) for mode in wf_lead)

def j_at_site(fsyst, wf_lead):
    j_operator = kwant.operator.Current(fsyst, sum=False)
    return (j_operator(mode) for mode in wf_lead)

def jz_at_site(fsyst, wf_lead,jz_op=sigma_z):
    jz_operator = kwant.operator.Current(fsyst, jz_op, sum=False)
    return (jz_operator(mode) for mode in wf_lead)

def rho_J_energy_site(fsyst,energy:float):
    '''Calculate charge density/curent on each site'''
    # Determine the size of the arrays needed
    sample_wf = kwant.wave_function(fsyst, energy=energy)(0)
    # print(f"At energy={energy},sample_wf.shape={sample_wf.shape}")
    max_num_modes = sample_wf.shape[0]
    site_num = int(sample_wf.shape[1]/2)  # the length of each mode including two orbitals is twice the number of sites
    J = kwant.operator.Current(fsyst, sum=False)
    j_num = len(J(sample_wf[0]))
    num_leads = len(fsyst.leads)
    # Initialize NumPy arrays
    rho_site = np.zeros((num_leads, max_num_modes, site_num))
    J_site = np.zeros((num_leads, max_num_modes, j_num))

    wf = kwant.wave_function(fsyst, energy=energy)
    num_modes =wf(0).shape[0]
    for which_lead in range(num_leads):
        modes = wf(which_lead)
        for mode_idx, rho_idx, j_idx in zip(range(num_modes), rho_at_site(fsyst, modes), j_at_site(fsyst, modes)):
            rho_site[which_lead, mode_idx] = rho_idx
            J_site[which_lead, mode_idx] = j_idx
    return rho_site,J_site


def JJz_energy_lead(syst,p_syst,energy):
    # J[energy][lead_out][lead_in][mode]
    # Jz[energy][lead_out][lead_in][mode]
    left_upper_lead = current_cut(p_syst,'lu')
    left_lower_lead = current_cut(p_syst,'ll')
    right_upper_lead = current_cut(p_syst,'ru')
    right_lower_lead = current_cut(p_syst,'rl')
    
    J_leftupper = kwant.operator.Current(syst, where=left_upper_lead, sum=True)
    J_leftlower = kwant.operator.Current(syst, where=left_lower_lead, sum=True)
    J_rightupper = kwant.operator.Current(syst, where=right_upper_lead, sum=True)
    J_rightlower = kwant.operator.Current(syst, where=right_lower_lead , sum=True)

    Jz_leftupper = kwant.operator.Current(syst, s_z, where=left_upper_lead, sum=True)
    Jz_leftlower = kwant.operator.Current(syst, s_z, where=left_lower_lead, sum=True)
    Jz_rightupper = kwant.operator.Current(syst, s_z, where=right_upper_lead, sum=True)
    Jz_rightlower = kwant.operator.Current(syst, s_z, where=right_lower_lead , sum=True)
    
    J, Jz = [], []
    for erg in tqdm(energy,desc="Progress", ascii=False, ncols=75):
        wf = kwant.wave_function(syst,energy=erg)
        j_erg,jz_erg = [],[]
        for which_lead in range(4):
            j0 = [J_leftlower(mode) for mode in wf(which_lead)]  
            j1 = [J_rightlower(mode) for mode in wf(which_lead)]  
            j2 = [J_leftupper(mode) for mode in wf(which_lead)]  
            j3 = [J_rightupper(mode) for mode in wf(which_lead)]
            j_lead = [j0,j1,j2,j3]
            j_erg.append(j_lead)
            
            jz0 = [Jz_leftlower(mode) for mode in wf(which_lead)]  
            jz1 = [Jz_rightlower(mode) for mode in wf(which_lead)]  
            jz2 = [Jz_leftupper(mode) for mode in wf(which_lead)]  
            jz3 = [Jz_rightupper(mode) for mode in wf(which_lead)]
            jz_lead = [jz0,jz1,jz2,jz3]
            jz_erg.append(jz_lead)
        J.append(j_erg)
        Jz.append(jz_erg)
    return J, Jz
JJz_energy = JJz_energy_lead # Assign an alias to provide backward compatibility


def JJz_SHEenergy_lead(syst,p_syst,energy):
    # J[energy][lead_out][lead_in][mode]
    # Jz[energy][lead_out][lead_in][mode]
    left_upper_lead = current_cut(p_syst,'lu')
    left_lower_lead = current_cut(p_syst,'ll')
    right_upper_lead = current_cut(p_syst,'ru')
    right_lower_lead = current_cut(p_syst,'rl')
    middle_lower_lead = current_cut(p_syst,'ml')
    middle_upper_lead = current_cut(p_syst,'mu')
    
    J_leftupper = kwant.operator.Current(syst, where=left_upper_lead, sum=True)
    J_leftlower = kwant.operator.Current(syst, where=left_lower_lead, sum=True)
    J_rightupper = kwant.operator.Current(syst, where=right_upper_lead, sum=True)
    J_rightlower = kwant.operator.Current(syst, where=right_lower_lead , sum=True)
    J_middlelower = kwant.operator.Current(syst, where=middle_lower_lead , sum=True)
    J_middleupper = kwant.operator.Current(syst, where=middle_upper_lead , sum=True)
    
    
    Jz_leftupper = kwant.operator.Current(syst, s_z, where=left_upper_lead, sum=True)
    Jz_leftlower = kwant.operator.Current(syst, s_z, where=left_lower_lead, sum=True)
    Jz_rightupper = kwant.operator.Current(syst, s_z, where=right_upper_lead, sum=True)
    Jz_rightlower = kwant.operator.Current(syst, s_z, where=right_lower_lead , sum=True)
    Jz_middlelower = kwant.operator.Current(syst, s_z, where=middle_lower_lead , sum=True)
    Jz_middleupper = kwant.operator.Current(syst, s_z, where=middle_upper_lead , sum=True)
    
    J, Jz = [], []
    for erg in tqdm(energy,desc="Progress", ascii=False, ncols=75):
        wf = kwant.wave_function(syst,energy=erg)
        j_erg,jz_erg = [],[]
        for which_lead in range(6):
            j0 = [J_leftlower(mode) for mode in wf(which_lead)]  
            j1 = [J_rightlower(mode) for mode in wf(which_lead)]  
            j2 = [J_leftupper(mode) for mode in wf(which_lead)]  
            j3 = [J_rightupper(mode) for mode in wf(which_lead)]
            j4 = [J_middleupper(mode) for mode in wf(which_lead)]
            j5 = [J_middlelower(mode) for mode in wf(which_lead)]
            j_lead = [j0,j1,j2,j3,j4,j5]
            j_erg.append(j_lead)
            
            jz0 = [Jz_leftlower(mode) for mode in wf(which_lead)]  
            jz1 = [Jz_rightlower(mode) for mode in wf(which_lead)]  
            jz2 = [Jz_leftupper(mode) for mode in wf(which_lead)]  
            jz3 = [Jz_rightupper(mode) for mode in wf(which_lead)]
            jz4 = [Jz_middleupper(mode) for mode in wf(which_lead)]
            jz5 = [Jz_middlelower(mode) for mode in wf(which_lead)]
            jz_lead = [jz0,jz1,jz2,jz3,jz4,jz5]
            jz_erg.append(jz_lead)
        J.append(j_erg)
        Jz.append(jz_erg)
    return J, Jz
JJz_SHEenergy = JJz_SHEenergy_lead # Assign an alias to provide downward compatibility


#### Bottom level (global quantities) #####

def voltage_4T(fsyst,energy:float,Iin=1):
    sm = kwant.smatrix(fsyst, energy)
    G01 = sm.transmission(0, 1)
    G02 = sm.transmission(0, 2)
    G03 = sm.transmission(0, 3)

    G10 = sm.transmission(1, 0)
    G12 = sm.transmission(1, 2)
    G13 = sm.transmission(1, 3)

    G20 = sm.transmission(2, 0)
    G21 = sm.transmission(2, 1)
    G23 = sm.transmission(2, 3)

    G_Mat = [[G01 + G02 + G03, -G01, -G02],
             [-G10, G10 + G12 + G13, -G12],
             [-G20, -G21, G20 + G21 + G23]]
    
    I_Vec = [0, 0, Iin]  # lead #3 is grounded, [# lead]
    try:
        V_Vec = list(np.linalg.solve(e2h * np.array(G_Mat), np.array(I_Vec)))
        V_Vec.append(0) # append voltage V3 for Lead #3
    except:
        V_Vec = tuple([np.nan]*4)
    return tuple(V_Vec)

    
def voltage_6T(fsyst,energy:float,Iin=1):
    sm = kwant.smatrix(fsyst, energy)
    def tm(leadout, sigmaout, leadin, sigmain):
        return sm.transmission((leadout, sigmaout), (leadin, sigmain))
    G01 = sm.transmission(0, 1)
    G02 = sm.transmission(0, 2)
    G03 = sm.transmission(0, 3)
    G04 = sm.transmission(0, 4)
    G05 = sm.transmission(0, 5)

    G10 = sm.transmission(1, 0)
    G12 = sm.transmission(1, 2)
    G13 = sm.transmission(1, 3)
    G14 = sm.transmission(1, 4)
    G15 = sm.transmission(1, 5)

    G20 = sm.transmission(2, 0)
    G21 = sm.transmission(2, 1)
    G23 = sm.transmission(2, 3)
    G24 = sm.transmission(2, 4)
    G25 = sm.transmission(2, 5)

    G40 = sm.transmission(4, 0)
    G41 = sm.transmission(4, 1)
    G42 = sm.transmission(4, 2)
    G43 = sm.transmission(4, 3)
    G45 = sm.transmission(4, 5)

    G50 = sm.transmission(5, 0)
    G51 = sm.transmission(5, 1)
    G52 = sm.transmission(5, 2)
    G53 = sm.transmission(5, 3)
    G54 = sm.transmission(5, 4)

    G_Mat = [[G01 + G02 + G03 + G04 + G05, -G01, -G02, -G04, -G05],
             [-G10, G10 + G12 + G13 + G14 + G15, -G12, -G14, -G15],
             [-G20, -G21, G20 + G21 + G23 + G24 + G25, -G24, -G25],
             [-G40, -G41, -G42, G40 + G41 + G42 + G43 + G45, -G45],
             [-G50, -G51, -G52, -G54, G50 + G51 + G52 + G53 + G54]
            ]
    
    I_Vec = [0, 0, Iin, 0, 0]  # lead #3 is grounded, [# lead] I_Vec = [#0,#1,#2,#4,#5]
    try:
        V_Vec = list(np.linalg.solve(e2h * np.array(G_Mat), np.array(I_Vec)))
        V_Vec.insert(3,0) # append voltage V3 for Lead #3
        V_Vec = tuple(V_Vec)
        Is5up   =   el/(4*np.pi) * ((tm(4,0,0,0) + tm(4,0,0,1)) * (V_Vec[4] - V_Vec[0]) 
                                    + (tm(4,0,1,0) + tm(4,0,1,1)) * (V_Vec[4]-V_Vec[1]) 
                                    + (tm(4,0,2,0) + tm(4,0,2,1)) * (V_Vec[4] - V_Vec[2]) 
                                    + (tm(4,0,3,0) + tm(4,0,3,1)) * (V_Vec[4]- V_Vec[3])
                                    + (tm(4,0,5,0) + tm(4,0,5,1)) * (V_Vec[4] - V_Vec[5])
                                   )

        Is5down =   el/(4*np.pi) * ((tm(4,1,0,0) + tm(4,1,0,1)) * (V_Vec[4] - V_Vec[0]) 
                                    + (tm(4,1,1,0) + tm(4,1,1,1)) * (V_Vec[4]-V_Vec[1])  
                                    + (tm(4,1,2,0) + tm(4,1,2,1)) * (V_Vec[4] - V_Vec[2]) 
                                    + (tm(4,1,3,0) + tm(4,1,3,1)) * (V_Vec[4]- V_Vec[3])
                                    + (tm(4,1,5,0) + tm(4,1,5,1)) * (V_Vec[4] - V_Vec[5])
                                   )     
    except:
        
        V_Vec = tuple([np.nan]*6)
        Is5up = np.nan
        Is5down = np.nan
        
    return V_Vec, Is5up, Is5down
        

def vary_energy_voltage_4T(fsyst,energies:Iterable,Iin=1):
    voltages = []
    for energy in tqdm(energies,desc="Progress", ascii=False, ncols=75):
        volt_at_this_energy = voltage_4T(fsyst,energy,Iin)
        voltages.append(volt_at_this_energy)
    return voltages
solve_voltage = vary_energy_voltage_4T

def vary_energy_voltage_6T(fsyst,energies:Iterable,Iin=1):
    voltages = []
    Is5up = []
    Is5down = []
    for energy in tqdm(energies,desc="Progress", ascii=False, ncols=75):
        volt_at_this_energy, Is5up_at_this_energy, Is5down_at_this_energy = voltage_6T(fsyst,energy,Iin)
        voltages.append(volt_at_this_energy)
        Is5up.append(Is5up_at_this_energy)
        Is5down.append(Is5down_at_this_energy)
    return voltages, Is5up, Is5down
solve_SHEvoltage = vary_energy_voltage_6T

#### Middle level #####

def vary_xinp_voltage_4T(template,p,xinp,xvalue,energy:float,Iin=1):
    if isinstance(xinp,str):
        p[str(xinp)] = xvalue
    elif isinstance(xinp,tuple) and isinstance(xvalue,tuple):
        for xxinp, xxvalue in zip(xinp,xvalue):
            p[str(xxinp)] = xxvalue
    else:
        raise ValueError('(xinp,xvalue) should be either (str,numbers) or (tuple,tuple)')
    fsyst = template(p)
    if len(fsyst.leads) != 4:
        raise ValueError('template should make a system with 4 terminals')
    return voltage_4T(fsyst,energy,Iin)

def vary_xinp_voltage_6T(template,p,xinp,xvalue,energy:float,Iin=1):
    if isinstance(xinp,str):
        p[str(xinp)] = xvalue
    elif isinstance(xinp,tuple) and isinstance(xvalue,tuple):
        for xxinp, xxvalue in zip(xinp,xvalue):
            p[str(xxinp)] = xxvalue
    else:
        raise ValueError('(xinp,xvalue) should be either (str,numbers) or (tuple,tuple)')
    fsyst = template(p)
    if len(fsyst.leads) != 6:
        raise ValueError('template should make a system with 6 terminals')
    return voltage_6T(fsyst,energy,Iin)
    
def vary_xinp_rho_J_energy_site(template,p,xinp,xvalue,energy:float):
    if isinstance(xinp,str):
        p[str(xinp)] = xvalue
    elif isinstance(xinp,tuple) and isinstance(xvalue,tuple):
        for xxinp, xxvalue in zip(xinp,xvalue):
            p[str(xxinp)] = xxvalue
    else:
        raise ValueError('(xinp,xvalue) should be either (str,numbers) or (tuple,tuple)')
    fsyst = template(p)
    return rho_J_energy_site(fsyst,energy)


def vary_xinp_idos(template,p,xinp,xvalue,energy_range:Iterable):
    if isinstance(xinp,str):
        p[str(xinp)] = xvalue
    elif isinstance(xinp,tuple) and isinstance(xvalue,tuple):
        for xxinp, xxvalue in zip(xinp,xvalue):
            p[str(xxinp)] = xxvalue
    else:
        raise ValueError('(xinp,xvalue) should be either (str,numbers) or (tuple,tuple)')
    area = hbar_area(p)
    dos = []
    fsyst = template(p)
    rho = kwant.operator.Density(fsyst,sum=True)
    num_leads = len(fsyst.leads)
    for energy in energy_range:
        wf = kwant.wave_function(fsyst,energy=energy)
        all_states = np.vstack([wf(i) for i in range(num_leads)])
        dos.append(sum(rho(mode) for mode in all_states)/area/(2*np.pi))
    y = np.array(dos)
    idos = abs(energy_range[0]-energy_range[1])*np.nancumsum(y)    
    return idos,energy_range
#### Top level (Interface) #####

    

def find_position(sorted_list,x):
    index = bisect.bisect_left(sorted_list,x)
    if index != len(sorted_list):
        return index
    return -1

def hbar_area(p): 
    half_xlen_leg = p['half_xlen_leg']
    ylen_leg = p['ylen_leg']
    half_xlen_neck=p['half_xlen_neck']
    ylen_neck=p['ylen_neck']
    return (half_xlen_leg*2)*ylen_leg*2+(half_xlen_neck*2)*ylen_neck

# Calculate density of state
def get_idos(sys,p,energies):
    area = hbar_area(p)
    dos = []
    rho = kwant.operator.Density(sys,sum=True)
    num_leads = len(sys.leads)
    for energy in energies:
        wf = kwant.wave_function(sys,energy=energy)
        all_states = np.vstack([wf(i) for i in range(num_leads)])
        dos.append(sum(rho(mode) for mode in all_states)/area/(2*np.pi))
    y = np.array(dos)
    idos = abs(energies[0]-energies[1])*np.nancumsum(y)
    return idos,energies

def get_idosB(sys,p,energies,Bfield):
    area = hbar_area(p)
    dos = []
    rho = kwant.operator.Density(sys,sum=True)
    num_leads = len(sys.leads)
    for energy in energies:
        wf = kwant.wave_function(sys,energy=energy,params=dict(B=Bfield))
        all_states = np.vstack([wf(i) for i in range(num_leads)])
        dos.append(sum(rho(mode) for mode in all_states)/area/(2*np.pi))
    y = np.array(dos)
    idos = abs(energies[0]-energies[1])*np.nancumsum(y)
    return idos,(energies,Bfield)

def energy_to_density(idos,energies,energy):
    index = find_position(energies,energy)
    if index == -1:
        raise ValueError("need more eigenstates")
    return idos[index]

def density_to_energy(idos,energies,density):
    index = find_position(idos,density)
    if index == -1:
        raise ValueError("need more eigenstates")
    return energies[index]

################## Plots ###############
def plot_device_band(syst,p_syst,figsize=(10,6),ylim=[-0.01,0.05],momenta =np.linspace(-0.3, 0.3, 100)):
    fig = plt.figure(figsize=figsize,tight_layout=True)
    gs = fig.add_gridspec(1,3)
    ax1,ax2 = [fig.add_subplot(x) for x in [gs[0,0],gs[0,1:]]]
    kwant.plot(syst,ax=ax1)
    kwant.plotter.bands(syst.leads[0], momenta=momenta,ax=ax2)
    ax2.set_ylim(ylim[0],ylim[1])
    ax2.set_xlabel("Momentum [nm$^{-1}$]")
    ax2.set_ylabel("Energy [eV]")
    draw_bulk(p_syst,momenta=momenta,ax=ax2)
    return [ax1,ax2]

def plot_transport_erg(syst,p_syst,rnl,rl,energy,figsize=(15,6),momenta =np.linspace(-0.3, 0.3, 100)):
    fig, axs = plt.subplots(1,3,figsize=figsize,tight_layout=True)
    axs[0].plot(energy, rnl)
    axs[1].plot(energy, rl)
    axs[0].set_xlabel('E[eV]')
    axs[1].set_xlabel('E[eV]')
    axs[0].set_ylabel('$R_{34,12}$ [$\Omega$]')
    axs[1].set_ylabel('$R_{34,34}$ [$\Omega$]')
    kwant.plotter.bands(syst.leads[0], momenta=momenta,ax=axs[2])
    axs[2].set_ylim(min(energy),max(energy))
    axs[2].set_xlabel("Momentum [nm$^{-1}$]")
    axs[2].set_ylabel("Energy [eV]")
    return axs
    
def plot_transport_dsy(syst,p_syst,rnl,rl,density,figsize=(12,6)):
    fig, axs = plt.subplots(1,2,figsize=figsize,tight_layout=True)
    axs[0].plot(density, rnl)
    axs[1].plot(density, rl)
    axs[0].set_xlabel('Density [10$^{11}$cm$^{-2}$]')
    axs[1].set_xlabel('Density [10$^{11}$cm$^{-2}$]')
    axs[0].set_ylabel('$R_{34,12}$ [$\Omega$]')
    axs[1].set_ylabel('$R_{34,34}$ [$\Omega$]')
    return axs

############### IO Functions ###############   
def pklsave(data,filename='data.pkl'):
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(data,f)
def pklimport(filename='data.pkl'):
    import pickle
    with open(filename,'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def draw_bulk(p,momenta=np.linspace(-0.3,0.3,100),ax=None):
    vField = p['vs']
    m = p['m']
    t = p['ts']
    energy_up = [t*k**2/m+vField*k for k in momenta]
    energy_down = [t*k**2/m-vField*k for k in momenta]
    if ax is None:
        plt.plot(momenta,energy_up,'r-')
        plt.plot(momenta,energy_down,'b-')
    else:
        ax.plot(momenta,energy_up,'r-')
        ax.plot(momenta,energy_down,'b-')