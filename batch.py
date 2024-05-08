import kwant
import random
import bisect
import numpy as np
from copy import copy
from typing import Iterable
from tqdm import tqdm
from physics import *

# __all__ = ['mkhbar_4t','mkhbar_6t','']

def mkhbar_4t(geop,hamp_sys, hamp_lead=None,finalized=False,conservation_law=None):
    from device import Hbar # place this inside function to avoid circular import
    ''' Return a hbar-shaped FiniteSystem or Builder() with four terminals'''
    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    # Hamiltonian parameters belong to both scattering region and leads
    ts, tl = hamp_sys['ts'],hamp_lead['tl']
    ws, wl = hamp_sys['ws'],hamp_lead['wl']
    vs, vl = hamp_sys['vs'],hamp_lead['vl']    
    ms, ml = hamp_sys['ms'],hamp_lead['ml']
    # hamiltonian parameters only belong to the scattering region: lattice constant a, disorder strength Wdis
    Wdis, a = hamp_sys['Wdis'],hamp_sys['a']
    # Geometric parameters for the scattering region
    lx_leg = geop['lx_leg']
    ly_leg = geop['ly_leg']
    lx_neck=geop['lx_neck']
    ly_neck=geop['ly_neck']
    
    lat = kwant.lattice.square(a,norbs=2)

    def onsite(site):    #  mu is the chem pot of the system
        rand_num= random.uniform(-1,1)
        return (4 * ts/(ms*a**2) + Wdis*rand_num) * s_0 + 4 * ws * s_z

    def lead_onsite(site):
        return (4 * tl/(ms*a**2)) * s_0 + 4 * wl * s_z

    def hop_x(site1,site2):
        return -ts/(ms*a**2) * s_0 + 1j * vs/(2*a) * s_y - ws * s_z

    def hop_y(site1,site2):
        return -ts/(ms*a**2) * s_0 - 1j * vs/(2*a) * s_x - ws * s_z

    def lead_hop_x(site1,site2):
        return -tl/(ml*a**2)* s_0 + 1j * vl/(2*a) * s_y - wl * s_z

    def lead_hop_y(site1,site2):
        return -tl/(ml*a**2) * s_0 - 1j * vl/(2*a) * s_x - wl * s_z
    
    # Define Fundamental Domain
    for i in range(lx_leg):
    # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j>0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i>0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):   # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg+ly_neck, ly_leg*2+ly_neck):
            syst[lat(i, j)] = onsite
            if j>ly_leg+ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i>0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg//2 - lx_neck//2, lx_leg//2 + lx_neck//2):   # central connecting neck
        for j in range(ly_leg, ly_leg+ly_neck):
            syst[lat(i, j)] = onsite
            if j>=ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i>lx_leg//2 - lx_neck//2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg+ly_neck), lat(i, ly_leg+ly_neck - 1)] = hop_y
        

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law = conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j>0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law = conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j>0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law = conservation_law)
    for j in range(ly_leg+ly_neck, ly_leg*2+ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j>ly_leg+ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law = conservation_law)
    for j in range(ly_leg+ly_neck, ly_leg*2+ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j>ly_leg+ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)
    
    if finalized:
        return syst.finalized()
    else:
        return syst
    

def mkhbar_6t(geop,hamp_sys,hamp_lead,finalized=False):
    ''' Return a hbar-shaped FiniteSystem or Builder() with six terminals'''
    syst = mkhbar_4t(geop,hamp_sys,hamp_lead,finalized=False,conservation_law=-s_z) # import a normal 4-terminal system to start with convervation_law = -s_z to ensure independent spin channels
    tl = hamp_lead['tl']
    wl = hamp_lead['wl']
    vl = hamp_lead['vl']
    a = hamp_sys['a']
    ml = hamp_lead['ml']
    lx_leg = syst.lx_leg
    lx_neck = syst.lx_neck
    
    lat = kwant.lattice.square(a,norbs=2)

    def lead_onsite(site):
        return (4 * tl/(ml*a**2)) * s_0 + 4 * wl * s_z

    def lead_hop_x(site1,site2):
        return -tl/(ml*a**2)* s_0 + 1j * vl/(2*a) * s_y - wl * s_z

    def lead_hop_y(site1,site2):
        return -tl/(ml*a**2) * s_0 - 1j * vl/(2*a) * s_x - wl * s_z
    
    sym_upper_lead = kwant.TranslationalSymmetry((0, a))
    sym_bottom_lead = kwant.TranslationalSymmetry((0, -a))

    # Add top leads to measure spin current
    # lead No.4   # top middle
    top_middle_lead = kwant.Builder(sym_upper_lead, conservation_law = -s_z)
    for i in range(lx_leg//2 - lx_neck//2, lx_leg//2 + lx_neck//2):
        top_middle_lead[lat(i, 0)] = lead_onsite
        if i>lx_leg//2 - lx_neck//2:
            top_middle_lead[lat(i,0),lat(i-1,0)] = lead_hop_x
        top_middle_lead[lat(i,1),lat(i,0)] = lead_hop_y
    syst.attach_lead(top_middle_lead)

    # lead No.5   # bottom middle
    bot_middle_lead = kwant.Builder(sym_bottom_lead, conservation_law = -s_z)
    for i in range(lx_leg//2 - lx_neck//2, lx_leg//2 + lx_neck//2):
        bot_middle_lead[lat(i, 0)] = lead_onsite
        if i>lx_leg//2 - lx_neck//2:
            bot_middle_lead[lat(i,0),lat(i-1,0)] = lead_hop_x
        bot_middle_lead[lat(i,1),lat(i,0)] = lead_hop_y
    syst.attach_lead(bot_middle_lead)
    
    if finalized:
        return syst.finalized()
    else:
        return syst
    

def j_at_terminal(syst, wf_lead, which_terminal):
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
    lx_leg = syst.lx_leg
    ly_leg = syst.ly_leg
    ly_neck = syst.ly_neck
    lx_neck = syst.lx_neck
    def cutpos(which_terminal):
        def left_upper_lead(site_to,site_from):
            return site_from.pos[0] <= 0 and site_to.pos[0] >0 and site_from.pos[1] >= ly_leg+ly_neck and site_to.pos[1]>=ly_leg+ly_neck
        def left_lower_lead(site_to,site_from):
            return site_from.pos[0] <= 0 and site_to.pos[0] >0 and site_from.pos[1] < ly_leg and site_to.pos[1]<ly_leg
        def right_upper_lead(site_to,site_from):
            return site_from.pos[0] <= lx_leg-2 and site_to.pos[0] >lx_leg-2 and site_from.pos[1] >= ly_leg+ly_neck and site_to.pos[1] >= ly_leg+ly_neck
        def right_lower_lead(site_to,site_from):
            return site_from.pos[0] <= lx_leg-2 and site_to.pos[0] >lx_leg-2 and site_from.pos[1] < ly_leg and site_to.pos[1]<ly_leg
        def middle_lower_lead(site_to,site_from):
            return site_from.pos[1] <= 0 and site_to.pos[1] >0 and site_from.pos[0] >= lx_leg//2 - lx_neck//2 and site_from.pos[0]<lx_leg//2 + lx_neck//2 and site_to.pos[0] >= lx_leg//2 - lx_neck//2 and site_to.pos[0]<lx_leg//2 + lx_neck//2
        def middle_upper_lead(site_to,site_from):
            return site_from.pos[1] < ly_leg*2+ly_neck-1 and site_to.pos[1] >=ly_leg*2+ly_neck-1 and site_from.pos[0] >= lx_leg//2 - lx_neck//2 and site_from.pos[0]<lx_leg//2 + lx_neck//2 and site_to.pos[0] >= lx_leg//2 - lx_neck//2 and site_to.pos[0]<lx_leg//2 + lx_neck//2
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
    fsyst = syst.finalized()
    j_operator = kwant.operator.Current(fsyst, where=cutpos(which_terminal), sum=True)
    return (j_operator(mode) for mode in wf_lead)


def rho_at_site(syst, wf_lead):
    fsyst = syst.finalized()
    rho_operator = kwant.operator.Density(fsyst, sum=False)
    return (rho_operator(mode) for mode in wf_lead)

def rhoz_at_site(syst, wf_lead,rhoz_op=sigma_z):
    fsyst = syst.finalized()
    rhoz_operator = kwant.operator.Density(fsyst, rhoz_op, sum=False)
    return (rhoz_operator(mode) for mode in wf_lead)

def j_at_site(syst, wf_lead):
    fsyst = syst.finalized()
    j_operator = kwant.operator.Current(fsyst, sum=False)
    return (j_operator(mode) for mode in wf_lead)

def rho_j_energy_site(syst,energy:float):
    '''Calculate charge density/curent on each site'''
    fsyst = syst.finalized()
    # Determine the size of the arrays needed
    wf = kwant.wave_function(fsyst, energy=energy)
    sample_wf = wf(0)
    # print(f"At energy={energy},sample_wf.shape={sample_wf.shape}")
    max_num_modes = sample_wf.shape[0]
    site_num = int(sample_wf.shape[1]//2)  # the length of each mode including two orbitals is twice the number of sites
    j_operator = kwant.operator.Current(fsyst, sum=False)
    j_num = len(j_operator(sample_wf[0]))
    num_leads = len(syst.leads)
    # Initialize NumPy arrays
    rho_site = np.zeros((num_leads, max_num_modes, site_num))
    j_site = np.zeros((num_leads, max_num_modes, j_num))

    num_modes =wf(0).shape[0]
    for which_lead in range(num_leads):
        modes = wf(which_lead)
        for mode_idx, rho_idx, j_idx in zip(range(num_modes), rho_at_site(syst, modes), j_at_site(syst, modes)):
            rho_site[which_lead, mode_idx] = rho_idx
            j_site[which_lead, mode_idx] = j_idx
    return rho_site, j_site


def vvector_4t(syst,energy:float, ivector=[0,0,1,-1]):
    ivec = copy(ivector)
    fsyst = syst.finalized()
    which_ground = ivector.index(min(ivector)) # according to the provided current vector, determine the number of lead that is grounded.

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

    G30 = sm.transmission(3, 0)
    G31 = sm.transmission(3, 1)
    G32 = sm.transmission(3, 2)

    mat_full = np.array([[G01+G02+G03,-G01,-G02,-G03],
                [-G10,G10+G12+G13,-G12,-G13],
                [-G20,-G21,G20+G21+G23,-G23],
                [-G30,-G31,-G32,G30+G31+G32]])
    
    Gmat = np.delete(np.delete(mat_full,which_ground,axis=0),which_ground,axis=1)
    
    ivec.remove(min(ivec))  # lead #3 is grounded, [# lead]
    ivec = np.array(ivec)
    try:
        vvec = list(np.linalg.solve(e2h * Gmat, ivec))
        vvec.insert(which_ground,0)
    except:
        vvec = [np.nan]*4
    return tuple(vvec)

    
def vvector_6t(syst,energy:float,ivector=[0,0,1,-1,0,0]):
    ivec = copy(ivector)
    fsyst = syst.finalized()
    sm = kwant.smatrix(fsyst, energy)
    which_ground = ivector.index(min(ivector)) # according to the provided current vector, determine the number of lead that is grounded.

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

    G30 = sm.transmission(3, 0)
    G31 = sm.transmission(3, 1)
    G32 = sm.transmission(3, 2)
    G34 = sm.transmission(3, 4)
    G35 = sm.transmission(3, 5)

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


    mat_full = np.array([[G01 + G02 + G03 + G04 + G05, -G01, -G02, -G03, -G04, -G05],
             [-G10, G10 + G12 + G13 + G14 + G15, -G12, -G13, -G14, -G15],
             [-G20, -G21, G20 + G21 + G23 + G24 + G25, -G23, -G24, -G25],
             [-G30, -G31, -G32, G30 + G31 + G32 + G34 + G35, -G34, -G35],
             [-G40, -G41, -G42, -G43, G40 + G41 + G42 + G43 + G45, -G45],
             [-G50, -G51, -G52, -G53, -G54, G50 + G51 + G52 + G53 + G54]
            ])
    
    Gmat = np.delete(np.delete(mat_full,which_ground,axis=0),which_ground,axis=1)
    ivec.remove(min(ivec))
    ivec = np.array(ivec)

    try:
        vvec = list(np.linalg.solve(e2h * Gmat, ivec))
        vvec.insert(which_ground,0)
        vvec = tuple(vvec)
        Is5up   =   el/(4*np.pi) * ((tm(4,0,0,0) + tm(4,0,0,1)) * (vvec[4] - vvec[0]) 
                                    + (tm(4,0,1,0) + tm(4,0,1,1)) * (vvec[4]-vvec[1]) 
                                    + (tm(4,0,2,0) + tm(4,0,2,1)) * (vvec[4] - vvec[2]) 
                                    + (tm(4,0,3,0) + tm(4,0,3,1)) * (vvec[4]- vvec[3])
                                    + (tm(4,0,5,0) + tm(4,0,5,1)) * (vvec[4] - vvec[5])
                                    )

        Is5down =   el/(4*np.pi) * ((tm(4,1,0,0) + tm(4,1,0,1)) * (vvec[4] - vvec[0]) 
                                    + (tm(4,1,1,0) + tm(4,1,1,1)) * (vvec[4]-vvec[1])  
                                    + (tm(4,1,2,0) + tm(4,1,2,1)) * (vvec[4] - vvec[2]) 
                                    + (tm(4,1,3,0) + tm(4,1,3,1)) * (vvec[4]- vvec[3])
                                    + (tm(4,1,5,0) + tm(4,1,5,1)) * (vvec[4] - vvec[5])
                                )     
    except:
        vvec = tuple([np.nan]*6)
        Is5up = np.nan
        Is5down = np.nan
        
    return vvec, Is5up, Is5down
        

def vary_energy_vvector_4t(syst,energies:Iterable,ivector=[0,0,1,-1]):
    vvec = []
    for energy in tqdm(energies,desc="Progress", ascii=False, ncols=75):
        vvec_at_this_energy = vvector_4t(syst,energy,ivector)
        vvec.append(vvec_at_this_energy)
    return vvec

def vary_energy_vvector_6t(syst,energies:Iterable,ivector=[0,0,1,-1,0,0]):
    vvec = []
    Is5up = []
    Is5down = []
    for energy in tqdm(energies, desc="Progress", ascii=False, ncols=75):
        vvec_at_this_energy, Is5up_at_this_energy, Is5down_at_this_energy = vvector_6t(syst,energy,ivector)
        vvec.append(vvec_at_this_energy)
        Is5up.append(Is5up_at_this_energy)
        Is5down.append(Is5down_at_this_energy)
    return vvec, Is5up, Is5down

#### Middle level #####

def varyx_voltage_4t(mktemplate, geop, hamp_sys, hamp_lead, xkey,xvalue,energy,ivector=[0,0,1,-1]):
    if isinstance(xkey,str):
        if xkey in geop:
            geop[xkey] = xvalue
        elif xkey in hamp_sys:
            hamp_sys[xkey] = xvalue
        elif xkey in hamp_lead:
            hamp_lead[xkey] = xvalue
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey,tuple) and isinstance(xvalue,tuple):
        for xxkey, xxvalue in zip(xkey,xvalue):
            if xxkey in geop:
                geop[xxkey] = xxvalue
            elif xxkey in hamp_sys:
                hamp_sys[xxkey] = xxvalue
            elif xxkey in hamp_lead:
                hamp_lead[xxkey] = xxvalue
            else:
                raise ValueError(f"The key {xxkey} does not exit")
    else:
        raise ValueError("(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)")
    syst = mktemplate(geop, hamp_sys, hamp_lead, False)
    if len(syst.leads) != 4:
        raise ValueError("template should make a system with 4 terminals")
    return vvector_4t(syst,energy,ivector)


def varyx_voltage_6T(mktemplate, geop, hamp_sys, hamp_lead, xkey, xvalue, energy,ivector=[0,0,1,-1,0,0]):
    if isinstance(xkey,str):
        if xkey in geop:
            geop[xkey] = xvalue
        elif xkey in hamp_sys:
            hamp_sys[xkey] = xvalue
        elif xkey in hamp_lead:
            hamp_lead[xkey] = xvalue
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey,tuple) and isinstance(xvalue,tuple):
        for xxkey, xxvalue in zip(xkey,xvalue):
            if xxkey in geop:
                geop[xxkey] = xxvalue
            elif xxkey in hamp_sys:
                hamp_sys[xxkey] = xxvalue
            elif xxkey in hamp_lead:
                hamp_lead[xxkey] = xxvalue
            else:
                raise ValueError(f"The key {xxkey} does not exit")
    else:
        raise ValueError("(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)")
    syst = mktemplate(geop, hamp_sys, hamp_lead, False)
    if len(syst.leads) != 6:
        raise ValueError('template should make a system with 6 terminals')
    return vvector_6t(syst,energy,ivector)
    
def varyx_rho_j_energy_site(mktemplate, geop, hamp_sys, hamp_lead, xkey, xvalue, energy:float):
    if isinstance(xkey,str):
        if xkey in geop:
            geop[xkey] = xvalue
        elif xkey in hamp_sys:
            hamp_sys[xkey] = xvalue
        elif xkey in hamp_lead:
            hamp_lead[xkey] = xvalue
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey,tuple) and isinstance(xvalue,tuple):
        for xxkey, xxvalue in zip(xkey,xvalue):
            if xxkey in geop:
                geop[xxkey] = xxvalue
            elif xxkey in hamp_sys:
                hamp_sys[xxkey] = xxvalue
            elif xxkey in hamp_lead:
                hamp_lead[xxkey] = xxvalue
            else:
                raise ValueError(f"The key {xxkey} does not exit")
    else:
        raise ValueError("(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)")
    syst = mktemplate(geop, hamp_sys, hamp_lead, False)
    return rho_j_energy_site(syst,energy)

def varyx_idos(mktemplate, geop, hamp_sys, hamp_lead, xkey, xvalue,energy_range:Iterable):
    if isinstance(xkey,str):
        if xkey in geop:
            geop[xkey] = xvalue
        elif xkey in hamp_sys:
            hamp_sys[xkey] = xvalue
        elif xkey in hamp_lead:
            hamp_lead[xkey] = xvalue
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey,tuple) and isinstance(xvalue,tuple):
        for xxkey, xxvalue in zip(xkey,xvalue):
            if xxkey in geop:
                geop[xxkey] = xxvalue
            elif xxkey in hamp_sys:
                hamp_sys[xxkey] = xxvalue
            elif xxkey in hamp_lead:
                hamp_lead[xxkey] = xxvalue
            else:
                raise ValueError(f"The key {xxkey} does not exit")
    else:
        raise ValueError("(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)")
    
    dos = []
    syst = mktemplate(geop, hamp_sys, hamp_lead)
    num_leads = len(syst.leads)
    fsyst = syst.finalized()
    rho = kwant.operator.Density(fsyst,sum=True)
    for energy in energy_range:
        wf = kwant.wave_function(fsyst,energy=energy)
        all_states = np.vstack([wf(i) for i in range(num_leads)])
        dos.append(sum(rho(mode) for mode in all_states)/syst.area/(2*np.pi)) # Here syst.area is the actual area / (lattice constant a)**2
    y = np.array(dos)
    idos = abs(energy_range[0]-energy_range[1])*np.nancumsum(y)    
    return idos,energy_range


def find_position(sorted_list,x):
    index = bisect.bisect_left(sorted_list,x)
    if index != len(sorted_list):
        return index
    return -1

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


