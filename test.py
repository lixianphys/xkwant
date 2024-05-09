import numpy as np
from batch import *
from physics import *
from utils import *
from templates import *


def test_hbar_from_cmodel():
    from device import Hbar
    geop = dict(lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
    bhz_continuum = '''
        + mu * kron(sigma_0, sigma_0)
        + M * kron(sigma_0, sigma_z)
        - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z) - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
        + A * (k_x * kron(sigma_z, sigma_x) + k_y * kron(sigma_0, sigma_y))
    '''
    hbar_from_cmodel = Hbar(geop)
    hbar_from_cmodel.build_byfill(bhz_continuum)
    ham_params = dict(A=0.09, B=-0.18, D=-0.065, M=-0.02, mu=0)
    print(hbar_from_cmodel)
    [hbar_from_cmodel.attach_lead_byfill(bhz_continuum,pos) for pos in ['bl','br','tl','tr']]
    hbar_from_cmodel.set_ham_params(ham_params)
    print(hbar_from_cmodel)

def test_hbar_from_mk():
    from batch import mkhbar_4t,mkhbar_6t
    geop = dict(lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
    ham_sys = dict(ts=tk,ws=0,vs=0.1,ms=0.05,Wdis=0,a=1)
    ham_lead = dict(tl=tk,wl=0,vl=0,ml=0.05)

    hbar_from_mk_4t = mkhbar_4t(geop,ham_sys,ham_lead,False)
    print(hbar_from_mk_4t)
    hbar_from_mk_6t = mkhbar_6t(geop,ham_sys,ham_lead,False)
    print(hbar_from_mk_6t)

def test_batch():
    lamd = np.linspace(0,80,2)
    Iin = 10e-9 # A
    target_density = 0.01
    geop = dict(lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
    ham_sys = dict(ts=tk,ws=0,vs=0.1,ms=0.05,Wdis=0,a=1)
    ham_lead = dict(tl=tk,wl=0,vl=0,ml=0.05)
    energy_range = np.linspace(0,0.15,6)
    print(list(density_to_energy(*varyx_idos(mkhbar_4t,geop,ham_sys,ham_lead,'vs',xvalue,energy_range),target_density) for xvalue in (l/1e3 for l in lamd)))
    print(list(density_to_energy(*varyx_idos(mkhbar_6t,geop,ham_sys,ham_lead,'vs',xvalue,energy_range),target_density) for xvalue in (l/1e3 for l in lamd)))
    print(*vary_energy_vvector_4t(mkhbar_4t(geop,ham_sys,ham_lead,False),energies=energy_range,ivector=[0,0,Iin,-Iin]))
    print(*vary_energy_vvector_6t(mkhbar_6t(geop,ham_sys,ham_lead,False),energies=energy_range,ivector=[0,0,Iin,-Iin,0,0]))
    rho_site,j_site = varyx_rho_j_energy_site(mkhbar_4t,geop,ham_sys,ham_lead,'vl',0.1,0.2)
    rho_site,j_site = varyx_rho_j_energy_site(mkhbar_4t,geop,ham_sys,ham_lead,('vs','vl'),(0.1,0.1),0.2)
    print(rho_site)
    print(j_site)