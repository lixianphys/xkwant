import kwant
import numpy as np
from copy import copy
from typing import Iterable, Union
from tqdm import tqdm
from xkwant.physics import *
from xkwant.utils import get_idos
from xkwant.schemas import GeomParams, HamParams

__all__ = [
    "vary_energy_vvector_4t",
    "vary_energy_vvector_6t",
    "varyx_voltage_4t",
    "varyx_voltage_6t",
    "varyx_rho_j_energy_site",
    "varyx_idos",
    "vvector_4t",
    "vvector_6t",
    "rho_j_energy_site",
]


def _rho_at_site(syst: kwant.Builder, wf_lead: kwant.wave_function) -> tuple:
    fsyst = syst.finalized()
    rho_operator = kwant.operator.Density(fsyst, sum=False)
    return (rho_operator(mode) for mode in wf_lead)


def _rhoz_at_site(syst: kwant.Builder, wf_lead: kwant.wave_function, rhoz_op=sigma_z) -> tuple:
    fsyst = syst.finalized()
    rhoz_operator = kwant.operator.Density(fsyst, rhoz_op, sum=False)
    return (rhoz_operator(mode) for mode in wf_lead)


def _j_at_site(syst: kwant.Builder, wf_lead: kwant.wave_function) -> tuple:
    fsyst = syst.finalized()
    j_operator = kwant.operator.Current(fsyst, sum=False)
    return (j_operator(mode) for mode in wf_lead)


def rho_j_energy_site(
    syst: kwant.Builder, 
    energy: float) -> tuple:
    """Calculate charge density/curent on each site at a specific energy"""
    fsyst = syst.finalized()
    # Determine the size of the arrays needed
    wf = kwant.wave_function(fsyst, energy=energy)
    sample_wf = wf(0)
    if len(sample_wf) == 0:  # the mode number == 0
        return None, None
    # print(f"At energy={energy},sample_wf.shape={sample_wf.shape}")
    max_num_modes = sample_wf.shape[0]
    site_num = (
        sample_wf.shape[1] // 2
    )  # the length of each mode including two orbitals is twice the number of sites
    j_operator = kwant.operator.Current(fsyst, sum=False)
    j_num = len(j_operator(sample_wf[0]))
    num_leads = len(syst.leads)
    # Initialize NumPy arrays
    rho_site = np.zeros((num_leads, max_num_modes, site_num))
    j_site = np.zeros((num_leads, max_num_modes, j_num))

    for which_lead in range(num_leads):
        modes = wf(which_lead)
        for mode_idx, rho_idx, j_idx in zip(
            range(len(modes)), _rho_at_site(syst, modes), _j_at_site(syst, modes)
        ):
            rho_site[which_lead, mode_idx] = rho_idx  # rho_site[#lead][#mode][#site]
            j_site[which_lead, mode_idx] = j_idx
    return rho_site, j_site


def vvector_4t(
    syst: kwant.Builder,
    energy: float,
    ivector: list = None
) -> tuple:
    """
    Calculate the voltage vector for a 4-terminal system

    Parameters
    ----------
    syst : kwant.Builder
        The system to calculate the voltage vector for
    energy : float
        The energy to calculate the voltage vector for
    ivector : list
        The current vector to calculate the voltage vector for

    Returns
    -------
    tuple
        The voltage vector for the 4-terminal system
    """
    if ivector is None:
        ivector = [0, 0, 1, -1]
    if len(ivector) != 4:
        raise ValueError("ivector should be a list of 4 elements")

    ivec = copy(ivector)
    fsyst = syst.finalized()
    which_ground = ivector.index(
        min(ivector)
    )  # according to the provided current vector, determine the number of lead that is grounded.

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

    mat_full = np.array(
        [
            [G01 + G02 + G03, -G01, -G02, -G03],
            [-G10, G10 + G12 + G13, -G12, -G13],
            [-G20, -G21, G20 + G21 + G23, -G23],
            [-G30, -G31, -G32, G30 + G31 + G32],
        ]
    )

    Gmat = np.delete(np.delete(mat_full, which_ground, axis=0), which_ground, axis=1)

    ivec.remove(min(ivec))  # lead #3 is grounded, [# lead]
    ivec = np.array(ivec)
    try:
        vvec = list(np.linalg.solve(e2h * Gmat, ivec))
        vvec.insert(which_ground, 0)
    except Exception as e:
        print(f"Failed to calculate the voltage vector due to {e}, return nan")
        vvec = [np.nan] * 4
    return tuple(vvec)


def vvector_6t(
    syst: kwant.Builder,
    energy: float,
    ivector: list = None
) -> tuple:
    """
    Calculate the voltage vector for a 6-terminal system

    Parameters
    ----------
    syst : kwant.Builder
        The system to calculate the voltage vector for
    energy : float
        The energy to calculate the voltage vector for
    ivector : list
        The current vector to calculate the voltage vector for

    Returns
    -------
    tuple
        The voltage vector for the 6-terminal system
    """
    if ivector is None:
        ivector = [0, 0, 1, -1, 0, 0]
    if len(ivector) != 6:
        raise ValueError("ivector should be a list of 6 elements")
    ivec = copy(ivector)
    fsyst = syst.finalized()
    sm = kwant.smatrix(fsyst, energy)
    which_ground = ivector.index(
        min(ivector)
    )  # according to the provided current vector, determine the number of lead that is grounded.

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

    mat_full = np.array(
        [
            [G01 + G02 + G03 + G04 + G05, -G01, -G02, -G03, -G04, -G05],
            [-G10, G10 + G12 + G13 + G14 + G15, -G12, -G13, -G14, -G15],
            [-G20, -G21, G20 + G21 + G23 + G24 + G25, -G23, -G24, -G25],
            [-G30, -G31, -G32, G30 + G31 + G32 + G34 + G35, -G34, -G35],
            [-G40, -G41, -G42, -G43, G40 + G41 + G42 + G43 + G45, -G45],
            [-G50, -G51, -G52, -G53, -G54, G50 + G51 + G52 + G53 + G54],
        ]
    )

    Gmat = np.delete(np.delete(mat_full, which_ground, axis=0), which_ground, axis=1)
    ivec.remove(min(ivec))
    ivec = np.array(ivec)

    try:
        vvec = list(np.linalg.solve(e2h * Gmat, ivec))
        vvec.insert(which_ground, 0)
        vvec = tuple(vvec)
        Is5up = (
            el
            / (4 * np.pi)
            * (
                (tm(4, 0, 0, 0) + tm(4, 0, 0, 1)) * (vvec[4] - vvec[0])
                + (tm(4, 0, 1, 0) + tm(4, 0, 1, 1)) * (vvec[4] - vvec[1])
                + (tm(4, 0, 2, 0) + tm(4, 0, 2, 1)) * (vvec[4] - vvec[2])
                + (tm(4, 0, 3, 0) + tm(4, 0, 3, 1)) * (vvec[4] - vvec[3])
                + (tm(4, 0, 5, 0) + tm(4, 0, 5, 1)) * (vvec[4] - vvec[5])
            )
        )

        Is5down = (
            el
            / (4 * np.pi)
            * (
                (tm(4, 1, 0, 0) + tm(4, 1, 0, 1)) * (vvec[4] - vvec[0])
                + (tm(4, 1, 1, 0) + tm(4, 1, 1, 1)) * (vvec[4] - vvec[1])
                + (tm(4, 1, 2, 0) + tm(4, 1, 2, 1)) * (vvec[4] - vvec[2])
                + (tm(4, 1, 3, 0) + tm(4, 1, 3, 1)) * (vvec[4] - vvec[3])
                + (tm(4, 1, 5, 0) + tm(4, 1, 5, 1)) * (vvec[4] - vvec[5])
            )
        )
    except Exception as e:
        print(f"Failed to calculate the voltage vector due to {e}, return nan")
        vvec = tuple([np.nan] * 6)
        Is5up = np.nan
        Is5down = np.nan

    return vvec, Is5up, Is5down


def vary_energy_vvector_4t(
    syst: kwant.Builder,
    energies: Iterable,
    ivector: list = None
) -> tuple:
    """
    Calculate the voltage vector for a 4-terminal system at a range of energies

    Parameters
    ----------
    syst : kwant.Builder
        The system to calculate the voltage vector for
    energies : Iterable
        The energies to calculate the voltage vector for
    ivector : list
        The current vector to calculate the voltage vector for

    """
    if ivector is None:
        ivector = [0, 0, 1, -1]
    vvec = []
    for energy in tqdm(energies, desc="Progress", ascii=False, ncols=75):
        vvec_at_this_energy = vvector_4t(syst, energy, ivector)
        vvec.append(vvec_at_this_energy)
    return vvec


def vary_energy_vvector_6t(
    syst: kwant.Builder,
    energies: Iterable,
    ivector: list = None
) -> tuple:
    """
    Calculate the voltage vector for a 6-terminal system at a range of energies

    Parameters
    ----------
    syst : kwant.Builder
        The system to calculate the voltage vector for
    energies : Iterable
        The energies to calculate the voltage vector for
    ivector : list
        The current vector to calculate the voltage vector for

    """
    if ivector is None:
        ivector = [0, 0, 1, -1, 0, 0]
    vvec = []
    Is5up = []
    Is5down = []
    for energy in tqdm(energies, desc="Progress", ascii=False, ncols=75):
        vvec_at_this_energy, Is5up_at_this_energy, Is5down_at_this_energy = vvector_6t(
            syst, energy, ivector
        )
        vvec.append(vvec_at_this_energy)
        Is5up.append(Is5up_at_this_energy)
        Is5down.append(Is5down_at_this_energy)
    return vvec, Is5up, Is5down


def varyx_voltage_4t(
    mktemplate: callable,
    geop: GeomParams,
    hamp_sys: HamParams,
    hamp_lead: HamParams,
    xkey: Union[str,tuple],
    xvalue: Union[float,tuple],
    energy: float,
    ivector: list = None
) -> tuple:
    """
    Vary the parameters of the system and calculate the voltage vector for a 4-terminal system

    Parameters
    ----------
    mktemplate : callable
        The template to use for the system
    geop : GeomParams
        The geometry parameters of the system
    hamp_sys : HamParams
        The Hamiltonian parameters of the system
    hamp_lead : HamParams
        The Hamiltonian parameters of the leads
    xkey : str or tuple
        The key to vary the parameters of the system
    xvalue : float or tuple
        The value to vary the parameters of the system
    energy : float
        The energy to calculate the voltage vector for
    ivector : list
        The current vector to calculate the voltage vector for

    """
    if ivector is None:
        ivector = [0, 0, 1, -1]
    if len(ivector) != 4:
        raise ValueError("ivector should be a list of 4 elements")
    if isinstance(xkey, str) and isinstance(xvalue, (int, float)):
        if xkey in geop.to_dict().keys():
            setattr(geop, xkey, xvalue)
        elif xkey.endswith("_sys") and xkey.replace("_sys", "") in hamp_sys.to_dict().keys():
            setattr(hamp_sys, xkey.replace("_sys", ""), xvalue)
        elif xkey.endswith("_lead") and xkey.replace("_lead", "") in hamp_lead.to_dict().keys():
            setattr(hamp_lead, xkey.replace("_lead", ""), xvalue)
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey, tuple) and isinstance(xvalue, tuple):
        for xxkey, xxvalue in zip(xkey, xvalue):
            if xxkey in geop.to_dict().keys():
                setattr(geop, xxkey, xxvalue)
            elif xxkey.endswith("_sys") and xxkey.replace("_sys", "") in hamp_sys.to_dict().keys():
                setattr(hamp_sys, xxkey.replace("_sys", ""), xxvalue)
            elif xxkey.endswith("_lead") and xxkey.replace("_lead", "") in hamp_lead.to_dict().keys():
                setattr(hamp_lead, xxkey.replace("_lead", ""), xxvalue)
            else:
                raise ValueError(f"The key {xxkey} does not exit")
    else:
        raise ValueError("(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)")

    syst = mktemplate(geop = geop, hamp_sys = hamp_sys, hamp_lead = hamp_lead, finalized=False)

    if len(syst.leads) != 4:
        raise ValueError("template should make a system with 4 terminals")
    vvec = vvector_4t(syst, energy, ivector)
    del syst
    return vvec


def varyx_voltage_6t(
    mktemplate: callable,
    geop: GeomParams,
    hamp_sys: HamParams,
    hamp_lead: HamParams,
    xkey: Union[str, tuple],
    xvalue: Union[float, tuple],
    energy: float,
    ivector: list | None = None,
) -> tuple:
    """
    Vary the parameters of the system and calculate the voltage vector for a 6-terminal system

    Parameters
    ----------
    mktemplate : callable
        The template function to use for the system
    geop : GeomParams
        The geometry parameters of the system
    hamp_sys : HamParams
        The Hamiltonian parameters of the system
    hamp_lead : HamParams
        The Hamiltonian parameters of the leads
    xkey : str or tuple
        The key to vary the parameters of the system
    xvalue : float or tuple
        The value to vary the parameters of the system
    """
    if ivector is None:
        ivector = [0, 0, 1, -1, 0, 0]
    if len(ivector) != 6:
        raise ValueError("ivector should be a list of 6 elements")
    if isinstance(xkey, str) and isinstance(xvalue, (int, float)):
        if xkey in geop.to_dict().keys():
            setattr(geop, xkey, xvalue)
        elif xkey.endswith("_sys") and xkey.replace("_sys", "") in hamp_sys.to_dict().keys():
            setattr(hamp_sys, xkey.replace("_sys", ""), xvalue)
        elif xkey.endswith("_lead") and xkey.replace("_lead", "") in hamp_lead.to_dict().keys():
            setattr(hamp_lead, xkey.replace("_lead", ""), xvalue)
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey, tuple) and isinstance(xvalue, tuple):
        for xxkey, xxvalue in zip(xkey, xvalue):
            if xxkey in geop.to_dict().keys():
                setattr(geop, xxkey, xxvalue)
            elif xxkey.endswith("_sys") and xxkey.replace("_sys", "") in hamp_sys.to_dict().keys():
                setattr(hamp_sys, xxkey.replace("_sys", ""), xxvalue)
            elif xxkey.endswith("_lead") and xxkey.replace("_lead", "") in hamp_lead.to_dict().keys():
                setattr(hamp_lead, xxkey.replace("_lead", ""), xxvalue)
            else:
                raise ValueError(f"The key {xxkey} does not exit") 
    else:
        raise ValueError("(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)")
    syst = mktemplate(geop, hamp_sys, hamp_lead, False)
    if len(syst.leads) != 6:
        raise ValueError("template should make a system with 6 terminals")
    vvec, Is5up, Is5down = vvector_6t(syst, energy, ivector)
    del syst
    return vvec, Is5up, Is5down


def varyx_rho_j_energy_site(
    mktemplate: callable, 
    geop: GeomParams, 
    hamp_sys: HamParams, 
    hamp_lead: HamParams, 
    xkey: Union[str,tuple], 
    xvalue: Union[float,tuple], 
    energy: float
) -> tuple:
    """
    Vary the parameters of the system and calculate the charge density/current on each site at a specific energy for a set of parameters defined by xkey and xvalue
    """
    if isinstance(xkey, str) and isinstance(xvalue, (int, float)):
        if xkey in geop.to_dict().keys():
            setattr(geop, xkey, xvalue)
        elif xkey.endswith("_sys") and xkey.replace("_sys", "") in hamp_sys.to_dict().keys():
            setattr(hamp_sys, xkey.replace("_sys", ""), xvalue)
        elif xkey.endswith("_lead") and xkey.replace("_lead", "") in hamp_lead.to_dict().keys():
            setattr(hamp_lead, xkey.replace("_lead", ""), xvalue)
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey, tuple) and isinstance(xvalue, tuple):
        for xxkey, xxvalue in zip(xkey, xvalue):
            if xxkey in geop.to_dict().keys():
                setattr(geop, xxkey, xxvalue)
            elif xxkey.endswith("_sys") and xxkey.replace("_sys", "") in hamp_sys.to_dict().keys():
                setattr(hamp_sys, xxkey.replace("_sys", ""), xxvalue)
            elif xxkey.endswith("_lead") and xxkey.replace("_lead", "") in hamp_lead.to_dict().keys():
                setattr(hamp_lead, xxkey.replace("_lead", ""), xxvalue)
            else:
                raise ValueError(f"The key {xxkey} does not exit") 
    else:
        raise TypeError(
            "(xkey, xvalue) should be either (str,numbers) or (tuple,tuple)"
        )
    syst = mktemplate(geop, hamp_sys, hamp_lead, False)
    rho_site, j_site = rho_j_energy_site(syst, energy)
    del syst
    return rho_site, j_site


def varyx_idos(
    mktemplate: callable,
    geop: GeomParams,
    hamp_sys: HamParams,
    hamp_lead: HamParams,
    xkey: Union[str, tuple],
    xvalue: Union[float, tuple],
    energy_range: Iterable
) -> tuple:
    if isinstance(xkey, str) and isinstance(xvalue, (int, float)):
        if xkey in geop.to_dict().keys():
            setattr(geop, xkey, xvalue)
        elif xkey.endswith("_sys") and xkey.replace("_sys", "") in hamp_sys.to_dict().keys():
            setattr(hamp_sys, xkey.replace("_sys", ""), xvalue)
        elif xkey.endswith("_lead") and xkey.replace("_lead", "") in hamp_lead.to_dict().keys():
            setattr(hamp_lead, xkey.replace("_lead", ""), xvalue)
        else:
            raise ValueError(f"The key {xkey} does not exit")
    elif isinstance(xkey, tuple) and isinstance(xvalue, tuple):
        for xxkey, xxvalue in zip(xkey, xvalue):
            if xxkey in geop.to_dict().keys():
                setattr(geop, xxkey, xxvalue)
            elif xxkey.endswith("_sys") and xxkey.replace("_sys", "") in hamp_sys.to_dict().keys():
                setattr(hamp_sys, xxkey.replace("_sys", ""), xxvalue)
            elif xxkey.endswith("_lead") and xxkey.replace("_lead", "") in hamp_lead.to_dict().keys():
                setattr(hamp_lead, xxkey.replace("_lead", ""), xxvalue)
            else:
                raise ValueError(f"The key {xxkey} does not exit") 
    else:
        raise ValueError(
            "(xkey,xvalue) should be either (str,numbers) or (tuple,tuple)"
        )

    syst = mktemplate(geop, hamp_sys, hamp_lead)
    idos = get_idos(syst, energy_range)
    del syst
    return idos


def j_at_terminal(syst: kwant.Builder, wf_lead: kwant.wave_function, which_terminal: str)->tuple[np.ndarray, np.ndarray]:
    """Calculate the charge current due to the propagation modes (wf_lead) from a single lead to the terminal defined by which_terminal
    Parameters
    ----------
    syst : `FiniteSystem` instance
    wf_lead : kwant.wave_function from a single lead
    which_terminal : 'lu','ll','ru','rl','ml','mu'

    Returns
    -------
    Raises
    ------
    Notes
    -----
    """
    lx_leg = syst.lx_leg
    ly_leg = syst.ly_leg
    ly_neck = syst.ly_neck
    lx_neck = syst.lx_neck

    def cutpos(which_terminal):
        def left_upper_lead(site_to, site_from):
            return (
                site_from.pos[0] <= 0
                and site_to.pos[0] > 0
                and site_from.pos[1] >= ly_leg + ly_neck
                and site_to.pos[1] >= ly_leg + ly_neck
            )

        def left_lower_lead(site_to, site_from):
            return (
                site_from.pos[0] <= 0
                and site_to.pos[0] > 0
                and site_from.pos[1] < ly_leg
                and site_to.pos[1] < ly_leg
            )

        def right_upper_lead(site_to, site_from):
            return (
                site_from.pos[0] <= lx_leg - 2
                and site_to.pos[0] > lx_leg - 2
                and site_from.pos[1] >= ly_leg + ly_neck
                and site_to.pos[1] >= ly_leg + ly_neck
            )

        def right_lower_lead(site_to, site_from):
            return (
                site_from.pos[0] <= lx_leg - 2
                and site_to.pos[0] > lx_leg - 2
                and site_from.pos[1] < ly_leg
                and site_to.pos[1] < ly_leg
            )

        def middle_lower_lead(site_to, site_from):
            return (
                site_from.pos[1] <= 0
                and site_to.pos[1] > 0
                and site_from.pos[0] >= lx_leg // 2 - lx_neck // 2
                and site_from.pos[0] < lx_leg // 2 + lx_neck // 2
                and site_to.pos[0] >= lx_leg // 2 - lx_neck // 2
                and site_to.pos[0] < lx_leg // 2 + lx_neck // 2
            )

        def middle_upper_lead(site_to, site_from):
            return (
                site_from.pos[1] < ly_leg * 2 + ly_neck - 1
                and site_to.pos[1] >= ly_leg * 2 + ly_neck - 1
                and site_from.pos[0] >= lx_leg // 2 - lx_neck // 2
                and site_from.pos[0] < lx_leg // 2 + lx_neck // 2
                and site_to.pos[0] >= lx_leg // 2 - lx_neck // 2
                and site_to.pos[0] < lx_leg // 2 + lx_neck // 2
            )

        if which_terminal == "lu":
            return left_upper_lead
        elif which_terminal == "ll":
            return left_lower_lead
        elif which_terminal == "ru":
            return right_upper_lead
        elif which_terminal == "rl":
            return right_lower_lead
        elif which_terminal == "ml":
            return middle_lower_lead
        else:
            return middle_upper_lead

    fsyst = syst.finalized()
    j_operator = kwant.operator.Current(fsyst, where=cutpos(which_terminal), sum=True)
    return (j_operator(mode) for mode in wf_lead)
