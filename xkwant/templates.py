"""
This module contains the templates for building Hbar-shaped devices with various Hamiltonians.
"""
import kwant
import random
import numpy as np
from typing import Union
from xkwant.physics import *
from xkwant.schemas import GeomParams, HamParams

__all__ = [
    "doublerashba_mkhbar_4t",
    "doublequad_mkhbar_4t",
    "doubledirac_mkhbar_4t",
    "gappeddirac_mkhbar_4t",
    "mkhbar_4t",
    "mkhbar_6t",
]


def doublerashba_mkhbar_4t(
    geop: GeomParams, hamp_sys: HamParams, hamp_lead: HamParams=None, finalized:bool=False, conservation_law=None
) -> Union[kwant.Builder, kwant.Builder.finalized]:
    """
    Constructs the system consisting of two identical Rashba surface states and includes inversion symmetry breaking term and hybridization term.

    Parameters
    ----------
    geop: GeomParams
        Geometric parameters for the scattering region
    hamp_sys: HamParams
        Hamiltonian parameters for the scattering region
    hamp_lead: HamParams
        Hamiltonian parameters for the leads
    finalized: bool
        Whether to finalize the system
    conservation_law:
        Conservation law for the system
    """
    from .device import Hbar
    if hamp_lead is None:
        hamp_lead = hamp_sys

    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    ts, tl = hamp_sys.hop, hamp_lead.hop
    ms, ml = hamp_sys.mass, hamp_lead.mass
    ws, wl = hamp_sys.wilson, hamp_lead.wilson
    vs, vl = hamp_sys.soc, hamp_lead.soc

    invs, invl = (
        hamp_sys.inv,
        hamp_lead.inv,
    )  # this term is inversion breaking term.
    hybs, hybl = (
        hamp_sys.hyb,
        hamp_lead.hyb,
    )

    # Geometric parameters for the scattering region
    lx_leg = geop.lx_leg
    ly_leg = geop.ly_leg
    lx_neck = geop.lx_neck
    ly_neck = geop.ly_neck
    a = geop.a  # lattice constant a

    lat = kwant.lattice.square(a, norbs=4)
    tau_0 = s_0
    tau_x = s_x
    tau_z = s_z

    def hop_x(site1, site2):
        return 1j * vs / (2 * a) * np.kron(tau_z, s_y) - ws * np.kron(tau_0, s_z)

    def hop_y(site1, site2):
        return -1j * vs / (2 * a) * np.kron(tau_z, s_x) - ws * np.kron(tau_0, s_z)

    def onsite(site):
        return (
            (4 * ts / (ms * a**2)) * np.kron(tau_0, s_0)
            + 4 * ws * np.kron(tau_0, s_z)
            + invs * np.kron(tau_z, s_0)
            + hybs * np.kron(tau_x, s_x)
        )

    def lead_onsite(site):
        return (
            (4 * tl / (ml * a**2)) * np.kron(tau_0, s_0)
            + 4 * wl * np.kron(tau_0, s_z)
            + invl * np.kron(tau_z, s_0)
            + hybl * np.kron(tau_x, s_x)
        )

    def hop_x(site1, site2):
        return (
            -ts / (ms * a**2) * np.kron(tau_0, s_0)
            + 1j * vs / (2 * a) * np.kron(tau_z, s_y)
            - ws * np.kron(tau_0, s_z)
        )

    def hop_y(site1, site2):
        return (
            -ts / (ms * a**2) * np.kron(tau_0, s_0)
            - 1j * vs / (2 * a) * np.kron(tau_z, s_x)
            - ws * np.kron(tau_0, s_z)
        )

    def lead_hop_x(site1, site2):
        return (
            -tl / (ml * a**2) * np.kron(tau_0, s_0)
            + 1j * vl / (2 * a) * np.kron(tau_z, s_y)
            - wl * np.kron(tau_0, s_z)
        )

    def lead_hop_y(site1, site2):
        return (
            -tl / (ml * a**2) * np.kron(tau_0, s_0)
            - 1j * vl / (2 * a) * np.kron(tau_z, s_x)
            - wl * np.kron(tau_0, s_z)
        )

    # Define Fundamental Domain
    for i in range(lx_leg):
        # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j > 0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):
        # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
            syst[lat(i, j)] = onsite
            if j > ly_leg + ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        # central connecting neck
        for j in range(ly_leg, ly_leg + ly_neck):
            syst[lat(i, j)] = onsite
            if j >= ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > lx_leg // 2 - lx_neck // 2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg + ly_neck), lat(i, ly_leg + ly_neck - 1)] = hop_y

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst


def doublequad_mkhbar_4t(
    geop: GeomParams, hamp_sys: HamParams, hamp_lead: HamParams=None, finalized:bool=False, conservation_law=None
)-> Union[kwant.Builder, kwant.Builder.finalized]:
    """
    Constructs the system consisting of two identical quadratic surface states and includes inversion symmetry breaking term and hybridization term.

    Parameters
    ----------
    geop: GeomParams
        Geometric parameters for the scattering region
    hamp_sys: HamParams
        Hamiltonian parameters for the scattering region
    hamp_lead: HamParams
        Hamiltonian parameters for the leads
    finalized: bool
        Whether to finalize the system
    conservation_law:
        Conservation law for the system
    """
    from .device import Hbar
    if hamp_lead is None:
        hamp_lead = hamp_sys

    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    ts, tl = hamp_sys.hop, hamp_lead.hop
    ms, ml = hamp_sys.mass, hamp_lead.mass

    invs, invl = (
        hamp_sys.inv,
        hamp_lead.inv,
    )  # this term is inversion breaking term.
    hybs, hybl = (
        hamp_sys.hyb,
        hamp_lead.hyb,
    )
    # Geometric parameters for the scattering region
    lx_leg = geop.lx_leg
    ly_leg = geop.ly_leg
    lx_neck = geop.lx_neck
    ly_neck = geop.ly_neck
    a = geop.a  # lattice constant a

    lat = kwant.lattice.square(a, norbs=2)
    tau_0 = s_0
    tau_x = s_x
    tau_z = s_z

    def onsite(site):
        return (4 * ts / (ms * a**2)) * tau_0 + invs * tau_z + hybs * tau_x

    def lead_onsite(site):
        return (4 * tl / (ml * a**2)) * tau_0 + invl * tau_z + hybl * tau_x

    def hop_x(site1, site2):
        return -ts / (ms * a**2) * tau_0

    def hop_y(site1, site2):
        return -ts / (ms * a**2) * tau_0

    def lead_hop_x(site1, site2):
        return -tl / (ml * a**2) * tau_0

    def lead_hop_y(site1, site2):
        return -tl / (ml * a**2) * tau_0

    # Define Fundamental Domain
    for i in range(lx_leg):
        # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j > 0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):
        # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
            syst[lat(i, j)] = onsite
            if j > ly_leg + ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        # central connecting neck
        for j in range(ly_leg, ly_leg + ly_neck):
            syst[lat(i, j)] = onsite
            if j >= ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > lx_leg // 2 - lx_neck // 2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg + ly_neck), lat(i, ly_leg + ly_neck - 1)] = hop_y

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst


def doubledirac_mkhbar_4t(
    geop: GeomParams, hamp_sys: HamParams, hamp_lead: HamParams=None, finalized:bool=False, conservation_law=None
)-> Union[kwant.Builder, kwant.Builder.finalized]:
    """
    Constructs the system consisting of two identical Dirac surface states and includes inversion symmetry breaking term and hybridization term.

    Parameters
    ----------
    geop: GeomParams
        Geometric parameters for the scattering region
    hamp_sys: HamParams
        Hamiltonian parameters for the scattering region
    hamp_lead: HamParams
        Hamiltonian parameters for the leads
    finalized: bool
        Whether to finalize the system
    conservation_law:
        Conservation law for the system
    """
    from .device import Hbar
    if hamp_lead is None:
        hamp_lead = hamp_sys
    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    ws, wl = hamp_sys.wilson, hamp_lead.wilson
    vs, vl = hamp_sys.soc, hamp_lead.soc
    invs, invl = (
        hamp_sys.inv,
        hamp_lead.inv,
    )  # this term is inversion breaking term.
    # hamiltonian parameters only belong to the scattering region: disorder strength Wdis
    hybs, hybl = (
        hamp_sys.hyb,
        hamp_lead.hyb,
    )
    # Geometric parameters for the scattering region
    lx_leg = geop.lx_leg
    ly_leg = geop.ly_leg
    lx_neck = geop.lx_neck
    ly_neck = geop.ly_neck
    a = geop.a  # lattice constant a

    lat = kwant.lattice.square(a, norbs=4)
    tau_0 = s_0
    tau_x = s_x
    tau_z = s_z

    def onsite(site):
        return (
            4 * ws * np.kron(tau_0, s_z)
            + invs * np.kron(tau_z, s_0)
            + hybs * np.kron(tau_x, s_x)
        )

    def lead_onsite(site):
        return (
            4 * wl * np.kron(tau_0, s_z)
            + invl * np.kron(tau_z, s_0)
            + hybl * np.kron(tau_x, s_x)
        )

    def hop_x(site1, site2):
        return 1j * vs / (2 * a) * np.kron(tau_z, s_y) - ws * np.kron(tau_0, s_z)

    def hop_y(site1, site2):
        return -1j * vs / (2 * a) * np.kron(tau_z, s_x) - ws * np.kron(tau_0, s_z)

    def lead_hop_x(site1, site2):
        return +1j * vl / (2 * a) * np.kron(tau_z, s_y) - wl * np.kron(tau_0, s_z)

    def lead_hop_y(site1, site2):
        return -1j * vl / (2 * a) * np.kron(tau_z, s_x) - wl * np.kron(tau_0, s_z)

    # Define Fundamental Domain
    for i in range(lx_leg):
        # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j > 0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):
        # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
            syst[lat(i, j)] = onsite
            if j > ly_leg + ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        # central connecting neck
        for j in range(ly_leg, ly_leg + ly_neck):
            syst[lat(i, j)] = onsite
            if j >= ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > lx_leg // 2 - lx_neck // 2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg + ly_neck), lat(i, ly_leg + ly_neck - 1)] = hop_y

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst


def gappeddirac_mkhbar_4t(
    geop: GeomParams, hamp_sys: HamParams, hamp_lead: HamParams=None, finalized:bool=False, conservation_law=None
)-> Union[kwant.Builder, kwant.Builder.finalized]:
    """
    Constructs the system consisting of one gapped Dirac surface states and includes a gap term "gapped".

    Parameters
    ----------
    geop: GeomParams
        Geometric parameters for the scattering region
    hamp_sys: HamParams
        Hamiltonian parameters for the scattering region
    hamp_lead: HamParams
        Hamiltonian parameters for the leads
    finalized: bool
        Whether to finalize the system
    conservation_law: 
        Conservation law for the system
    """
    from .device import Hbar  # place this inside function to avoid circular import

    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    # Hamiltonian parameters belong to both scattering region and leads
    ws, wl = hamp_sys.wilson, hamp_lead.wilson
    vs, vl = hamp_sys.soc, hamp_lead.soc
    ds, dl = (
        hamp_sys.gapped,
        hamp_lead.gapped,
    )  # this term is the gap of the dirac dispersion at k=0.
    # Geometric parameters for the scattering region
    lx_leg = geop.lx_leg
    ly_leg = geop.ly_leg
    lx_neck = geop.lx_neck
    ly_neck = geop.ly_neck
    a = geop.a  # lattice constant a

    lat = kwant.lattice.square(a, norbs=2)

    def onsite(site):
        rand_num = random.uniform(-1, 1)
        return 4 * ws * s_z + ds * s_z

    def lead_onsite(site):
        return 4 * wl * s_z + dl * s_z

    def hop_x(site1, site2):
        return 1j * vs / (2 * a) * s_y - ws * s_z

    def hop_y(site1, site2):
        return -1j * vs / (2 * a) * s_x - ws * s_z

    def lead_hop_x(site1, site2):
        return 1j * vl / (2 * a) * s_y - wl * s_z

    def lead_hop_y(site1, site2):
        return -1j * vl / (2 * a) * s_x - wl * s_z

    # Define Fundamental Domain
    for i in range(lx_leg):
        # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j > 0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):
        # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
            syst[lat(i, j)] = onsite
            if j > ly_leg + ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        # central connecting neck
        for j in range(ly_leg, ly_leg + ly_neck):
            syst[lat(i, j)] = onsite
            if j >= ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > lx_leg // 2 - lx_neck // 2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg + ly_neck), lat(i, ly_leg + ly_neck - 1)] = hop_y

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst


def mkhbar_4t(geop: GeomParams, hamp_sys: HamParams, hamp_lead: HamParams=None, finalized:bool=False, conservation_law=None)-> Union[kwant.Builder, kwant.Builder.finalized]:
    from .device import Hbar  # place this inside function to avoid circular import
    """
    Constructs a general hbar-shaped system consisting of four terminals.

    Parameters
    ----------
    geop: GeomParams
        Geometric parameters for the scattering region
    hamp_sys: HamParams
        Hamiltonian parameters for the scattering region
    hamp_lead: HamParams
        Hamiltonian parameters for the leads
    finalized: bool
        Whether to finalize the system
    conservation_law:
        Conservation law for the system
    """
    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    # Hamiltonian parameters belong to both scattering region and leads
    ts, tl = hamp_sys.hop, hamp_lead.hop
    ws, wl = hamp_sys.wilson, hamp_lead.wilson
    vs, vl = hamp_sys.soc, hamp_lead.soc
    ms, ml = hamp_sys.mass, hamp_lead.mass
    # hamiltonian parameters only belong to the scattering region: disorder strength Wdis
    Wdis = hamp_sys.wdis
    # Geometric parameters for the scattering region
    lx_leg = geop.lx_leg
    ly_leg = geop.ly_leg
    lx_neck = geop.lx_neck
    ly_neck = geop.ly_neck
    a = geop.a  # lattice constant a

    lat = kwant.lattice.square(a, norbs=2)

    def onsite(site):
        rand_num = random.uniform(-1, 1)
        return (4 * ts / (ms * a**2) + Wdis * rand_num) * s_0 + 4 * ws * s_z

    def lead_onsite(site):
        return (4 * tl / (ms * a**2)) * s_0 + 4 * wl * s_z

    def hop_x(site1, site2):
        return -ts / (ms * a**2) * s_0 + 1j * vs / (2 * a) * s_y - ws * s_z

    def hop_y(site1, site2):
        return -ts / (ms * a**2) * s_0 - 1j * vs / (2 * a) * s_x - ws * s_z

    def lead_hop_x(site1, site2):
        return -tl / (ml * a**2) * s_0 + 1j * vl / (2 * a) * s_y - wl * s_z

    def lead_hop_y(site1, site2):
        return -tl / (ml * a**2) * s_0 - 1j * vl / (2 * a) * s_x - wl * s_z

    # Define Fundamental Domain
    for i in range(lx_leg):
        # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j > 0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):
        # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
            syst[lat(i, j)] = onsite
            if j > ly_leg + ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        # central connecting neck
        for j in range(ly_leg, ly_leg + ly_neck):
            syst[lat(i, j)] = onsite
            if j >= ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > lx_leg // 2 - lx_neck // 2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg + ly_neck), lat(i, ly_leg + ly_neck - 1)] = hop_y

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst


def mkhbar_6t(geop: GeomParams, hamp_sys: HamParams, hamp_lead: HamParams=None, finalized=False)-> Union[kwant.Builder, kwant.Builder.finalized]:
    """Return a hbar-shaped FiniteSystem or Builder() with six terminals"""
    # import a normal 4-terminal system to start with, convervation_law = -s_z to ensure independent spin channels
    syst = mkhbar_4t(geop, hamp_sys, hamp_lead, finalized=False, conservation_law=-s_z)
    tl = hamp_lead.hop
    wl = hamp_lead.wilson
    vl = hamp_lead.soc
    ml = hamp_lead.mass
    a = geop.a
    lx_leg = syst.lx_leg
    lx_neck = syst.lx_neck

    lat = kwant.lattice.square(a, norbs=2)

    def lead_onsite(site):
        return (4 * tl / (ml * a**2)) * s_0 + 4 * wl * s_z

    def lead_hop_x(site1, site2):
        return -tl / (ml * a**2) * s_0 + 1j * vl / (2 * a) * s_y - wl * s_z

    def lead_hop_y(site1, site2):
        return -tl / (ml * a**2) * s_0 - 1j * vl / (2 * a) * s_x - wl * s_z

    sym_upper_lead = kwant.TranslationalSymmetry((0, a))
    sym_bottom_lead = kwant.TranslationalSymmetry((0, -a))

    # Add top leads to measure spin current
    # lead No.4   # top middle
    top_middle_lead = kwant.Builder(sym_upper_lead, conservation_law=-s_z)
    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        top_middle_lead[lat(i, 0)] = lead_onsite
        if i > lx_leg // 2 - lx_neck // 2:
            top_middle_lead[lat(i, 0), lat(i - 1, 0)] = lead_hop_x
        top_middle_lead[lat(i, 1), lat(i, 0)] = lead_hop_y
    syst.attach_lead(top_middle_lead)

    # lead No.5   # bottom middle
    bot_middle_lead = kwant.Builder(sym_bottom_lead, conservation_law=-s_z)
    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        bot_middle_lead[lat(i, 0)] = lead_onsite
        if i > lx_leg // 2 - lx_neck // 2:
            bot_middle_lead[lat(i, 0), lat(i - 1, 0)] = lead_hop_x
        bot_middle_lead[lat(i, 1), lat(i, 0)] = lead_hop_y
    syst.attach_lead(bot_middle_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst


# Integrate B field into the Hamiltonian
def mkhbar_4t_magf(
    geop, hamp_sys, hamp_lead=None, finalized=False, conservation_law=None
):
    from .device import Hbar  # place this inside function to avoid circular import

    """Return a hbar-shaped FiniteSystem or Builder() with four terminals"""
    syst = Hbar(geop)
    syst.set_ham_params(hamp_sys)
    # Hamiltonian parameters belong to both scattering region and leads
    ts, tl = hamp_sys["ts"], hamp_lead["tl"]
    ws, wl = hamp_sys["ws"], hamp_lead["wl"]
    vs, vl = hamp_sys["vs"], hamp_lead["vl"]
    ms, ml = hamp_sys["ms"], hamp_lead["ml"]
    # hamiltonian parameters only belong to the scattering region: lattice constant a, disorder strength Wdis
    Wdis = hamp_sys["Wdis"]
    # Geometric parameters for the scattering region
    a = geop["a"]
    lx_leg = geop["lx_leg"]
    ly_leg = geop["ly_leg"]
    lx_neck = geop["lx_neck"]
    ly_neck = geop["ly_neck"]

    lat = kwant.lattice.square(a, norbs=2)

    def onsite(site):  #  mu is the chem pot of the system
        rand_num = random.uniform(-1, 1)
        return (4 * ts / (ms * a**2) + Wdis * rand_num) * s_0 + 4 * ws * s_z

    def lead_onsite(site):
        return (4 * tl / (ms * a**2)) * s_0 + 4 * wl * s_z

    def hop_x(site1, site2, B):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        phase = np.exp(-1j * B * (y1 + y2) / 2)
        return phase * (-ts / (ms * a**2) * s_0 + 1j * vs / (2 * a) * s_y) - ws * s_z

    def hop_y(site1, site2, B):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        phase = np.exp(-1j * B * (x1 + x2) / 2)
        return phase * (-ts / (ms * a**2) * s_0 - 1j * vs / (2 * a) * s_x) - ws * s_z

    def lead_hop_x(site1, site2, B):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        phase = np.exp(-1j * B * (y1 + y2) / 2)
        return phase * (-tl / (ml * a**2) * s_0 + 1j * vl / (2 * a) * s_y) - wl * s_z

    def lead_hop_y(site1, site2, B):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        phase = np.exp(-1j * B * (x1 + x2) / 2)
        return phase * (-tl / (ml * a**2) * s_0 - 1j * vl / (2 * a) * s_x) - wl * s_z

    # Define Fundamental Domain
    for i in range(lx_leg):
        # bottom horizontal leg:  lx_leg in x direction, ly_leg in y direction, (0,0) site at its left-bottom corner
        for j in range(ly_leg):
            # On-site Hamiltonian
            syst[lat(i, j)] = onsite
            if j > 0:
                # Hopping in y-direction
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                # Hopping in x-direction
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg):
        # top horizontal leg:  xlen_leg in x direction, ly_leg in y direction
        for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
            syst[lat(i, j)] = onsite
            if j > ly_leg + ly_neck:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = hop_x

    for i in range(lx_leg // 2 - lx_neck // 2, lx_leg // 2 + lx_neck // 2):
        # central connecting neck
        for j in range(ly_leg, ly_leg + ly_neck):
            syst[lat(i, j)] = onsite
            if j >= ly_leg:
                syst[lat(i, j), lat(i, j - 1)] = hop_y
            if i > lx_leg // 2 - lx_neck // 2:
                syst[lat(i, j), lat(i - 1, j)] = hop_x
        syst[lat(i, ly_leg + ly_neck), lat(i, ly_leg + ly_neck - 1)] = hop_y

    # Define Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))

    # lead No. 0   # bottom left
    bot_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_left_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_left_lead)

    # lead No. 1   # bottom right
    bot_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg):
        bot_right_lead[lat(0, j)] = lead_onsite
        if j > 0:
            bot_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        bot_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(bot_right_lead)

    # lead No. 2   # top left
    top_left_lead = kwant.Builder(sym_left_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_left_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_left_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_left_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_left_lead)

    # lead No. 3   # top right
    top_right_lead = kwant.Builder(sym_right_lead, conservation_law=conservation_law)
    for j in range(ly_leg + ly_neck, ly_leg * 2 + ly_neck):
        top_right_lead[lat(0, j)] = lead_onsite
        if j > ly_leg + ly_neck:
            top_right_lead[lat(0, j), lat(0, j - 1)] = lead_hop_y
        top_right_lead[lat(1, j), lat(0, j)] = lead_hop_x
    syst.attach_lead(top_right_lead)

    if finalized:
        return syst.finalized()
    else:
        return syst
