import numpy as np
import pytest


from xkwant.batch import (    
    vary_energy_vvector_4t,
    varyx_voltage_4t,
    varyx_rho_j_energy_site,
    varyx_idos,
    vvector_4t,
)
from xkwant.physics import *
from xkwant.templates import (doubledirac_mkhbar_4t, gappeddirac_mkhbar_4t, doublequad_mkhbar_4t, doublerashba_mkhbar_4t, mkhbar_4t, mkhbar_6t)
from xkwant.schemas import GeomParams, HamParams, BundleParams
from xkwant.utils import get_dos, get_dos_kpm, get_idos

class TestTemplates:
    def test_doublequad_mkhbar_4t(self):
        """
        Test constructing the doublequad_mkhbar_4t template
        """
        geop = GeomParams(a=1, lx_leg=30, ly_leg=30, lx_neck=15, ly_neck=15)
        ham_sys = HamParams(mass=0.1, hop=0.3, inv=0.1, hyb=0.1)
        ham_lead = HamParams(mass=0.1, hop=0.3, inv=0.1, hyb=0.1)
        doublequad_mkhbar_4t(geop = geop, hamp_sys = ham_sys, hamp_lead = ham_lead, finalized = True)
    
    def test_doublerashba_mkhbar_4t(self):
        """
        Test constructing the doublerashba_mkhbar_4t template
        """
        geop = GeomParams(a=1, lx_leg=30, ly_leg=30, lx_neck=15, ly_neck=15)
        ham_sys = HamParams(mass= 0.1, hop=0.3, wilson=0.1, soc=0.3, inv=0, hyb=0.1)
        ham_lead = HamParams(mass= 0.1, hop=0.3, wilson=0.1, soc=0.3, inv=0, hyb=0.1)
        doublerashba_mkhbar_4t(geop = geop, hamp_sys = ham_sys, hamp_lead = ham_lead, finalized = True)

    def test_doubledirac_mkhbar_4t(self):
        """
        Test constructing the doubledirac_mkhbar_4t template
        """
        geop = GeomParams(a=1, lx_leg=30, ly_leg=30, lx_neck=15, ly_neck=15)
        ham_sys = HamParams(wilson=0.1, soc=0.3, inv=0.1, hyb=0.1)
        ham_lead = HamParams(wilson=0.1, soc=0.3, inv=0.1, hyb=0.1)
        doubledirac_mkhbar_4t(geop = geop, hamp_sys = ham_sys, hamp_lead = ham_lead, finalized = True)


    def test_gappeddirac_mkhbar_4t(self):
        """
        Test contructing the gappeddirac_mkhbar_4t template
        """
        geop = GeomParams(a=1, lx_leg=30, ly_leg=30, lx_neck=15, ly_neck=15)
        ham_sys = HamParams(wilson=0.1, soc=0.3, gapped=0.1)
        ham_lead = HamParams(wilson=0.1, soc=0.3, gapped=0.1)
        gappeddirac_mkhbar_4t(geop = geop, hamp_sys = ham_sys, hamp_lead = ham_lead, finalized = True)

class TestHbar:
    def test_hbar_from_continuum_model(self):
        from xkwant.device import Hbar

        geop = GeomParams(a=1, lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
        bhz_continuum = """
            + mu * kron(sigma_0, sigma_0)
            + M * kron(sigma_0, sigma_z)
            - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z) - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
            + A * (k_x * kron(sigma_z, sigma_x) + k_y * kron(sigma_0, sigma_y))
        """
        hbar_from_continuum_model = Hbar(geop)
        ham_params = dict(A=0.09, B=-0.18, D=-0.065, M=-0.02, mu=0)
        lead_params = ham_params
        hbar_from_continuum_model.build_byfill(bhz_continuum, ham_params)
        for pos in ["bl", "br", "tl", "tr"]:
            hbar_from_continuum_model.attach_lead_byfill(bhz_continuum, lead_params, pos)
        hbar_from_continuum_model.set_ham_params(ham_params)



class TestBatch:
    Iin = 10e-9  # A
    geop = GeomParams(a=1, lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
    ham_sys = HamParams(hop=0.1, soc=0.1, inv=0.1, hyb=0.1)
    ham_lead = HamParams(hop=0.1, soc=0.1, inv=0.1, hyb=0.1)
    energy_range = np.arange(0, 0.15, 0.03)
    hbar = mkhbar_4t(geop=geop, hamp_sys=ham_sys, hamp_lead=ham_lead, finalized=False)

    def test_vvector_4t(self):
        _ = vvector_4t(
            self.hbar,
            energy=0.1,
            ivector=[0, 0, self.Iin, -self.Iin],
        )

    def test_varyx_voltage_4t(self):

        _ = varyx_voltage_4t(
            mktemplate = mkhbar_4t,
            geop = self.geop,
            hamp_sys = self.ham_sys,
            hamp_lead = self.ham_lead,
            xkey = "soc_sys",
            xvalue = 0.1,
            energy = 0.2,
        )

        _ = varyx_voltage_4t(
            mktemplate = mkhbar_4t,
            geop = self.geop,
            hamp_sys = self.ham_sys,
            hamp_lead = self.ham_lead,
            xkey = ("soc_sys", "soc_lead"),
            xvalue = (0.1, 0.1),
            energy = 0.2,
        )

    
    def test_varyx_rho_j_energy_site(self):
        _ = varyx_rho_j_energy_site(
            mktemplate = mkhbar_4t,
            geop = self.geop,
            hamp_sys = self.ham_sys,
            hamp_lead = self.ham_lead,
            xkey = "soc_sys",
            xvalue = 0.1,
            energy = 0.2,
        )

        _ = varyx_rho_j_energy_site(
            mktemplate = mkhbar_4t,
            geop = self.geop,
            hamp_sys = self.ham_sys,
            hamp_lead = self.ham_lead,
            xkey = ("soc_sys", "soc_lead"),
            xvalue = (0.1, 0.1),
            energy = 0.2
        )

    def test_varyx_idos(self):
        _ = varyx_idos(
            mktemplate = mkhbar_4t,
            geop = self.geop,
            hamp_sys = self.ham_sys,
            hamp_lead = self.ham_lead,
            xkey = "soc_sys",
            xvalue = 0.1,
            energy_range = self.energy_range
        )

        _ = varyx_idos(
            mktemplate = mkhbar_4t,
            geop = self.geop,
            hamp_sys = self.ham_sys,
            hamp_lead = self.ham_lead,
            xkey = ("soc_sys", "soc_lead"),
            xvalue = (0.1, 0.1),
            energy_range = self.energy_range
        )
    
    def test_vary_energy_vvector_4t(self):
        _ = vary_energy_vvector_4t(
                mkhbar_4t(geop=self.geop, hamp_sys=self.ham_sys, hamp_lead=self.ham_lead, finalized=False),
                energies=self.energy_range,
                ivector=[0, 0, self.Iin, -self.Iin],
            )
    


class TestUtils:

    def test_get_dos_kpm(self):
        geop = GeomParams(a=1, lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
        ham_sys = HamParams(hop=0.1, soc=0.1, inv=0.1, hyb=0.1)
        ham_lead = HamParams(hop=0.1, soc=0.1, inv=0.1, hyb=0.1)

        hbar = mkhbar_4t(geop, ham_sys, ham_lead, False)
        energy_resolution = 0.6
        energy_range = np.arange(0, 6, energy_resolution)
        _ = get_dos(hbar, energy_range)
        _ = get_dos_kpm(hbar, energy_resolution=energy_resolution)


    def test_get_idos(self):
        N1, L = 36, 90
        # core parameters
        geop = GeomParams(
            a=L / N1,
            lx_leg=int(N1),
            ly_leg=int(N1 / 6),
            lx_neck=int(N1 / 6),
            ly_neck=int(N1 / 6),
        )
        hamp_sys = HamParams(hop=0.1, soc=0.1, inv=0.1, hyb=0.1)
        hamp_lead = HamParams(hop=0.1, soc=0.1, inv=0.1, hyb=0.1)

        hbar = mkhbar_4t(geop, hamp_sys, hamp_lead, finalized=False)
        energy_range = np.arange(0, 0.6, 0.01)
        _ = get_idos(hbar, energy_range, use_kpm=True)
        _ = get_idos(hbar, energy_range, use_kpm=False)


