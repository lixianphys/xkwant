import os
import pickle
import numpy as np
from xkwant.batch import *
from xkwant.templates import *
from xkwant.physics import *
from xkwant.utils import *
from xkwant.log import log_function_call
from xkwant.config import LATTICE_CONST_HGTE


@log_function_call
def main(
    syst,
    densities=np.arange(0.001, 0.01, 0.001),
    idos_energy_range=np.arange(0, 0.1, 0.001),
    Iin=1e-9,
    idos_kpm=False,
    savepath=None,
):
    if savepath is None:
        savepath = os.getcwd()
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    idos, idos_energy_range = get_idos(syst, idos_energy_range, use_kpm=idos_kpm)

    target_energies = [
        density_to_energy(idos, idos_energy_range, target_density)
        for target_density in densities
    ]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [
        vvector_4t(syst, energy, [0, 0, Iin, -Iin]) for energy in target_energies
    ]
    deltaV12_inmuV = [(volts[0] - volts[1]) * 1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2] - volts[3]) * 1e6 for volts in voltages_list]

    return densities, deltaV12_inmuV, deltaV34_inmuV, idos, idos_energy_range


if __name__ == "__main__":

    from datetime import datetime

    densities = np.arange(0.001, 0.009, 0.0002)
    idos_energy_range = np.arange(0, 0.2, 0.002)
    Iin = 10e-9  # A
    # grid parameters
    N1 = 1000  # the number of lattices in the longitudinal direction
    L = N1 * LATTICE_CONST_HGTE
    idos_kpm = False
    # core parameters
    geop = dict(
        a=L / N1,
        lx_leg=int(N1),
        ly_leg=int(N1 / 6),
        lx_neck=int(N1 / 6),
        ly_neck=int(N1 / 6),
    )

    einv = 0
    for ehyb in np.arange(0, 0.04, 0.01):
        try:
            hamp_sys = dict(
                ws=0.1, vs=0.28, invs=einv, hybs=ehyb
            )  # hbar*vf = 280 meV nm and inversion-symmetry breaking term = 4.2 meV (From SM, PRL 106, 126803 (2011) )
            hamp_lead = dict(wl=0.1, vl=0.28, invl=einv, hybl=ehyb)
            syst = test_doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)

            vd_d, vd_v12, vd_v34, idos, idos_energy_range = main(
                syst,
                densities=densities,
                idos_energy_range=idos_energy_range,
                Iin=Iin,
                idos_kpm=idos_kpm,
                # savepath="plots/rashba_vary_density",
            )

            data = {
                "densities": densities,
                "idos": idos,
                "idos_energy_range": idos_energy_range,
                "Iin": Iin,
                "idos_kpm": idos_kpm,
                "N1": N1,
                "L": L,
                "geometric_params": geop,
                "hamiltonian_params_sys": hamp_sys,
                "hamiltonian_params_lead": hamp_lead,
                "voltage_V12": vd_v12,
                "voltage_V34": vd_v34,
            }

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            with open(
                f"dt_{os.path.basename(__file__)}_ei_{einv}_eh_{ehyb}_{timestamp}.pkl",
                "wb",
            ) as f:
                pickle.dump(data, f)
        except ValueError as e:
            print(
                f"Calculations for einv={einv} and ehyb={ehyb} failed, but continue.."
            )
            continue
        except IndexError as e:
            print(
                f"Calculations for einv={einv} and ehyb={ehyb} failed, but continue.."
            )
            continue
