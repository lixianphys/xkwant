import kwant
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xkwant.batch import *
from xkwant.templates import *
from xkwant.physics import *
from xkwant.utils import *
from xkwant.log import log_function_call
from xkwant.config import DEFAULT_CMAP


@log_function_call
def main(
    syst,
    densities=np.arange(0.001, 0.01, 0.001),
    idos_energy_range=np.arange(0, 0.1, 10),
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

    # rvl_l, rvl_v12, rvl_v34 = rashba_vary_lambda(lamd = np.arange(0,320,30),single_lead_current=True,target_density = 0.01,savepath='plots/rashba_vary_lambda')
    densities = np.arange(0.001, 0.009, 0.001)
    Iin = 10e-9  # A
    # grid parameters
    N1, L = 200, 200 * 0.646
    # core parameters
    geop = dict(
        a=L / N1,
        lx_leg=int(N1),
        ly_leg=int(N1 / 6),
        lx_neck=int(N1 / 6),
        ly_neck=int(N1 / 6),
    )
    hamp_sys = dict(ws=0.1, vs=0.3, invs=0, hybs=0.05)
    hamp_lead = dict(wl=0.1, vl=0.3, invl=0, hybl=0.05)
    syst = doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
    idos_energy_range = np.arange(0, 0.1, 0.001)
    idos_kpm = True

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
    with open(f"data_{os.path.basename(__file__)}_{timestamp}.pkl", "wb") as f:
        pickle.dump(data, f)

    kwant.plotter.bands(syst.finalized().leads[0])
    plt.show()
    plt.plot(data["densities"], data["voltage_V12"])
    plt.show()
    plt.plot(data["densities"], data["voltage_V34"])
    plt.show()
    plt.plot(data["idos_energy_range"], data["idos"])
    plt.show()
