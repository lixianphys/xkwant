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


def rashba_lambda(
    syst,
    target_density=0.01,
    idos_energy_range=np.arange(0, 0.1, 10),
    Iin=10e-9,
    idos_kpm=False,
):
    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    idos, idos_energy_range = get_idos(syst, idos_energy_range, use_kpm=idos_kpm)

    target_energy = density_to_energy(idos, idos_energy_range, target_density)
    # calculate terminal voltages in a 4 terminal hbar
    volts = vvector_4t(syst, target_energy, [0, 0, Iin, -Iin])
    deltaV12_inmuV = (volts[0] - volts[1]) * 1e6
    deltaV34_inmuV = (volts[2] - volts[3]) * 1e6

    return deltaV12_inmuV, deltaV34_inmuV


if __name__ == "__main__":

    from datetime import datetime

    lambda_list = np.arange(0, 90, 10)
    dV12_list = []
    dV34_list = []
    idos_energy_range = np.arange(0, 0.1, 0.001)
    target_density = 0.01
    # grid parameters
    N1 = 36  # the number of lattices in the longitudinal direction
    L = 90
    Iin = 10e-9
    idos_kpm = False
    # core parameters
    geop = dict(
        a=L / N1,
        lx_leg=int(N1),
        ly_leg=int(N1 / 6),
        lx_neck=int(N1 / 6),
        ly_neck=int(N1 / 6),
    )
    for lambda_val in lambda_list:
        print(f"Start calculation for Lambda = {lambda_val} meV")

        hamp_sys = dict(ts=tk, ws=0, vs=lambda_val / 1e3, ms=0.05, Wdis=0)
        hamp_lead = dict(tl=tk, wl=0, vl=lambda_val / 1e3, ml=0.05)

        syst = mkhbar_4t(geop, hamp_sys, hamp_lead)

        deltaV12_inmuV, deltaV34_inmuV = rashba_lambda(
            syst,
            target_density=target_density,
            idos_energy_range=idos_energy_range,
            Iin=Iin,
            idos_kpm=idos_kpm,
        )
        dV12_list.append(deltaV12_inmuV)
        dV34_list.append(deltaV34_inmuV)

    data = {
        "lambda_list": lambda_list,
        "idos_energy_range": idos_energy_range,
        "Iin": Iin,
        "idos_kpm": idos_kpm,
        "N1": N1,
        "L": L,
        "geometric_params": geop,
        "hamiltonian_params_sys": hamp_sys,
        "hamiltonian_params_lead": hamp_lead,
        "voltage_V12": dV12_list,
        "voltage_V34": dV34_list,
    }
    plt.plot(lambda_list, dV12_list)
    plt.show()

    # now = datetime.now()
    # timestamp = now.strftime("%Y%m%d_%H%M")
    # with open(f"data_{os.path.basename(__file__)}_{timestamp}.pkl", "wb") as f:
    #     pickle.dump(data, f)
