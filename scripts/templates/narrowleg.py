import os
import pickle
import numpy as np
from xkwant.batch import *
from xkwant.templates import *
from xkwant.physics import *
from xkwant.utils import *
from dirachbar import main as dirachbar_main
from xkwant.schemas import HamParams, GeomParams
if __name__ == "__main__":

    from datetime import datetime

    densities = np.arange(0.001, 0.009, 0.0002)
    idos_energy_range = np.arange(0, 0.3, 0.0001)
    lambda_val = 280
    plot_local_quantity = False
    plot_single_lead = True
    Iin = 10e-9  # A

    # hamp_sys = dict(ts=0, ws=lambda_val / 3e3, vs=lambda_val / 1e3, ms=0.05, Wdis=0)
    # hamp_lead = dict(tl=0, wl=lambda_val / 3e3, vl=lambda_val / 1e3, ml=0.05)

    hamp_sys = HamParams(hop=tk, wilson=0, soc=0, mass=0.05, wdis=0)
    hamp_lead = HamParams(hop=tk, wilson=0, soc=0, mass=0.05)
    idos_kpm = False

    # grid parameters
    for N1 in np.concatenate([np.arange(30, 100, 10), np.arange(100, 500, 100)]):
        try:
            L = N1 * LATTICE_CONST_HGTE
            # core parameters
            geop = GeomParams(
                a=L / N1,
                lx_leg=int(N1),
                ly_leg=int(60 / 6),  # fix the width of the leg
                lx_neck=int(N1 / 6),
                ly_neck=int(N1 / 6),
            )
            syst = mkhbar_4t(
                geop, hamp_sys, hamp_lead
            )  # This system won't be changed anymore
            vd_d, vd_v12, vd_v34, idos, idos_energy_range = dirachbar_main(
                syst,
                densities=densities,
                lambda_val=lambda_val,
                idos_energy_range=idos_energy_range,
                Iin=Iin,
                idos_kpm=idos_kpm,
                plot_local_quantity=plot_local_quantity,
                plot_single_lead=plot_single_lead,
                # savepath="plots/rashba_vary_density",
            )

            data = {
                "densities": densities,
                "lambda_val": lambda_val,
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
                f"data/narrowleg_data/quad_forcomparison/dt_{os.path.basename(__file__)}_N1_{N1}_{timestamp}.pkl",
                "wb",
            ) as f:
                pickle.dump(data, f)
            print(f"N1={N1} is done")
        except ValueError as e:
            print(f"Calculation for N1={N1} failed, but continue..")
            continue
