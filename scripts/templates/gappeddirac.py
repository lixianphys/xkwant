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
from xkwant.schemas import HamParams, GeomParams
DEFAULT_CMAP = "jet"

@log_function_call
def main(
    syst,
    densities=np.arange(0.001, 0.01, 0.001),
    lambda_val=300,
    idos_energy_range=np.arange(0, 0.1, 10),
    Iin=1e-9,
    idos_kpm=False,
    plot_local_quantity=False,
    plot_single_lead=True,
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

    # Calculate more local quantities and plot them separately for each density
    if plot_local_quantity:
        for i, (energy, voltages) in enumerate(zip(target_energies, voltages_list)):
            rho_site, J_site = rho_j_energy_site(syst, energy)
            fsyst = syst.finalized()
            print(f"hamp_sys:{hamp_sys}")
            if rho_site is not None:
                total_modes = len(rho_site[0])
                print(
                    f"At $density$={densities[i]:0.5f}, the number of modes:{total_modes}, the energy is {energy:0.5f}"
                )
                fig, axs = prepare_plot(
                    xlabel="Density [10$^{11}$ cm$^{-2}$]",
                    xlim=(min(densities) * 1e3 - 1, max(densities) * 1e3 + 1),
                    ylabel2="$\Delta V_{12}(\lambda)$ [$\mu$V]",
                    figsize=(10, 6),
                )
                kwant.plotter.density(
                    fsyst,
                    np.array(
                        sum(
                            rho_site[0][mode_num]
                            + rho_site[1][mode_num]
                            + rho_site[2][mode_num]
                            + rho_site[3][mode_num]
                            for mode_num in range(total_modes)
                        )
                    ),
                    ax=axs[0, 1],
                    cmap=DEFAULT_CMAP,
                )
                if plot_single_lead:
                    kwant.plotter.current(
                        fsyst,
                        np.array(
                            sum(J_site[3][mode_num] for mode_num in range(total_modes))
                        ),
                        ax=axs[1, 1],
                        cmap=DEFAULT_CMAP,
                        linecolor="w",
                    )  # electron flow from this lead (grounded) to others
                else:
                    kwant.plotter.current(
                        fsyst,
                        sum(
                            J_site[0][mode_num] * voltages[0]
                            + J_site[1][mode_num] * voltages[1]
                            + J_site[2][mode_num] * voltages[2]
                            for mode_num in range(total_modes)
                        ),
                        ax=axs[1, 1],
                        cmap=DEFAULT_CMAP,
                        linecolor="w",
                    )
                x = [density * 1e3 for density in densities]
                axs[0, 0].plot(x, deltaV12_inmuV)
                axs[0, 0].scatter(x, deltaV12_inmuV)
                axs[0, 0].scatter(
                    densities[i] * 1e3, deltaV12_inmuV[i], color="red", s=100
                )

                axs[1, 0].plot(x, deltaV34_inmuV)
                axs[1, 0].scatter(x, deltaV34_inmuV)
                axs[1, 0].scatter(
                    densities[i] * 1e3, deltaV34_inmuV[i], color="red", s=100
                )

                plt.savefig(
                    os.path.join(
                        savepath,
                        f"dirac_lamb_{lambda_val}_density_{densities[i]:0.5f}.png",
                    )
                )
                plt.close(fig)  # to avoid 'figure.max_open_warning'

            else:
                print(
                    f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed"
                )
    return densities, deltaV12_inmuV, deltaV34_inmuV, idos, idos_energy_range


if __name__ == "__main__":

    from datetime import datetime

    lambda_val = 0.28
    densities = np.arange(0.001, 0.009, 0.0005)
    idos_energy_range = np.arange(0, 0.2, 0.002)
    plot_local_quantity = False
    plot_single_lead = True
    idos_kpm = False
    Iin = 10e-9  # A
    # grid parameters
    N1 = 300
    L = N1 * LATTICE_CONST_HGTE
    # core parameters
    geop = GeomParams(
        a=L / N1,
        lx_leg=int(N1),
        ly_leg=int(N1 / 6),
        lx_neck=int(N1 / 6),
        ly_neck=int(N1 / 6),
    )
    for gap in [0, 0.01]:
        try:
            hamp_sys = HamParams(hop=0, wilson=0.1, soc=lambda_val, gapped=gap, mass=0.05, wdis=0)
            hamp_lead = HamParams(hop=0, wilson=0.1, soc=lambda_val, gapped=gap, mass=0.05)
            syst = gappeddirac_mkhbar_4t(
                geop, hamp_sys, hamp_lead
            )  # This system won't be changed anymore

            vd_d, vd_v12, vd_v34, idos, idos_energy_range = main(
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
                f"data/gappeddirac_data/dt_{os.path.basename(__file__)}_gap_{gap}_{timestamp}.pkl",
                "wb",
            ) as f:
                pickle.dump(data, f)
        except ValueError as e:
            print(f"Calculation for gap={gap} failed, but continue..")
            continue

