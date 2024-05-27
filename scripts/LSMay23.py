# For large structures
import kwant
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to sys.path
sys.path.append(parent_dir)


from xkwant.batch import *
from xkwant.templates import *
from xkwant.physics import *
from xkwant.utils import *
from xkwant.log import log_function_call


@log_function_call
def dirac_vary_density(
    densities=np.arange(0.001, 0.01, 0.001),
    lamb=300,
    single_lead_current=False,
    savepath=None,
):
    if savepath is None:
        savepath = os.getcwd()
    lamb = lamb / 1e3
    Iin = 10e-9  # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1, L = 3600, 9000
    # core parameters
    geop = dict(
        lx_leg=int(N1), ly_leg=int(N1 / 6), lx_neck=int(N1 / 6), ly_neck=int(N1 / 6)
    )
    hamp_sys = dict(ts=0, ws=lamb / 3, vs=lamb, ms=0.05, Wdis=0, a=L / N1)
    hamp_lead = dict(tl=0, wl=lamb / 3, vl=lamb, ml=0.05)
    syst = mkhbar_4t(geop, hamp_sys, hamp_lead)  # This system won't be changed anymore

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.arange(0, 0.06, 0.0001)
    target_energies = [
        density_to_energy(*get_idos(syst, energy_range), target_density)
        for target_density in densities
    ]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [
        vvector_4t(syst, energy, [0, 0, Iin, -Iin]) for energy in target_energies
    ]
    deltaV12_inmuV = [(volts[0] - volts[1]) * 1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2] - volts[3]) * 1e6 for volts in voltages_list]

    # Calculate more local quantities and plot them separately for each density
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
                cmap="jet",
            )
            if single_lead_current:
                kwant.plotter.current(
                    fsyst,
                    np.array(
                        sum(J_site[3][mode_num] for mode_num in range(total_modes))
                    ),
                    ax=axs[1, 1],
                    cmap="jet",
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
                    cmap="jet",
                    linecolor="w",
                )
            x = [density * 1e3 for density in densities]
            axs[0, 0].plot(x, deltaV12_inmuV)
            axs[0, 0].scatter(x, deltaV12_inmuV)
            axs[0, 0].scatter(densities[i] * 1e3, deltaV12_inmuV[i], color="red", s=100)

            axs[1, 0].plot(x, deltaV34_inmuV)
            axs[1, 0].scatter(x, deltaV34_inmuV)
            axs[1, 0].scatter(densities[i] * 1e3, deltaV34_inmuV[i], color="red", s=100)

            plt.savefig(
                os.path.join(
                    savepath, f"dirac_lambda{lamb*1e3}_density_{densities[i]:0.5f}.png"
                )
            )
            plt.close(fig)  # to avoid 'figure.max_open_warning'

        else:
            print(
                f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed"
            )

    return densities, deltaV12_inmuV, deltaV34_inmuV


if __name__ == "__main__":
    dvdr_d, dvdr_v12, dvdr_v34 = dirac_vary_density(
        densities=np.arange(0.001, 0.011, 0.001),
        lamb=300,
        single_lead_current=True,
        savepath=os.path.join(parent_dir, "plots/lls_dirac_vary_density"),
    )
