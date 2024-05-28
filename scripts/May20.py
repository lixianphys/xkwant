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


@log_function_call
def dirac_vary_lambda(
    lamd=np.linspace(4, 80, 20),
    single_lead_current=False,
    target_density=0.01,
    savepath=None,
):
    """A copy from rashba_vary_lambda, but instead using a pure Dirac-type Hamiltonian defined by lambda"""
    if savepath is None:
        savepath = os.getcwd()
    Iin = 10e-9  # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1, L = 36, 90
    # core parameters
    geop = dict(
        lx_leg=int(N1), ly_leg=int(N1 / 6), lx_neck=int(N1 / 6), ly_neck=int(N1 / 6)
    )
    hamp_sys = dict(ts=0, ws=0.1 / 3, vs=0.1, ms=0.05, Wdis=0, a=L / N1)
    hamp_lead = dict(tl=0, wl=0.1 / 3, vl=0.1, ml=0.05)

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.arange(0, 0.06, 0.0001)
    target_energies = [
        density_to_energy(
            *varyx_idos(
                mkhbar_4t,
                geop,
                hamp_sys,
                hamp_lead,
                ("vs", "vl", "ws", "wl"),
                (xvalue, xvalue, xvalue / 3, xvalue / 3),
                energy_range,
            ),
            target_density,
        )
        for xvalue in (la / 1e3 for la in lamd)
    ]
    # calculate terminal voltages in a 4 terminal hbar
    voltages_list = [
        varyx_voltage_4t(
            mkhbar_4t,
            geop,
            hamp_sys,
            hamp_lead,
            ("vs", "vl", "ws", "wl"),
            (xvalue, xvalue, xvalue / 3, xvalue / 3),
            energy,
            [0, 0, Iin, -Iin],
        )
        for xvalue, energy in zip((la / 1e3 for la in lamd), target_energies)
    ]
    deltaV12_inmuV = [(volts[0] - volts[1]) * 1e6 for volts in voltages_list]
    deltaV34_inmuV = [(volts[2] - volts[3]) * 1e6 for volts in voltages_list]

    # Calculate more local quantities and plot them separately for each \lambda value
    for i, (xvalue, energy, voltages) in enumerate(
        zip((la / 1e3 for la in lamd), target_energies, voltages_list)
    ):
        rho_site, J_site = varyx_rho_j_energy_site(
            mkhbar_4t,
            geop,
            hamp_sys,
            hamp_lead,
            ("vs", "vl", "ws", "wl"),
            (xvalue, xvalue, xvalue / 3, xvalue / 3),
            energy,
        )
        sys_dirac = mkhbar_4t(geop, hamp_sys, hamp_lead).finalized()
        print(f"hamp_sys:{hamp_sys}")
        if rho_site is not None:
            total_modes = len(rho_site[0])
            print(
                f"At $\\lambda$={xvalue*1e3}, the number of modes:{total_modes}, the energy is {energy:0.5f}"
            )
            fig, axs = prepare_plot(
                xlabel="$\\lambda$ [meV nm]",
                xlim=(min(lamd) - 1, max(lamd) + 1),
                ylabel2="$\\Delta V_{12}(\\lambda)$ [$\\mu$V]",
                figsize=(10, 6),
            )
            kwant.plotter.density(
                sys_dirac,
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
                    sys_dirac,
                    np.array(
                        sum(J_site[3][mode_num] for mode_num in range(total_modes))
                    ),
                    ax=axs[1, 1],
                    cmap="jet",
                    linecolor="w",
                )  # electron flow from this lead (grounded) to others
            else:
                kwant.plotter.current(
                    sys_dirac,
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
            axs[0, 0].plot(lamd, deltaV12_inmuV)
            axs[0, 0].scatter(lamd, deltaV12_inmuV)
            axs[0, 0].scatter(xvalue * 1e3, deltaV12_inmuV[i], color="red", s=100)

            axs[1, 0].plot(lamd, deltaV34_inmuV)
            axs[1, 0].scatter(lamd, deltaV34_inmuV)
            axs[1, 0].scatter(xvalue * 1e3, deltaV34_inmuV[i], color="red", s=100)

            plt.savefig(
                os.path.join(
                    savepath, f"dirac_lambda_{xvalue*1e3}_density_{target_density}.png"
                )
            )
            plt.close(fig)  # to avoid 'figure.max_open_warning'

        else:
            print(
                f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed"
            )

    return lamd, deltaV12_inmuV, deltaV34_inmuV


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
    N1, L = 36, 90
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


@log_function_call
def quad_vary_density(
    densities=np.arange(0.001, 0.01, 0.001), single_lead_current=False, savepath=None
):
    if savepath is None:
        savepath = os.getcwd()
    Iin = 10e-9  # A
    deltaV12_inmuV = []
    deltaV34_inmuV = []
    target_energies = []
    # grid parameters
    N1, L = 36, 90
    # core parameters
    geop = dict(
        lx_leg=int(N1), ly_leg=int(N1 / 6), lx_neck=int(N1 / 6), ly_neck=int(N1 / 6)
    )
    hamp_sys = dict(ts=tk, ws=0, vs=0, ms=0.05, Wdis=0, a=L / N1)
    hamp_lead = dict(tl=tk, wl=0, vl=0, ml=0.05)
    syst = mkhbar_4t(geop, hamp_sys, hamp_lead)  # This system won't be changed anymore

    # First calculate the integral density of state to further derive the target energy for later calculations which correspond to the fixed target density
    energy_range = np.arange(0, 0.035, 0.0001)
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

            plt.savefig(os.path.join(savepath, f"quad_density_{densities[i]:0.5f}.png"))
            plt.close(fig)  # to avoid 'figure.max_open_warning'

        else:
            print(
                f"Warning: no mode appears at the energy = {energy}, thus calculation for the local density failed"
            )
    return densities, deltaV12_inmuV, deltaV34_inmuV


if __name__ == "__main__":
    # rvl_l, rvl_v12, rvl_v34 = rashba_vary_lambda(lamd = np.arange(0,320,30),single_lead_current=True,target_density = 0.01,savepath='plots/rashba_vary_lambda')
    # dvl_l, dvl_v12, dvl_v34 = dirac_vary_lambda(lamd = np.arange(10,320,30),single_lead_current=True,target_density = 0.01,savepath='plots/dirac_vary_lambda')
    # dvd_d, dvd_v12, dvd_v34 = dirac_vary_density(densities = np.arange(0.001,0.011,0.001),lamb = 300,single_lead_current=True,savepath='plots/dirac_vary_density')
    # dvdr_d, dvdr_v12, dvdr_v34 = dirac_vary_density(densities = np.arange(0.001,0.011,0.001),lamb = 10,single_lead_current=True,savepath='plots/dirac_vary_density_ref')
    # qvd_d, qvd_v12, qvd_v34 = quad_vary_density(densities = np.arange(0.001,0.011,0.001),single_lead_current=True,savepath='plots/quad_vary_density')
    # rvd_d, rvd_v12, rvd_v34 = rashba_vary_density(densities = np.arange(0.001,0.011,0.001),lamb = 300, single_lead_current=True,savepath='plots/rashba_vary_density')
    # data = {'rvl_l':rvl_l,'rvl_v12':rvl_v12, 'rvl_v34':rvl_v34,
    #         'dvl_l':dvl_l, 'dvl_v12':dvl_v12, 'dvl_v34':dvl_v34,
    #         'dvd_d':dvd_d, 'dvd_v12':dvd_v12, 'dvd_v34':dvd_v34,
    #         'dvdr_d':dvdr_d, 'dvdr_v12':dvdr_v12, 'dvdr_v34':dvdr_v34,
    #         'qvd_d':qvd_d, 'qvd_v12':qvd_v12, 'qvd_v34':qvd_v34,
    #         'rvd_d':rvd_d, 'rvd_v12':rvd_v12, 'rvd_v34':rvd_v34}
    # with open('tempdata.pkl','wb') as f:
    #     pickle.dump(data,f)

    # load and plot
    with open("tempdata.pkl", "rb") as f:
        data = pickle.load(f)

    def plot_results_3types():
        _, axes = plt.subplots(3, 2, figsize=(10, 8), tight_layout=True)
        axes[0, 0].plot(
            [x * 1e3 for x in data["dvd_d"]],
            data["dvd_v12"],
            label="$\\lambda=300$ $meV\cdot nm$",
        )
        axes[0, 0].plot(
            [x * 1e3 for x in data["dvdr_d"]],
            data["dvdr_v12"],
            label="$\\lambda=10$ $meV\cdot nm$",
        )
        axes[1, 0].plot([x * 1e3 for x in data["qvd_d"]], data["qvd_v12"])
        axes[2, 0].plot([x * 1e3 for x in data["rvd_d"]], data["rvd_v12"])
        axes[0, 1].plot([x * 1e3 for x in data["dvd_d"]], data["dvd_v34"])
        axes[0, 1].plot([x * 1e3 for x in data["dvdr_d"]], data["dvdr_v34"])
        axes[1, 1].plot([x * 1e3 for x in data["qvd_d"]], data["qvd_v34"])
        axes[2, 1].plot([x * 1e3 for x in data["rvd_d"]], data["rvd_v34"])

        axes[0, 0].scatter([x * 1e3 for x in data["dvd_d"]], data["dvd_v12"])
        axes[0, 0].scatter([x * 1e3 for x in data["dvdr_d"]], data["dvdr_v12"])
        axes[1, 0].scatter([x * 1e3 for x in data["qvd_d"]], data["qvd_v12"])
        axes[2, 0].scatter([x * 1e3 for x in data["rvd_d"]], data["rvd_v12"])
        axes[0, 1].scatter([x * 1e3 for x in data["dvd_d"]], data["dvd_v34"])
        axes[0, 1].scatter([x * 1e3 for x in data["dvdr_d"]], data["dvdr_v34"])
        axes[1, 1].scatter([x * 1e3 for x in data["qvd_d"]], data["qvd_v34"])
        axes[2, 1].scatter([x * 1e3 for x in data["rvd_d"]], data["rvd_v34"])

        [axes[i, j].set_xlim(0, 10) for i in range(3) for j in range(2)]
        [
            axes[i, j].set_xlabel("Density [10$^{11}$cm$^{-2}$]")
            for i in range(3)
            for j in range(2)
        ]
        [ax.set_ylabel("$\Delta V_{34}(\lambda)$ [$\mu$V]") for ax in axes[:, 0]]
        [ax.set_ylabel("$\Delta V_{12}(\lambda)$ [$\mu$V]") for ax in axes[:, 1]]
        axes[0, 0].legend(loc="lower left")
        axes[0, 0].text(
            x=3, y=0, s="$H_m=0,H_\lambda\\neq 0$ (Dirac)", color="r", fontsize=15
        )
        axes[1, 0].text(
            x=3,
            y=0.45,
            s="$H_m\\neq 0,H_\lambda=0$ (Quadratic)",
            color="r",
            fontsize=15,
        )
        axes[2, 0].text(
            x=3,
            y=45,
            s="$H_m\\neq 0,H_\lambda\\neq 0$ (Rashba)\n $\\lambda=300$ $meV\cdot nm$",
            color="r",
            fontsize=15,
        )
        axes[2, 0].set_ylim(33, 50)
        axes[0, 0].text(x=3, y=0.06, s="Nonlocal (I=10 nA)", color="b", fontsize=15)
        axes[0, 1].text(x=3, y=535, s="Local (I=10 nA)", color="b", fontsize=15)

        plt.show()

    def plot_lambda_compr():
        _, axes = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
        axes[0, 0].plot(data["rvl_l"], data["rvl_v12"])
        axes[1, 0].plot(data["dvl_l"], data["dvl_v12"])
        axes[0, 1].plot(data["rvl_l"], data["rvl_v34"])
        axes[1, 1].plot(data["dvl_l"], data["dvl_v34"])

        axes[0, 0].scatter(data["rvl_l"], data["rvl_v12"])
        axes[1, 0].scatter(data["dvl_l"], data["dvl_v12"])
        axes[0, 1].scatter(data["rvl_l"], data["rvl_v34"])
        axes[1, 1].scatter(data["dvl_l"], data["dvl_v34"])

        [axes[i, j].set_xlim(0, 300) for i in range(2) for j in range(2)]
        [
            axes[i, j].set_xlabel("$\\lambda$ [$meV\cdot nm$]")
            for i in range(2)
            for j in range(2)
        ]
        [ax.set_ylabel("$\Delta V_{34}(\lambda)$ [$\mu$V]") for ax in axes[:, 0]]
        [ax.set_ylabel("$\Delta V_{12}(\lambda)$ [$\mu$V]") for ax in axes[:, 1]]
        axes[0, 0].text(
            x=50,
            y=30,
            s="$H_m\\neq 0,H_\lambda\\neq 0$ (Rashba)",
            color="r",
            fontsize=15,
        )
        axes[1, 0].text(
            x=50,
            y=-0.1855,
            s="$H_m=0,H_\lambda\\neq 0$ (Dirac)",
            color="r",
            fontsize=15,
        )

        axes[0, 0].text(x=100, y=45, s="Nonlocal (I=10 nA)", color="b", fontsize=15)
        axes[0, 1].text(x=100, y=310, s="Local (I=10 nA)", color="b", fontsize=15)

        plt.show()

    plot_lambda_compr()

    # axes[0,0].text(x=4,y=500,s='$H_m=0,H_\lambda\\neq 0$',color='r',fontsize=05)
