import matplotlib.pyplot as plt
import pickle
import sys
import os
import glob
import re
import kwant
from xkwant.templates import *
from xkwant.utils import density_to_energy


def ei_scenario(template, base_dir, ei):

    # Use glob to find all files with 'ei_' in their names
    all_files = glob.glob(os.path.join(base_dir, "**", "*ei_*"), recursive=True)

    # Filter the files to include only those with the exact 'ei_{ei}' pattern
    matching_files = all_files

    # Function to extract the value of xx from the filename
    def extract_eh_value(filename):
        match = re.search(r"eh_([\d\.]+)", filename)
        if match:
            return float(match.group(1))
        return -1

    # Get the value of xx for each file and sort the list of tuples by this value
    files_with_eh_value = [(file, extract_eh_value(file)) for file in matching_files]
    sorted_files = sorted(files_with_eh_value, key=lambda x: x[1])

    fig, axes = plt.subplots(3, len(sorted_files), figsize=(24, 12), tight_layout=True)

    for i, (directory_path, eh_value) in enumerate(sorted_files):
        print(directory_path)
        with open(directory_path, "rb") as f:
            data = pickle.load(f)

        filename = os.path.basename(directory_path)

        geop = data["geometric_params"]
        hamp_sys = data["hamiltonian_params_sys"]
        hamp_lead = data["hamiltonian_params_lead"]
        idos = data["idos"]
        idos_energy_range = data["idos_energy_range"]
        densities = data["densities"]

        syst = template(geop, hamp_sys, hamp_lead)
        max_eng, min_eng = density_to_energy(
            idos, idos_energy_range, max(densities)
        ), density_to_energy(idos, idos_energy_range, min(densities))

        kwant.plotter.bands(
            syst.finalized().leads[0],
            ax=axes[0][i],
            momenta=np.arange(-0.4, 0.4, 0.004),
        )
        axes[0][i].axhline(y=max_eng, linestyle="--")
        axes[0][i].axhline(y=min_eng, linestyle="--")
        axes[0][i].set_ylim(-0.1, 0.1)
        axes[0][i].set_xlabel(r"k (nm$^{-1}$)")
        axes[0][i].set_ylabel(r"E (eV)")
        axes[0][i].text(x=-0.3, y=0.12, s=f"$\Delta_h$={eh_value} eV", fontsize=18)

        axes[1][i].plot(data["densities"], data["voltage_V12"], color="k")
        axes[1][i].scatter(data["densities"], data["voltage_V12"], s=15, color="r")
        axes[1][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[1][i].set_ylabel(r"V34 [$\mu$V]")
        axes[2][i].plot(data["densities"], data["voltage_V34"], color="k")
        axes[2][i].scatter(data["densities"], data["voltage_V34"], s=15, color="r")
        axes[2][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[2][i].set_ylabel(r"V12 [$\mu$V]")

    # fig.suptitle(
    #     f"$\Delta_h=0.0,0.01,0.02,0.03$",
    #     x=0.5,
    #     y=0.98,
    #     ha="center",
    #     fontsize=20,
    # )

    fig.subplots_adjust(top=0.95)

    plt.savefig(f"plots/allinone_ei_{ei}.pdf")


def eh_scenario(base_dir, eh):

    # Use glob to find all files with 'ei_' in their names
    all_files = glob.glob(os.path.join(base_dir, "**", "*eh_*"), recursive=True)

    # Filter the files to include only those with the exact 'ei_{ei}' pattern
    matching_files = [
        file
        for file in all_files
        if re.search(rf"eh_{re.escape(eh)}(_|\.)", os.path.basename(file))
    ]

    # Function to extract the value of xx from the filename
    def extract_ei_value(filename):
        match = re.search(r"ei_([\d\.]+)", filename)
        if match:
            return float(match.group(1))
        return -1

    # Get the value of xx for each file and sort the list of tuples by this value
    files_with_ei_value = [(file, extract_ei_value(file)) for file in matching_files]
    sorted_files = sorted(files_with_ei_value, key=lambda x: x[1])
    fig, axes = plt.subplots(3, len(sorted_files), figsize=(20, 12))

    for i, (directory_path, ei_value) in enumerate(sorted_files):
        with open(directory_path, "rb") as f:
            data = pickle.load(f)

        filename = os.path.basename(directory_path)

        geop = data["geometric_params"]
        hamp_sys = data["hamiltonian_params_sys"]
        hamp_lead = data["hamiltonian_params_lead"]
        idos = data["idos"]
        idos_energy_range = data["idos_energy_range"]
        densities = data["densities"]

        try:
            syst = doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
        except KeyError:
            try:
                syst = gappeddirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
            except KeyError:
                syst = mkhbar_4t(geop, hamp_sys, hamp_lead)

        max_eng, min_eng = density_to_energy(
            idos, idos_energy_range, max(densities)
        ), density_to_energy(idos, idos_energy_range, min(densities))

        kwant.plotter.bands(
            syst.finalized().leads[0], ax=axes[0][i], momenta=np.arange(-0.5, 0.5, 0.01)
        )
        axes[0][i].axhline(y=max_eng, linestyle="--")
        axes[0][i].axhline(y=min_eng, linestyle="--")
        axes[0][i].set_ylim(-0.3, 0.3)

        axes[1][i].plot(data["densities"], data["voltage_V12"], color="k")
        axes[1][i].scatter(data["densities"], data["voltage_V12"], s=15, color="r")
        axes[1][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[1][i].set_ylabel(r"V34 [$\mu$V]")
        axes[2][i].plot(data["densities"], data["voltage_V34"], color="k")
        axes[2][i].scatter(data["densities"], data["voltage_V34"], s=15, color="r")
        axes[2][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[2][i].set_ylabel(r"V12 [$\mu$V]")

    fig.suptitle(
        f"$\Delta_h={eh}$, $\Delta_i=0.0,0.02,0.04,0.06$",
        x=0.5,
        y=0.98,
        ha="center",
        fontsize=20,
    )

    fig.subplots_adjust(top=0.95)

    plt.savefig(f"plots/doublesurfaces/allinone_eh_{eh}.png")


#####################################


def gap_scenario(template, base_dir):
    # Use glob to find all files with 'ei_' in their names
    all_files = glob.glob(os.path.join(base_dir, "**"), recursive=False)
    # Filter the files to include only those with the exact 'ei_{ei}' pattern
    matching_files = all_files

    # Function to extract the value of xx from the filename
    def extract_gap_value(filename):
        match = re.search(r"gap_([\d\.]+)", filename)
        if match:
            return float(match.group(1))
        return -1

    # Get the value of xx for each file and sort the list of tuples by this value
    files_with_gap_value = [(file, extract_gap_value(file)) for file in matching_files]
    sorted_files = sorted(files_with_gap_value, key=lambda x: x[1])
    fig, axes = plt.subplots(3, len(sorted_files), figsize=(30, 12))

    for i, (directory_path, gap_value) in enumerate(sorted_files):
        with open(directory_path, "rb") as f:
            data = pickle.load(f)

        filename = os.path.basename(directory_path)

        geop = data["geometric_params"]
        hamp_sys = data["hamiltonian_params_sys"]
        hamp_lead = data["hamiltonian_params_lead"]
        idos = data["idos"]
        idos_energy_range = data["idos_energy_range"]
        densities = data["densities"]
        syst = template(geop, hamp_sys, hamp_lead)
        max_eng, min_eng = density_to_energy(
            idos, idos_energy_range, max(densities)
        ), density_to_energy(idos, idos_energy_range, min(densities))

        kwant.plotter.bands(
            syst.finalized().leads[0], ax=axes[0][i], momenta=np.arange(-0.5, 0.5, 0.01)
        )
        axes[0][i].axhline(y=max_eng, linestyle="--")
        axes[0][i].axhline(y=min_eng, linestyle="--")
        axes[0][i].set_xlabel(r"k (nm$^{-1}$)")
        axes[0][i].set_ylabel(r"E (eV)")
        axes[0][i].set_ylim(-0.2, 0.2)
        axes[0][i].text(x=-0.3, y=0.22, s=f"$\Delta_g$={gap_value} eV", fontsize=18)

        axes[1][i].plot(data["densities"], data["voltage_V12"], color="k")
        axes[1][i].scatter(data["densities"], data["voltage_V12"], s=15, color="r")
        axes[1][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[1][i].set_ylabel(r"V34 [$\mu$V]")
        axes[2][i].plot(data["densities"], data["voltage_V34"], color="k")
        axes[2][i].scatter(data["densities"], data["voltage_V34"], s=15, color="r")
        axes[2][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[2][i].set_ylabel(r"V12 [$\mu$V]")

    fig.subplots_adjust(top=0.95)

    plt.savefig(f"plots/allinone.pdf")


def narrowleg_scenario(base_dir):
    # Use glob to find all files with 'ei_' in their names
    all_files = glob.glob(os.path.join(base_dir, "**"), recursive=False)
    # Filter the files to include only those with the exact 'ei_{ei}' pattern
    matching_files = all_files

    # Function to extract the value of xx from the filename
    def extract_gap_value(filename):
        match = re.search(r"N1_([\d\.]+)", filename)
        if match:
            return float(match.group(1))
        return -1

    # Get the value of xx for each file and sort the list of tuples by this value
    files_with_n1_value = [(file, extract_gap_value(file)) for file in matching_files]
    sorted_files = sorted(files_with_n1_value, key=lambda x: x[1])
    fig, axes = plt.subplots(3, len(sorted_files), figsize=(40, 12))

    for i, (directory_path, n1_value) in enumerate(sorted_files):
        with open(directory_path, "rb") as f:
            data = pickle.load(f)

        filename = os.path.basename(directory_path)

        geop = data["geometric_params"]
        hamp_sys = data["hamiltonian_params_sys"]
        hamp_lead = data["hamiltonian_params_lead"]
        idos = data["idos"]
        idos_energy_range = data["idos_energy_range"]
        densities = data["densities"]

        try:
            syst = doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
        except KeyError:
            try:
                syst = gappeddirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
            except KeyError:
                syst = mkhbar_4t(geop, hamp_sys, hamp_lead)

        max_eng, min_eng = density_to_energy(
            idos, idos_energy_range, max(densities)
        ), density_to_energy(idos, idos_energy_range, min(densities))

        # kwant.plotter.bands(
        #     syst.finalized().leads[0], ax=axes[0][i], momenta=np.arange(-0.5, 0.5, 0.01)
        # )
        kwant.plotter.plot(syst, ax=axes[0][i])

        axes[1][i].plot(data["densities"], data["voltage_V12"], color="k")
        axes[1][i].scatter(data["densities"], data["voltage_V12"], s=15, color="r")
        axes[1][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[1][i].set_ylabel(r"V34 [$\mu$V]")
        axes[2][i].plot(data["densities"], data["voltage_V34"], color="k")
        axes[2][i].scatter(data["densities"], data["voltage_V34"], s=15, color="r")
        axes[2][i].set_xlabel(r"Density [nm$^{-2}$]")
        axes[2][i].set_ylabel(r"V12 [$\mu$V]")

    fig.suptitle(
        f"$N_1=30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900$, while keeping the leg width unchanged",
        x=0.5,
        y=0.98,
        ha="center",
        fontsize=20,
    )

    fig.subplots_adjust(top=0.95)

    plt.savefig(f"plots/narrowleg/kpmoff/allinone.png")


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <scenario_name> <path_to_folder> [value]")
        sys.exit(1)

    scenario_name = sys.argv[1]
    base_dir = sys.argv[2]
    if scenario_name == "gap":
        gap_scenario(base_dir)
    elif scenario_name == "ei":
        try:
            ei = sys.argv[3]
            ei_scenario(base_dir, ei)
        except IndexError:
            print("Usage: python script.py <scenario_name> <path_to_folder> [value]")
    elif scenario_name == "eh":
        try:
            eh = sys.argv[3]
            eh_scenario(base_dir, eh)
        except IndexError:
            print("Usage: python script.py <scenario_name> <path_to_folder> [value]")
    elif scenario_name == "nl":
        narrowleg_scenario(base_dir)
    else:
        print("Invalid scenario number. Please choose 1, 2, or 3.")
        sys.exit(1)


def ehei_scenario(template, base_dir, type):

    # Use glob to find all files with 'ei_' in their names
    all_files = glob.glob(os.path.join(base_dir, "**"), recursive=True)[1:]

    # Function to extract the value of xx from the filename
    def extract_ei_value(filename):
        match = re.search("ei_([\d\.]+)", filename)
        if match:
            return float(match.group(1))
        return -1

    def extract_eh_value(filename):
        match = re.search("eh_([\d\.]+)", filename)
        if match:
            return float(match.group(1))
        return -1

    # Get the value of xx for each file and sort the list of tuples by this value
    files_with_values = [
        (file, extract_eh_value(file), extract_ei_value(file)) for file in all_files
    ]
    all_ei_values = sorted(list(set([extract_ei_value(file) for file in all_files])))
    all_eh_values = sorted(list(set([extract_eh_value(file) for file in all_files])))
    sorted_files = sorted(files_with_values, key=lambda x: (x[1], x[2]))

    fig, axes = plt.subplots(
        len(all_eh_values),
        len(all_ei_values),
        figsize=(10, 10),
        tight_layout=True,
    )

    for i, eh in enumerate(all_eh_values):
        for j, ei in enumerate(all_ei_values):
            directory_path = sorted_files[i * len(all_ei_values) + j][0]
            with open(directory_path, "rb") as f:
                data = pickle.load(f)
            geop = data["geometric_params"]
            hamp_sys = data["hamiltonian_params_sys"]
            hamp_lead = data["hamiltonian_params_lead"]
            idos = data["idos"]
            idos_energy_range = data["idos_energy_range"]
            densities = data["densities"]
            if type == "b":
                syst = template(geop, hamp_sys, hamp_lead)
                max_eng, min_eng = density_to_energy(
                    idos, idos_energy_range, max(densities)
                ), density_to_energy(idos, idos_energy_range, min(densities))

                kwant.plotter.bands(
                    syst.finalized().leads[0],
                    ax=axes[i][j],
                    momenta=np.arange(-0.5, 0.5, 0.01),
                )
                axes[i][j].axhline(y=max_eng, linestyle="--")
                axes[i][j].axhline(y=min_eng, linestyle="--")
                axes[i][j].set_ylim(-0.3, 0.3)
                axes[i][j].set_xlabel(r"k (nm$^{-1}$)")
                axes[i][j].set_ylabel(r"E (eV)")

            elif type == "v34":

                axes[i][j].plot(data["densities"] * 1e3, data["voltage_V12"], color="k")
                axes[i][j].scatter(
                    data["densities"] * 1e3, data["voltage_V12"], s=15, color="r"
                )
                axes[i][j].set_xlabel(r"Density [10$^{11}$cm$^{-2}$]")
                axes[i][j].set_ylabel(r"V34 [$\mu$V]")
                axes[i][j].set_xticks([0, 2, 4, 6, 8])

            else:
                axes[i][j].plot(data["densities"] * 1e3, data["voltage_V34"], color="k")
                axes[i][j].scatter(
                    data["densities"] * 1e3, data["voltage_V34"], s=15, color="r"
                )
                axes[i][j].set_xlabel(r"Density [10$^{11}$cm$^{-2}$]")
                axes[i][j].set_ylabel(r"V12 [$\mu$V]")
                axes[i][j].set_xticks([0, 2, 4, 6, 8])

    fig.suptitle(
        f"$\Delta_h=0,0.02,0.04,0.06$ (vertical), $\Delta_i=0,0.01,0.02,0.03$ (horizontal)",
        x=0.5,
        y=0.98,
        ha="center",
        fontsize=20,
    )

    plt.savefig(f"plots/doublesurfaces/dirac/n100/allinone_v12.pdf")


if __name__ == "__main__":
    # main()
    # ei_scenario(doublequad_mkhbar_4t, "data/julia-hpc/juli11bis13/doublequad", 0)
    # ehei_scenario(
    #     new_doubledirac_mkhbar_4t, f"data/doublesurfaces_data/dirac/n100", "v12"
    # )
    gap_scenario(gappeddirac_mkhbar_4t, r"data/julia-hpc/juli19")
