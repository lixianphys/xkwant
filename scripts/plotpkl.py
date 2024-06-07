import matplotlib.pyplot as plt
import pickle
import sys
import os
from xkwant.templates import *
from xkwant.utils import density_to_energy

directory_path = sys.argv[1]

with open(directory_path, "rb") as f:
    data = pickle.load(f)

filename = os.path.basename(directory_path)
fig, axes = plt.subplots(2, 2, figsize=(10, 12))
fig.suptitle(
    f"filename:{filename}\nIin:{data['Iin']}\nidos_kpm: {data['idos_kpm']}\nN1: {data['N1']},L: {data['L']:0.2f}\nhamiltonian_params_sys: {data['hamiltonian_params_sys']}",
    x=0.5,
    y=0.95,
    ha="center",
)

fig.subplots_adjust(top=0.85)

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

kwant.plotter.bands(syst.finalized().leads[0], ax=axes[0][0])
axes[0][0].axhline(y=max_eng, linestyle="--")
axes[0][0].axhline(y=min_eng, linestyle="--")

axes[0][1].plot(data["densities"], data["voltage_V12"], color="k")
axes[0][1].scatter(data["densities"], data["voltage_V12"], s=15, color="r")
axes[0][1].set_xlabel("Density [nm$^{-2}$]")
axes[0][1].set_ylabel("V34 [$\\mu$V]")
axes[1][1].plot(data["densities"], data["voltage_V34"], color="k")
axes[1][1].scatter(data["densities"], data["voltage_V34"], s=15, color="r")
axes[1][1].set_xlabel("Density [nm$^{-2}$]")
axes[1][1].set_ylabel("V12 [$\\mu$V]")
axes[1][0].plot(data["idos"], data["idos_energy_range"], color="k")
axes[1][0].scatter(data["idos"], data["idos_energy_range"], s=15, color="r")
axes[1][0].set_xlabel("Density [nm$^{-2}$]")
axes[1][0].set_ylabel("Energy [eV]")


plt.savefig(f"plots/narrowleg/kpmoff/{filename}.png")
