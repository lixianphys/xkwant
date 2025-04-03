import matplotlib.pyplot as plt
import pickle
import sys
import os
import kwant
from xkwant.templates import *
from xkwant.utils import density_to_energy
import numpy as np

directory_path = sys.argv[1]

with open(directory_path, "rb") as f:
    data = pickle.load(f)


filename = os.path.basename(directory_path)
fig, axes = plt.subplots(2, 2, figsize=(10, 12))
fig.suptitle(
    f"filename:{filename}\nIin:{data['Iin']}\nidos_kpm: {data['idos_kpm']}\nN1: {data['N1']},L: {data['L']:0.2f}\n hamiltonian_params_sys: {data['hamiltonian_params_sys']}",
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

print(f"H_params_sys: {hamp_sys}\n")
print(f"H_params_lead: {hamp_lead}\n")

if 'ms' in hamp_sys and 'ws' in hamp_sys:
    syst = doublerashba_mkhbar_4t(geop, hamp_sys, hamp_lead)
elif 'ms' in hamp_sys and 'ws' not in hamp_sys:
    syst = doublequad_mkhbar_4t(geop, hamp_sys, hamp_lead)
elif 'ms' not in hamp_sys and 'ws' in hamp_sys:
    syst = doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
else:
    raise ValueError("Parameters for system Hamiltonian are not compatible with Dirac, Rashba, Quadratic systems")


density_to_label = [0.002, 0.0047, 0.0085]

max_eng, min_eng = density_to_energy(
    idos, idos_energy_range, max(densities)
), density_to_energy(idos, idos_energy_range, min(densities))

kwant.plotter.bands(
    syst.finalized().leads[0], ax=axes[0][0], momenta=np.arange(-0.5, 0.5, 0.01)
)
axes[0][0].axhline(y=max_eng, linestyle="--")
axes[0][0].axhline(y=min_eng, linestyle="--")
axes[0][0].set_ylim(-0.5, 0.5)

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

[axes[0][1].axvline(x=v, color="g") for v in density_to_label]
[axes[1][1].axvline(x=v, color="g") for v in density_to_label]

[
    axes[0][0].axhline(y=density_to_energy(idos, idos_energy_range, v), color="g")
    for v in density_to_label
]

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

plt.savefig("plots/{}.pdf".format(filename))
