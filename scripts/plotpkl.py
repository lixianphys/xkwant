import matplotlib.pyplot as plt
import pickle
import sys

if len(sys.argv) == 1:
    filename = "data_dirachbar.py_20240529_1041.pkl"
else:
    filename = sys.argv[1]

with open(filename, "rb") as f:
    data = pickle.load(f)

fig, axes = plt.subplots(3, 1, figsize=(8, 12), tight_layout=True)
fig.suptitle(
    f"filename:{filename}\nIin:{data['Iin']}\nidos_kpm: {data['idos_kpm']}\nN1: {data['N1']},L: {data['L']}\nhamiltonian_params_sys: {data['hamiltonian_params_sys']}",
    x=0.5,
    y=0.98,
    ha="center",
)
axes[0].plot(data["densities"], data["voltage_V12"], color="k")
axes[0].scatter(data["densities"], data["voltage_V12"], s=15, color="r")
axes[0].set_xlabel("Density [nm$^{-2}$]")
axes[0].set_ylabel("V34 [$\\mu$V]")
axes[1].plot(data["densities"], data["voltage_V34"], color="k")
axes[1].scatter(data["densities"], data["voltage_V34"], s=15, color="r")
axes[1].set_xlabel("Density [nm$^{-2}$]")
axes[1].set_ylabel("V12 [$\\mu$V]")
axes[2].plot(data["idos"], data["idos_energy_range"], color="k")
axes[2].scatter(data["idos"], data["idos_energy_range"], s=15, color="r")
axes[2].set_xlabel("Density [nm$^{-2}$]")
axes[2].set_ylabel("Energy [eV]")
plt.show()
