import bisect
import kwant
import kwant.kpm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# __all__ = ['energy_to_density','density_to_energy','get_idos','get_idos_kpm']


def find_position(sorted_list, x):
    index = bisect.bisect_left(sorted_list, x)
    if index != len(
        sorted_list
    ):  # Normal: x is not larger than all elements in sorted_list.
        return index
    return (
        -1
    )  # x is larger than all elements in sorted_list, return -1 to indicate error


def energy_to_density(idos, energies, energy):
    index = find_position(energies, energy)
    if index == -1:
        raise ValueError(
            "The given energy is too high to predict its corresponding density. Add more DOS values at higher energies in idos"
        )
    return (
        idos[index] + idos[index + 1]
    ) / 2  # Assign the mean of two closest values to its prediction


def density_to_energy(idos, energies, density):
    index = find_position(idos, density)
    if index == -1:
        raise ValueError(
            "The given density is too high to predict its corresponding energy. Add more DOS values at higher energies in idos"
        )
    return (
        energies[index] + energies[index + 1]
    ) / 2  # Assign the mean of two closest values to its prediction


def get_idos(syst, energy_range, use_kpm=False):
    """
    Calculate the integrated density of states (IDOS) for a given system over a specified energy range.

    Parameters:
    - system: A Kwant system (kwant.Builder).
    - energy_range: An array of energy values.

    Returns:
    - idos: Integrated density of states.
    - energy_range: The energy range used for the calculation. This is for calculating bands above zero energy, so energy_range should increase from zero.
    """

    if use_kpm:
        energy_resolution = (
            (max(energy_range) - min(energy_range)) * 5 / len(energy_range)
        )
        # if len(syst.leads)!=0:
        #     print('len(syst.leads)!=0')
        #     syst = copy.deepcopy(syst) # create a copy for later manipulation, do not alter the input Builder instance
        #     syst.leads = [] # remove all leads for using get_dos_kpm
        dos, energies = get_dos_kpm(syst, energy_resolution)
        dos = np.real(dos)  # To extract the real part
        dos = dos[energies > 0]  # Ignore DOS below zero energy
        energies = energies[energies > 0]
        idos = cumtrapz(dos, energies, initial=0)
        energy_range = np.array(energy_range)
        energy_range = energies[
            (energies >= min(energy_range)) & (energies <= max(energy_range))
        ]  # the energies here should include all possible eigenvalue of energy
        lowest_index = find_position(energies, min(energy_range))
        idos = idos[
            lowest_index : lowest_index + len(energy_range)
        ]  # This ensure the returned idos and energy_range have the same length
    else:
        dos = get_dos(syst, energy_range)
        dos = np.array(dos)  # To extract the real part
        dos = np.real(dos)
        idos = cumtrapz(dos, energy_range, initial=0)
    return idos, energy_range


def get_dos(syst, energy_range):
    fsyst = syst.finalized()
    num_leads = len(syst.leads)
    rho = kwant.operator.Density(fsyst, sum=True)
    dos = []
    for energy in energy_range:
        wf = kwant.wave_function(fsyst, energy=energy)
        all_states = np.vstack([wf(i) for i in range(num_leads)])
        dos.append(
            sum(rho(mode) for mode in all_states) / syst.area / (2 * np.pi)
        )  # Here syst.area is the actual area / (lattice constant a)**2
    return dos


def get_dos_kpm(syst, energy_resolution):
    fsyst = syst.finalized()
    spectrum = kwant.kpm.SpectralDensity(fsyst, rng=0)
    try:
        spectrum.add_moments(energy_resolution=energy_resolution)
    except ValueError as e:
        print("Fall back: Default resolution from kwant.kpm.SpectralDensity is used")
    energies, densities = spectrum()
    dos = [density / syst.area for density in densities]
    return dos, energies


############### Plot ###############################


def prepare_plot(xlabel: str, xlim: tuple, ylabel=None, ylabel2=None, figsize=(10, 6)):
    """prepare axes for complex plots"""
    fig, axs = plt.subplots(2, 2, figsize=figsize, tight_layout=True)
    [ax.set_xlabel(xlabel) for ax in axs[:, 0]]
    [ax.set_xlim(*xlim) for ax in axs[:, 0]]
    if ylabel is None:
        axs[0, 0].set_ylabel("$\Delta V_{34}(\lambda)$ [$\mu$V]")
    else:
        axs[0, 0].set_ylabel(ylabel)
    if ylabel2 is None:
        axs[1, 0].set_ylabel("$G_{SH}$ [e/8$\pi$]")
    else:
        axs[1, 0].set_ylabel(ylabel2)
    return fig, axs
