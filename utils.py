import bisect
import kwant
import numpy as np

__all__ = ['energy_to_density','density_to_energy','get_idos']

def find_position(sorted_list,x):
    index = bisect.bisect_left(sorted_list,x)
    if index != len(sorted_list):
        return index
    return -1

def energy_to_density(idos,energies,energy):
    index = find_position(energies,energy)
    if index == -1:
        raise ValueError("need more eigenstates")
    return idos[index]

def density_to_energy(idos,energies,density):
    index = find_position(idos,density)
    if index == -1:
        raise ValueError("need more eigenstates")
    return energies[index]

def get_idos(syst,energy_range):
    """
    Calculate the integrated density of states (IDOS) for a given system over a specified energy range.

    Parameters:
    - system: A Kwant system (kwant.Builder).
    - energy_range: An array of energy values.

    Returns:
    - idos: Integrated density of states.
    - energy_range: The energy range used for the calculation.
    """
    fsyst = syst.finalized()
    num_leads = len(syst.leads)
    rho = kwant.operator.Density(fsyst,sum=True)
    dos = []
    for energy in energy_range:
        wf = kwant.wave_function(fsyst,energy=energy)
        all_states = np.vstack([wf(i) for i in range(num_leads)])
        dos.append(sum(rho(mode) for mode in all_states)/syst.area/(2*np.pi)) # Here syst.area is the actual area / (lattice constant a)**2
    dos_array = np.array(dos)
    energy_step = abs(energy_range[0]-energy_range[1])
    idos = energy_step * np.nancumsum(dos_array)
    return idos, energy_range