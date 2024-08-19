import os
import pickle
import numpy as np
import scipy.sparse.linalg as sla
from xkwant.batch import *
from xkwant.templates import *
from xkwant.physics import *
from xkwant.utils import *
from xkwant.log import log_function_call
from xkwant.config import LATTICE_CONST_HGTE

# check out https://kwant-project.org/doc/latest/tutorial/spectrum

lambda_val = 300
plot_local_quantity = True
plot_single_lead = True
Iin = 10e-9  # A
# grid parameters
N1, L = 30, 30 * 0.646
# core parameters
geop = dict(
    a=L / N1,
    lx_leg=int(N1),
    ly_leg=int(N1 / 6),
    lx_neck=int(N1 / 6),
    ly_neck=int(N1 / 6),
)
hamp_sys = dict(ts=tk, ws=0 / 3e3, vs=0 / 1e3, ms=0.5, Wdis=0)
hamp_lead = dict(tl=tk, wl=0 / 3e3, vl=0 / 1e3, ml=0.5)
syst = mkhbar_4t_magf(geop, hamp_sys, hamp_lead)
fsyst = syst.finalized()

Bfields = np.arange(-100, 100, 1)
energies = []

for B in Bfields:
    # Obtain the Hamiltonian as a sparse matrix
    ham_mat = fsyst.hamiltonian_submatrix(params=dict(B=B), sparse=True)

    # we only calculate the 15 lowest eigenvalues
    ev = sla.eigsh(ham_mat.tocsc(), k=6, sigma=0, return_eigenvectors=False)

    energies.append(sorted(ev))
print(ham_mat.size)
print(tk)
plt.figure()
plt.plot(Bfields, energies)
plt.show()
