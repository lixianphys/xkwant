import matplotlib.pyplot as plt
from xkwant.templates import *
from xkwant.utils import get_idos
from xkwant.physics import tk, LATTICE_CONST_HGTE
import numpy as np
import kwant


all_eh_values = np.arange(0, 0.15, 0.05)
all_ei_values = np.arange(0, 0.0002, 0.0001)

N1 = 50  # the number of lattices in the longitudinal direction
L = N1 * LATTICE_CONST_HGTE
# core parameters
geop = dict(
    a=L / N1,
    lx_leg=int(N1),
    ly_leg=int(N1 / 6),
    lx_neck=int(N1 / 6),
    ly_neck=int(N1 / 6),
)

hamp_sys = dict(ws=0.1, vs=0.28, invs=0, hybs=0.3, ms=0.05, ts=tk)
hamp_lead = dict(wl=0.1, vl=0.28, invl=0, hybl=0.3, ml=0.05, tl=tk)
syst = doublequad_mkhbar_4t(geop, hamp_sys, hamp_lead)
fig, axes = plt.subplots(2, 1, figsize=(10, 10))


kwant.plotter.bands(
    syst.finalized().leads[0],
    ax=axes[0],
    momenta=np.arange(-0.5, 0.5, 0.01),
)
axes[0].set_ylim(-0.5, 1)
energy_range = np.arange(-0.1, 0.1, 0.01)
idos, energy_range = get_idos(syst, energy_range, use_kpm=False)
axes[1].plot(energy_range, idos)
# fig, axes = plt.subplots(
#     len(all_eh_values),
#     len(all_ei_values),
#     figsize=(10, 10),
#     tight_layout=True,
# )


# for i, eh in enumerate(all_eh_values):
#     for j, ei in enumerate(all_ei_values):
#         hamp_sys = dict(
#             ws=0.1, vs=0.28, invs=ei, hybs=eh, ms=0.05, ts=tk
#         )  # hbar*vf = 280 meV nm and inversion-symmetry breaking term = 4.2 meV (From SM, PRL 106, 126803 (2011) )
#         hamp_lead = dict(wl=0.1, vl=0.28, invl=ei, hybl=eh, ml=0.05, tl=tk)
#         syst = doublequad_mkhbar_4t(geop, hamp_sys, hamp_lead)

#         kwant.plotter.bands(
#             syst.finalized().leads[0],
#             ax=axes[i][j],
#             momenta=np.arange(-0.5, 0.5, 0.01),
#         )
#         axes[i][j].set_ylim(-0.3, 0.3)
#         axes[i][j].set_xlabel(r"k (nm$^{-1}$)")
#         axes[i][j].set_ylabel(r"E (eV)")

# ei = 0.1
# eh = 0

# hamp_sys = dict(
#     ws=0.1, vs=0.28, invs=ei, hybs=eh, ms=0.2, ts=tk
# )  # hbar*vf = 280 meV nm and inversion-symmetry breaking term = 4.2 meV (From SM, PRL 106, 126803 (2011) )
# hamp_lead = dict(wl=0.1, vl=0.28, invl=ei, hybl=eh, ml=0.2, tl=tk)
# # syst = test_doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
# # syst1 = doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)
# # syst2 = new_doubledirac_mkhbar_4t(geop, hamp_sys, hamp_lead)

# syst_rashba = doublerashba_mkhbar_4t(geop, hamp_sys, hamp_lead, finalized=True)
# syst_quad = doublequad_mkhbar_4t(geop, hamp_sys, hamp_lead, finalized=True)

# fig, axes = plt.subplots(
#     1,
#     2,
#     figsize=(10, 6),
#     tight_layout=True,
# )
# kwant.plotter.bands(syst_rashba.leads[0], ax=axes[0])
# # axes.set_ylim(-0.3, 0.3)

# kwant.plotter.bands(syst_quad.leads[0], ax=axes[1])
# # axes.set_ylim(-0.3, 0.3)
# [axe.set_xlabel(r"k (nm$^{-1}$)") for axe in axes]
# [axe.set_ylabel(r"E (eV)") for axe in axes]


plt.show()
