import tinyarray
import scipy.constants as sct

e0 = sct.e  # elementary charge
pi = sct.pi
hbar = sct.hbar
me = sct.m_e
tk = 38.0998 / 1e3  # = hbar^2/(2m_e) in eV*nm^2
e2h = 0.0000387405  # = e^2/h in A/V
el = 1.602176634e-19  # e in C
eh = 2.41799e14  # e/h in 1/s

s_0 = tinyarray.array([[1, 0], [0, 1]])
s_x = tinyarray.array([[0, 1], [1, 0]])
s_y = tinyarray.array([[0, -1j], [1j, 0]])
s_z = tinyarray.array([[1, 0], [0, -1]])

sigma_0 = s_0  # Alias
sigma_x = s_x  # Alias
sigma_y = s_y  # Alias
sigma_z = s_z  # Alias

LATTICE_CONST_HGTE = 0.646  # lattice constant of HgTe in nm from Wikipedia
