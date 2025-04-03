from dataclasses import dataclass
from typing import Union
from numpy import inf

@dataclass
class GeomParams():
    lx_leg: int
    ly_leg: int
    lx_neck: int
    ly_neck: int
    a: Union[float, int] = 1 # lattice constant
    def to_dict(self):
        return self.__dict__


@dataclass
class HamParams():
    hop: Union[float, int] = 0 # hopping vs, vl
    mass: Union[float, int] = inf # mass ms, ml
    wilson: Union[float, int] = 0 # Wilson term ws, wl
    soc: Union[float, int] = 0 # spin-orbit coupling ts, tl
    inv: Union[float, int] = 0 # inversion symmetry invs, invl
    hyb: Union[float, int] = 0 # hybridization hybs, hybl
    gapped: Union[float, int] = 0 # gap at k=0 for Dirac dispersion
    wdis: Union[float, int] = 0 # disorder strength
    def to_dict(self):
        return self.__dict__

        