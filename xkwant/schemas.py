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
    hop: Union[float, int] = 0 # hopping
    mass: Union[float, int] = inf # mass
    wilson: Union[float, int] = 0 # Wilson term 
    soc: Union[float, int] = 0 # spin-orbit coupling
    inv: Union[float, int] = 0 # inversion symmetry
    hyb: Union[float, int] = 0 # hybridization
    gapped: Union[float, int] = 0 # gap at k=0 for Dirac dispersion
    wdis: Union[float, int] = 0 # disorder strength
    def to_dict(self):
        return self.__dict__

if __name__ == "__main__":
    geop = GeomParams(a=1, lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
    print(geop.to_dict())