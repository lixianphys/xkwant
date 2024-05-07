from kwant import Builder
from kwant.continuum import build_discretized,discretize_symbolic
from physics import *

__all__ = ['Hbar']
class Hbar(Builder):
    def __init__(self,geo_param:dict):
        super(Hbar,self).__init__()
        self.lx_leg = geo_param['lx_leg']
        self.ly_leg = geo_param['ly_leg']
        self.lx_neck = geo_param['lx_neck']
        self.ly_neck = geo_param['ly_neck']
    def __str__(self):
        return (f"Instance of Hbar class: lx_leg={self.lx_leg},ly_leg={self.ly_leg},lx_neck={self.lx_neck},"
                f"ly_neck={self.ly_neck}")
    def build_byfill(self,continuum_model: str):
        skeleton = build_discretized(*discretize_symbolic(continuum_model))
        def hbar_shape(site):
            x,y = site.tag
            return ((0<=x<self.lx_leg and 0<=y<self.ly_leg) or (0<=x<self.lx_leg and
                                                               self.ly_leg+self.ly_neck<=y<self.ly_leg*2+self.ly_neck)
                    or (
                    self.lx_leg//2 - self.lx_neck//2 <=x< self.lx_leg//2 + self.lx_neck//2
                                                                                   and
                    self.ly_leg<=y<self.ly_leg+self.ly_neck))
        self.fill(template=skeleton,shape=hbar_shape,start=(0,0))

if __name__ == '__main__':
    geop = dict(lx_leg=1, ly_leg=1, lx_neck=1, ly_neck=1)
    bhz_continuum = '''
        + mu * kron(sigma_0, sigma_0)
        + M * kron(sigma_0, sigma_z)
        - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z) - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
        + A * (k_x * kron(sigma_z, sigma_x) + k_y * kron(sigma_0, sigma_y))
    '''
    print(sigma_0)
    testhbar = Hbar(geop)
    testhbar.build_byfill(bhz_continuum)


