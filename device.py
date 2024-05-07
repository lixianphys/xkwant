import kwant
import matplotlib.pyplot as plt
from kwant import Builder, TranslationalSymmetry
from kwant.continuum import build_discretized,discretize_symbolic
from physics import *


__all__ = ['Hbar']
class Hbar(Builder):
    def __init__(self,geo_params):
        super(Hbar,self).__init__()
        self.lx_leg = geo_params['lx_leg']
        self.ly_leg = geo_params['ly_leg']
        self.lx_neck = geo_params['lx_neck']
        self.ly_neck = geo_params['ly_neck']
    def __str__(self):
        return (f"Instance of Hbar class: lx_leg={self.lx_leg},ly_leg={self.ly_leg},lx_neck={self.lx_neck},"
                f"ly_neck={self.ly_neck}")
    def build_byfill(self,continuum_model):
        template = build_discretized(*discretize_symbolic(continuum_model))
        def hbar_shape(site):
            x,y = site.tag
            return ((0<=x<self.lx_leg and 0<=y<self.ly_leg) or (0<=x<self.lx_leg and
                                                               self.ly_leg+self.ly_neck<=y<self.ly_leg*2+self.ly_neck)
                    or (
                    self.lx_leg//2 - self.lx_neck//2 <=x< self.lx_leg//2 + self.lx_neck//2
                                                                                   and
                    self.ly_leg<=y<self.ly_leg+self.ly_neck))
        self.fill(template,hbar_shape,start=(0,0))
    def attach_lead_byfill(self,continuum_model,pos,conservation_law=None):
        template = build_discretized(*discretize_symbolic(continuum_model))
        if pos.upper() == 'BL':
            bot_left_lead = Builder(TranslationalSymmetry((-1,0)),conservation_law=conservation_law)
            bot_left_lead.fill(template,lambda site: 0 <= site.tag[1] <= self.ly_leg, (0, 1))
            self.attach_lead(bot_left_lead)
        elif pos.upper() == 'TL':
            top_left_lead = Builder(TranslationalSymmetry((-1,0)),conservation_law=conservation_law)
            top_left_lead.fill(template,lambda site: self.ly_leg+self.ly_neck <= site.tag[1] < self.ly_leg*2+self.ly_neck, (0,self.ly_leg+self.ly_neck))
            self.attach_lead(top_left_lead)
        elif pos.upper() == 'BR':
            bot_right_lead = Builder(TranslationalSymmetry((1,0)),conservation_law=conservation_law)
            bot_right_lead.fill(template,lambda site: 0 <= site.tag[1] <= self.ly_leg, (0,1))
            self.attach_lead(bot_right_lead)
        elif pos.upper() == 'TR':
            top_right_lead = Builder(TranslationalSymmetry((1,0)),conservation_law=conservation_law)
            top_right_lead.fill(template,lambda site: self.ly_leg+self.ly_neck <= site.tag[1] < self.ly_leg*2+self.ly_neck, (0,self.ly_leg+self.ly_neck))
            self.attach_lead(top_right_lead)
        else:
            raise ValueError(f"pos can only be BL, TL, BR, TR (case non-sensitive)")
    def set_ham_params(self,params):
        self.ham_params = params


if __name__ == '__main__':
    geop = dict(lx_leg=20, ly_leg=20, lx_neck=10, ly_neck=10)
    bhz_continuum = '''
        + mu * kron(sigma_0, sigma_0)
        + M * kron(sigma_0, sigma_z)
        - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z) - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
        + A * (k_x * kron(sigma_z, sigma_x) + k_y * kron(sigma_0, sigma_y))
    '''
    testhbar = Hbar(geop)
    testhbar.build_byfill(bhz_continuum)
    ham_params = dict(A=0.09, B=-0.18, D=-0.065, M=-0.02, mu=0)
    [testhbar.attach_lead_byfill(bhz_continuum,pos) for pos in ['bl','br','tl','tr']]
    testhbar.set_ham_params(ham_params)
    fsyst = testhbar.finalized()

    fig = plt.figure(figsize=(10,6),tight_layout=True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    kwant.plot(fsyst,ax=ax1)
    kwant.plotter.bands(fsyst.leads[0],params=testhbar.ham_params,ax=ax2)

    plt.show()



