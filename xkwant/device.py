from kwant import Builder, TranslationalSymmetry
from kwant.continuum import discretize, sympify, build_discretized, discretize_symbolic
from .physics import *
from .batch import *
import scipy.sparse.linalg as sla


__all__ = ["Hbar"]


class Hbar(Builder):
    def __init__(self, geo_params):
        super(Hbar, self).__init__()
        self.lx_leg = geo_params["lx_leg"]  # in units of a, the lattice constant
        self.ly_leg = geo_params["ly_leg"]
        self.lx_neck = geo_params["lx_neck"]
        self.ly_neck = geo_params["ly_neck"]
        self.a = geo_params["a"]
        self.area = (self.lx_leg * self.ly_leg * 2 + self.lx_neck * self.ly_neck) * (
            self.a**2
        )  # This area in units of nm^2 can be used for estimating the carrier density
        self.ham_params = dict()

    def __str__(self):
        formatted_ham_params = ", ".join(
            f"{key}={value}" for key, value in self.ham_params.items()
        )
        return (
            f"Instance of Hbar class:\n"
            f"Geometric parameters: lx_leg={self.lx_leg},ly_leg={self.ly_leg},lx_neck={self.lx_neck},ly_neck={self.ly_neck}\n"
            f"{len(self.leads)} leads have been attached\n"
            f"Hamitonian parameters: {formatted_ham_params}"
        )

    def build_byfill(self, continuum_model, params):
        self.set_ham_params(params)
        # template = build_discretized(*discretize_symbolic(continuum_model))
        model_params = sympify(continuum_model, locals=params)
        template = discretize(model_params, grid=self.a)

        def hbar_shape(site):
            x, y = site.tag
            return (
                (0 <= x < self.lx_leg and 0 <= y < self.ly_leg)
                or (
                    0 <= x < self.lx_leg
                    and self.ly_leg + self.ly_neck <= y < self.ly_leg * 2 + self.ly_neck
                )
                or (
                    self.lx_leg // 2 - self.lx_neck // 2
                    <= x
                    < self.lx_leg // 2 + self.lx_neck // 2
                    and self.ly_leg <= y < self.ly_leg + self.ly_neck
                )
            )

        self.fill(template, hbar_shape, start=(0, 0))

    def attach_lead_byfill(self, continuum_model, params, pos, conservation_law=None):
        # template = build_discretized(*discretize_symbolic(continuum_model))
        model_params = sympify(continuum_model, locals=params)
        template = discretize(model_params, grid=self.a)
        if pos.upper() == "BL":
            bot_left_lead = Builder(
                TranslationalSymmetry((-1, 0)), conservation_law=conservation_law
            )
            bot_left_lead.fill(
                template, lambda site: 0 <= site.tag[1] <= self.ly_leg, (0, 1)
            )
            self.attach_lead(bot_left_lead)
        elif pos.upper() == "TL":
            top_left_lead = Builder(
                TranslationalSymmetry((-1, 0)), conservation_law=conservation_law
            )
            top_left_lead.fill(
                template,
                lambda site: self.ly_leg + self.ly_neck
                <= site.tag[1]
                < self.ly_leg * 2 + self.ly_neck,
                (0, self.ly_leg + self.ly_neck),
            )
            self.attach_lead(top_left_lead)
        elif pos.upper() == "BR":
            bot_right_lead = Builder(
                TranslationalSymmetry((1, 0)), conservation_law=conservation_law
            )
            bot_right_lead.fill(
                template, lambda site: 0 <= site.tag[1] <= self.ly_leg, (0, 1)
            )
            self.attach_lead(bot_right_lead)
        elif pos.upper() == "TR":
            top_right_lead = Builder(
                TranslationalSymmetry((1, 0)), conservation_law=conservation_law
            )
            top_right_lead.fill(
                template,
                lambda site: self.ly_leg + self.ly_neck
                <= site.tag[1]
                < self.ly_leg * 2 + self.ly_neck,
                (0, self.ly_leg + self.ly_neck),
            )
            self.attach_lead(top_right_lead)
        else:
            raise ValueError(f"pos can only be BL, TL, BR, TR (case non-sensitive)")

    def ham_bfield(self, Bfields, num_ev):
        """Find the eigenvalues for each magnetic field value in Bfields. Make sure magnetic
        field B is implemented in Hamiltonian with symbol B"""
        if self.H == {}:
            return f"This Hbar system does not contain any sites."
        energies = []
        for B in Bfields:
            ham_mat = self.hamiltonin_submatrix(params=dict(B=B), sparse=True)
            ev = sla.eigsh(
                ham_mat.tocsc(), k=num_ev, sigma=0, return_eigenvectors=False
            )
            energies.append(ev)
        return energies

    def set_ham_params(self, params):
        self.ham_params = params
