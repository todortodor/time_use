#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:20:43 2026

@author: slepot
"""



# classes.py
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from functions import (
    clamp, safe_pow, EPS,
    ces_unit_cost, pigl_B, pigl_lambda, pigl_shares, bisect_root
)


@dataclass(frozen=True)
class ModelParams:
    """
    True (global) parameters.
    These are exactly what appear in the equilibrium system and definitions. :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}
    """
    # PIGL
    eps_engel: float
    beta_x: float
    beta_c: float
    beta_d: float
    kappa_x: float
    kappa_c: float
    kappa_d: float

    # Home-market CES
    omega_c: float
    omega_d: float
    eta_c: float
    eta_d: float

    # Disutility
    D_M: float
    D_c: float
    D_d: float
    rho: float   # <0 in baseline description :contentReference[oaicite:14]{index=14}
    phi: float


@dataclass
class Household:
    """
    Household object.
    You will solve over allocations (L_M, L_c, L_d) only.
    Everything else is computed internally and stored as attributes. :contentReference[oaicite:15]{index=15}
    """
    params: ModelParams

    # Exogenous variables (given to household)
    y: float    # effective wage
    pc: float   # market price care
    pd: float   # market price domestic
    a: float    # non-labor income

    # Home productivity (can be HH-specific)
    A_c: float = 1.0
    A_d: float = 1.0

    # ---- stored state (filled by evaluate_from_labor) ----
    LM: float = np.nan
    Lc: float = np.nan
    Ld: float = np.nan

    # Derived quantities
    E: float = np.nan
    L: float = np.nan
    lam: float = np.nan

    PcH: float = np.nan
    PdH: float = np.nan
    Pc: float = np.nan
    Pd: float = np.nan
    B: float = np.nan

    th_x: float = np.nan
    th_c: float = np.nan
    th_d: float = np.nan

    x: float = np.nan
    Sc: float = np.nan
    Sd: float = np.nan

    # Implied input splits (from conditional demands)
    ScH: float = np.nan
    SdH: float = np.nan
    ScM: float = np.nan
    SdM: float = np.nan

    # Residuals for the 3-equation labor-only system
    residuals: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))

    # --------------------------
    # Section 4.4 calculations
    # --------------------------

    def set_labor(self, LM: float, Lc: float, Ld: float) -> None:
        self.LM, self.Lc, self.Ld = float(LM), float(Lc), float(Ld)

    def compute_E(self) -> None:
        # (D1) E = y L_M + a :contentReference[oaicite:16]{index=16}
        self.E = float(self.y * self.LM + self.a)

    def compute_L_aggregate(self) -> None:
        """
        (D3) L = [ (D_M L_M)^{(ρ-1)/ρ} + (D_c L_c)^{(ρ-1)/ρ} + (D_d L_d)^{(ρ-1)/ρ} ]^{ρ/(ρ-1)}
        (The paper writes D_M(L_M) etc; this matches the marginal disutility form.) :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}
        """
        rho = self.params.rho
        r = (rho - 1.0) / rho
        zM = clamp(self.params.D_M * self.LM)
        zc = clamp(self.params.D_c * self.Lc)
        zd = clamp(self.params.D_d * self.Ld)
        inside = safe_pow(zM, r) + safe_pow(zc, r) + safe_pow(zd, r)
        self.L = float(safe_pow(inside, rho / (rho - 1.0)))

    def shadow_price_home(self, Li: float, Di: float, Ai: float, lam: float) -> float:
        """
        (D4) P_i^H = L^{1/φ + 1/ρ} * (D_i L_i)^(-1/ρ) / (A_i λ) :contentReference[oaicite:19]{index=19}
        """
        rho = self.params.rho
        phi = self.params.phi
        L_term = safe_pow(clamp(self.L), (1.0 / phi + 1.0 / rho))
        zi = clamp(Di * Li)
        return float(L_term * safe_pow(zi, -1.0 / rho) / (clamp(Ai) * clamp(lam)))

    def _prices_given_lambda(self, lam: float) -> tuple[float, float, float, float, float]:
        """
        Given λ, compute PcH,PdH,Pc,Pd,B.
        Uses (D4)-(D6). :contentReference[oaicite:20]{index=20}
        """
        PcH = self.shadow_price_home(self.Lc, self.params.D_c, self.A_c, lam)
        PdH = self.shadow_price_home(self.Ld, self.params.D_d, self.A_d, lam)

        Pc = ces_unit_cost(PcH, self.pc, self.params.omega_c, self.params.eta_c)  # (D5)
        Pd = ces_unit_cost(PdH, self.pd, self.params.omega_d, self.params.eta_d)  # (D5)
        B = pigl_B(Pc, Pd, self.params.beta_c, self.params.beta_d)                # (D6)
        return PcH, PdH, Pc, Pd, B

    def solve_lambda(self) -> None:
        """
        Solve for λ from:
          λ = 1/B(λ) * (E/B(λ))^{ε-1}
        where B depends on Pc,Pd which depend on λ through PcH,PdH.
        This is exactly (C1) + (D4)-(D6). :contentReference[oaicite:21]{index=21} :contentReference[oaicite:22]{index=22}
        """
        self.compute_E()
        self.compute_L_aggregate()

        eps = self.params.eps_engel

        def f(lam: float) -> float:
            _, _, Pc, Pd, B = self._prices_given_lambda(lam)
            lam_rhs = pigl_lambda(self.E, B, eps)
            return lam - lam_rhs

        # Build a bracket robustly: start wide and expand if needed
        lo, hi = 1e-12, 1e6
        flo = f(lo)
        fhi = f(hi)
        # Expand hi until sign change or cap iterations
        it = 0
        while flo * fhi > 0 and it < 60:
            hi *= 10.0
            fhi = f(hi)
            it += 1

        if flo * fhi > 0:
            raise RuntimeError("Could not bracket λ root; check parameters/inputs.")

        self.lam = float(bisect_root(f, lo, hi))

        # Store prices and B at the solved lambda
        self.PcH, self.PdH, self.Pc, self.Pd, self.B = self._prices_given_lambda(self.lam)

    def compute_pigl_demands(self) -> None:
        """
        Compute shares and total demands:
          ϑ_n = β_n + κ_n (E/B)^(-ε)  (D7)
          x = ϑx E;  Sc = ϑc E / Pc;  Sd = ϑd E / Pd  (P1)-(P3)
        :contentReference[oaicite:23]{index=23}
        """
        self.th_x, self.th_c, self.th_d = pigl_shares(
            self.E, self.B, self.params.eps_engel,
            self.params.beta_x, self.params.beta_c, self.params.beta_d,
            self.params.kappa_x, self.params.kappa_c, self.params.kappa_d
        )
        self.x = float(self.th_x * self.E)
        self.Sc = float(self.th_c * self.E / clamp(self.Pc))
        self.Sd = float(self.th_d * self.E / clamp(self.Pd))

    def compute_input_splits(self) -> None:
        """
        Conditional demands for inputs:
          S_iM = (1-ω_i) (P_i/p_i)^{η_i} S_i   (37)
          S_iH = ω_i (P_i/P_iH)^{η_i} S_i     (38)
        and identity S_iH = A_i L_i. :contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25}
        """
        self.ScM = float((1.0 - self.params.omega_c) * safe_pow(self.Pc / clamp(self.pc), self.params.eta_c) * self.Sc)
        self.SdM = float((1.0 - self.params.omega_d) * safe_pow(self.Pd / clamp(self.pd), self.params.eta_d) * self.Sd)

        self.ScH = float(self.params.omega_c * safe_pow(self.Pc / clamp(self.PcH), self.params.eta_c) * self.Sc)
        self.SdH = float(self.params.omega_d * safe_pow(self.Pd / clamp(self.PdH), self.params.eta_d) * self.Sd)

    # --------------------------
    # Labor-only equilibrium residuals
    # --------------------------

    def compute_labor_residuals(self) -> None:
        """
        3-equation system in (L_M, L_c, L_d):

        (1) Market work FOC (L1): λ y = L^{1/φ + 1/ρ} (D_M L_M)^(-1/ρ) :contentReference[oaicite:26]{index=26}
        (2) Care home-input identity: A_c L_c = S_cH (where S_cH is conditional home demand) :contentReference[oaicite:27]{index=27} :contentReference[oaicite:28]{index=28}
        (3) Domestic home-input identity: A_d L_d = S_dH :contentReference[oaicite:29]{index=29} :contentReference[oaicite:30]{index=30}
        """
        rho = self.params.rho
        phi = self.params.phi

        # LHS/RHS for L1
        L_term = safe_pow(clamp(self.L), (1.0 / phi + 1.0 / rho))
        rhs_L1 = L_term * safe_pow(clamp(self.params.D_M * self.LM), -1.0 / rho)
        eq1 = self.lam * self.y - rhs_L1

        # Home input identities via conditional home demand (38) + S_iH = A_i L_i (5)
        eq2 = self.A_c * self.Lc - self.ScH
        eq3 = self.A_d * self.Ld - self.SdH

        self.residuals = np.array([eq1, eq2, eq3], dtype=float)

    def evaluate_from_labor(self, LM: float, Lc: float, Ld: float) -> np.ndarray:
        """
        One call to evaluate everything and return residual vector in R^3.
        Also stores *all* intermediate objects as attributes.
        """
        self.set_labor(LM, Lc, Ld)
        self.solve_lambda()
        self.compute_pigl_demands()
        self.compute_input_splits()
        self.compute_labor_residuals()
        return self.residuals
