#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions.py  –  pure numeric helpers (no model state).

All functions are stateless and importable without side effects.
"""
from __future__ import annotations

import math
import numpy as np

EPS = 1e-12


# ─────────────────────────────────────────
# Basic numeric guards
# ─────────────────────────────────────────

def clamp(x: float, lo: float = EPS, hi: float = 1e300) -> float:
    return float(min(max(x, lo), hi))


def safe_pow(x: float, p: float) -> float:
    return float(clamp(x) ** p)


# ─────────────────────────────────────────
# CES / PIGL building blocks
# ─────────────────────────────────────────

def ces_unit_cost(p_home: float, p_market: float, omega: float, eta: float) -> float:
    """
    Composite price (unit cost of service i):
      P_i = [ ω p_H^(1-η) + (1-ω) p_M^(1-η) ]^(1/(1-η))
    Cobb-Douglas limit when η → 1.
    Matches (D5) / equation (32) of the model.
    """
    p_home   = clamp(p_home)
    p_market = clamp(p_market)
    omega    = clamp(omega, EPS, 1 - EPS)

    if abs(eta - 1.0) < 1e-10:          # Cobb-Douglas limit
        return float((p_home ** omega) * (p_market ** (1.0 - omega)))

    inside = omega * safe_pow(p_home, 1.0 - eta) + (1.0 - omega) * safe_pow(p_market, 1.0 - eta)
    return float(safe_pow(inside, 1.0 / (1.0 - eta)))


def pigl_B(Pc: float, Pd: float, beta_c: float, beta_d: float) -> float:
    """
    B(p) = Pc^βc · Pd^βd  (goods price = 1 is the numeraire).
    Matches (D6) / equation (3).
    """
    return float(clamp(Pc) ** beta_c * clamp(Pd) ** beta_d)


def pigl_lambda(E: float, B: float, eps_engel: float) -> float:
    """
    λ = (1/B) · (E/B)^{ε-1}.
    Matches (C1) / equation (13).
    """
    return float((1.0 / clamp(B)) * safe_pow(clamp(E) / clamp(B), eps_engel - 1.0))


def pigl_shares(E: float, B: float, eps_engel: float,
                beta_x: float, beta_c: float, beta_d: float,
                kappa_x: float, kappa_c: float, kappa_d: float,
                ) -> tuple[float, float, float]:
    """
    ϑ_n = β_n + κ_n · (E/B)^{-ε}.

    Returns raw model-implied shares without clamping.
    If shares are negative, this signals a unit inconsistency between
    the calibration (E in 1000-KSh) and the solver — fix at source,
    not here.
    """
    real = clamp(E) / clamp(B)
    adj  = safe_pow(real, -eps_engel)
    return (float(beta_x + kappa_x * adj),
            float(beta_c + kappa_c * adj),
            float(beta_d + kappa_d * adj))


# ─────────────────────────────────────────
# Gender-specific CES labor aggregate
# ─────────────────────────────────────────

def ces_labor_aggregate(LM: float, Lc: float, Ld: float,
                        D_M: float, D_c: float, D_d: float,
                        rho: float) -> float:
    """
    L^g = [ D_M^g (L_M^g)^{(ρ-1)/ρ} + D_c^g (L_c^g)^{(ρ-1)/ρ}
                                      + D_d^g (L_d^g)^{(ρ-1)/ρ} ]^{ρ/(ρ-1)}
    Equation (7) applied to one gender.
    """
    r     = (rho - 1.0) / rho
    zM    = clamp(D_M * LM)
    zc    = clamp(D_c * Lc)
    zd    = clamp(D_d * Ld)
    inner = safe_pow(zM, r) + safe_pow(zc, r) + safe_pow(zd, r)
    return float(safe_pow(inner, rho / (rho - 1.0)))


# ─────────────────────────────────────────
# Root-finding
# ─────────────────────────────────────────

def bisect_root(f, lo: float, hi: float,
                max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Robust bisection. Requires f(lo)·f(hi) < 0.
    """
    flo = f(lo)
    fhi = f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        raise ValueError("bisect_root: non-finite endpoint values.")
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("bisect_root: root not bracketed.")

    for _ in range(max_iter):
        mid  = 0.5 * (lo + hi)
        fmid = f(mid)
        if not math.isfinite(fmid):
            mid  = 0.5 * (mid + lo)
            fmid = f(mid)
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)
