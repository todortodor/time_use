#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:20:43 2026

@author: slepot
"""


# functions.py
from __future__ import annotations

import math
import numpy as np

EPS = 1e-12


def clamp(x: float, lo: float = EPS, hi: float = 1e300) -> float:
    return float(min(max(x, lo), hi))


def safe_pow(x: float, p: float) -> float:
    # for our use, x should be positive; clamp protects logs/powers
    return float(clamp(x) ** p)


def ces_unit_cost(p_home: float, p_market: float, omega: float, eta: float) -> float:
    """
    Composite price (unit cost):
      P = [ ω p_home^(1-η) + (1-ω) p_market^(1-η) ]^(1/(1-η))
    Matches (D5). :contentReference[oaicite:7]{index=7}
    """
    p_home = clamp(p_home)
    p_market = clamp(p_market)
    omega = clamp(omega, EPS, 1 - EPS)

    if abs(eta - 1.0) < 1e-10:
        # Cobb–Douglas limit
        return float((p_home ** omega) * (p_market ** (1.0 - omega)))

    inside = omega * safe_pow(p_home, 1.0 - eta) + (1.0 - omega) * safe_pow(p_market, 1.0 - eta)
    return float(safe_pow(inside, 1.0 / (1.0 - eta)))


def pigl_B(Pc: float, Pd: float, beta_c: float, beta_d: float) -> float:
    """
    B(p) = Pc^{βc} Pd^{βd}, goods price normalized to 1. Matches (D6)/(2). :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    """
    Pc = clamp(Pc)
    Pd = clamp(Pd)
    return float((Pc ** beta_c) * (Pd ** beta_d))


def pigl_lambda(E: float, B: float, eps_engel: float) -> float:
    """
    λ = 1/B * (E/B)^{ε-1}. Matches (C1)/(39). :contentReference[oaicite:10]{index=10}
    """
    E = clamp(E)
    B = clamp(B)
    return float((1.0 / B) * safe_pow(E / B, eps_engel - 1.0))


def pigl_shares(E: float, B: float, eps_engel: float,
               beta_x: float, beta_c: float, beta_d: float,
               kappa_x: float, kappa_c: float, kappa_d: float) -> tuple[float, float, float]:
    """
    ϑ_n = β_n + κ_n (E/B)^(-ε). Matches (D7)/(51). :contentReference[oaicite:11]{index=11}
    """
    E = clamp(E)
    B = clamp(B)
    real = E / B
    adj = safe_pow(real, -eps_engel)
    th_x = beta_x + kappa_x * adj
    th_c = beta_c + kappa_c * adj
    th_d = beta_d + kappa_d * adj
    return float(th_x), float(th_c), float(th_d)


def bisect_root(f, lo: float, hi: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Simple robust bisection. Requires f(lo) and f(hi) of opposite signs.
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
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if not math.isfinite(fmid):
            # if mid is bad, shrink interval a bit
            mid = 0.5 * (mid + lo)
            fmid = f(mid)

        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid

        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)
