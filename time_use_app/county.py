#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
county.py  –  county-level objects for the spatial equilibrium model.

Each county ℓ is characterised by:
  • Sector TFP  A^n_ℓ   for  n ∈ {x, c, d}
  • Market wage w_ℓ     (single labour market per county; sectoral wages
                          pin down goods/service prices via zero-profit)
  • Local prices p^c_ℓ, p^d_ℓ  (backed out from zero-profit + TFP)
  • Home production productivities  A^c_ℓ, A^d_ℓ
  • Population  N_ℓ  (observed; used to back out amenity)
  • Amenity  ξ_ℓ  (residual from spatial equilibrium condition)

Zero-profit conditions (Section 5.2 of model write-up):
    w^n_ℓ = p^n_ℓ · A^n_ℓ
With  p^x_ℓ = 1  (numeraire):
    w^x_ℓ = A^x_ℓ
    p^c_ℓ = w^c_ℓ / A^c_ℓ
    p^d_ℓ = w^d_ℓ / A^d_ℓ

In the simplest calibration a single county wage w_ℓ applies to all
sectors; sectoral wages then equal  w^n_ℓ = w_ℓ  and prices follow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from functions import clamp


# ═══════════════════════════════════════════════════════════════════════════════
# County data object
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class County:
    """
    All county-ℓ fundamentals needed by the spatial equilibrium.

    Prices are derived from wages and TFP via zero-profit; the user
    may supply them directly instead (set derive_prices=False).
    """
    # ── identifier ────────────────────────────────────────────────────────────
    name: str = "county"

    # ── labour market ─────────────────────────────────────────────────────────
    # Single county wage (applies to all sectors in baseline).
    # Effective wages for man/woman:  y^g_ℓ = w_ell * h^g
    w_ell: float = 20.0

    # ── sector TFP ────────────────────────────────────────────────────────────
    A_x: float = 1.0   # goods TFP       (also = w^x_ℓ under zero-profit + num.)
    A_c: float = 1.0   # care TFP        (market sector)
    A_d: float = 1.0   # domestic TFP    (market sector)

    # ── home production TFP (county-level, gender-neutral) ────────────────────
    A_c_home: float = 1.0
    A_d_home: float = 1.0

    # ── prices (derived or supplied directly) ─────────────────────────────────
    # If derive_prices=True, p^c and p^d are computed from zero-profit below.
    derive_prices: bool = True
    _pc: float = field(default=np.nan, repr=False)
    _pd: float = field(default=np.nan, repr=False)

    # ── observed population ───────────────────────────────────────────────────
    N: float = 1.0     # number of households (for market clearing weight)

    # ── amenity (filled by spatial equilibrium) ───────────────────────────────
    xi: float = np.nan

    # ── equilibrium household value (filled by spatial equilibrium) ───────────
    # V*_ℓ ≡ V(p_ℓ, E*_ℓ) − D(L*_ℓ)  at the optimal household allocation
    V_star: float = np.nan

    # ─────────────────────────────────────────────────────────────────────────
    # Derived prices
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def pc(self) -> float:
        """
        Market price of care services in county ℓ.
        Zero-profit:  p^c_ℓ = w_ℓ / A^c_ℓ
        (w_ell proxies the care-sector wage in the single-wage baseline)
        """
        if self.derive_prices:
            return float(self.w_ell / clamp(self.A_c))
        return float(self._pc)

    @pc.setter
    def pc(self, value: float) -> None:
        self._pc = float(value)
        self.derive_prices = False

    @property
    def pd(self) -> float:
        """
        Market price of domestic services in county ℓ.
        Zero-profit:  p^d_ℓ = w_ℓ / A^d_ℓ
        """
        if self.derive_prices:
            return float(self.w_ell / clamp(self.A_d))
        return float(self._pd)

    @pd.setter
    def pd(self, value: float) -> None:
        self._pd = float(value)
        self.derive_prices = False

    # ─────────────────────────────────────────────────────────────────────────
    # Effective wages for man and woman
    # ─────────────────────────────────────────────────────────────────────────

    def y_m(self, h_m: float, wage_gap: float = 1.0) -> float:
        """Effective wage of man:  y^m_ℓ = w_ℓ · h_m"""
        return float(self.w_ell * h_m)

    def y_f(self, h_f: float, wage_gap: float = 1.0) -> float:
        """Effective wage of woman:  y^f_ℓ = wage_gap · w_ℓ · h_f"""
        return float(wage_gap * self.w_ell * h_f)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience summary
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"County: {self.name}",
            f"  w_ell={self.w_ell:.4g}  A_x={self.A_x:.4g}  A_c={self.A_c:.4g}  A_d={self.A_d:.4g}",
            f"  pc={self.pc:.4g}  pd={self.pd:.4g}",
            f"  A_c_home={self.A_c_home:.4g}  A_d_home={self.A_d_home:.4g}",
            f"  N={self.N:.4g}",
            f"  xi={self.xi:.6g}" if np.isfinite(self.xi) else "  xi=<not set>",
            f"  V_star={self.V_star:.6g}" if np.isfinite(self.V_star) else "  V_star=<not set>",
        ]
        return "\n".join(lines)
