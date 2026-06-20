"""
corriente.py — Función de corriente psi (depurada) y caudal por secciones.

ψ resuelve el problema DUAL del de la carga h:

    d/dx( (1/kz) dψ/dx ) + d/dz( (1/kx) dψ/dz ) = 0

OJO: coeficientes INTERCAMBIADOS respecto a h (en x va 1/kz, en z va 1/kx).
Para medio isótropo (kx=kz=k) ambos son 1/k.

Condiciones de contorno (duales a las de h):
  - Contornos IMPERMEABLES = líneas de corriente -> ψ = const (Dirichlet):
      * fondo impermeable + borde izquierdo + borde derecho  -> ψ = 0
      * pantalla (ambas caras + pie)                          -> ψ = Q
  - Contornos de CARGA (lechos, equipotenciales) -> Neumann natural.

ψ es CONTINUA a través de la pantalla (no se desdoblan nodos, a diferencia de h).

Caudal por sección = diferencia de ψ. El caudal que cruza la franja vertical
bajo el pie entre z_corte y z_pie es  q = ψ(z_pie) - ψ(z_corte).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from malla import TipoNodo
from flujo import Solucion, caudal_entrada


def resolver_psi(sol: Solucion):
    """Resuelve psi. Devuelve (psi_nodo (nx,nz), Q). NaN en nodos FUERA."""
    m = sol.m
    c = m.contorno
    Q, _, _ = caudal_entrada(sol)

    dofid = -np.ones((m.nx, m.nz), dtype=int)
    n = 0
    for i in range(m.nx):
        for j in range(m.nz):
            if m.tipo_nodo[i, j] != TipoNodo.FUERA:
                dofid[i, j] = n
                n += 1

    A = sp.lil_matrix((n, n))

    def add(da, db, C):
        if da < 0 or db < 0 or C == 0:
            return
        A[da, da] -= C; A[da, db] += C
        A[db, db] -= C; A[db, da] += C

    for ic in range(m.nx - 1):
        dx = m.x[ic + 1] - m.x[ic]
        for jc in range(m.nz - 1):
            k_idx = m.capa_celda[ic, jc]
            if k_idx < 0:
                continue
            cap = c.capas[k_idx]
            dz = m.z[jc + 1] - m.z[jc]
            cx = 1.0 / cap.kz   # intercambiado
            cz = 1.0 / cap.kx
            Ch = cx * (dz / 2.0) / dx
            Cv = cz * (dx / 2.0) / dz
            add(dofid[ic, jc],     dofid[ic + 1, jc],     Ch)
            add(dofid[ic, jc + 1], dofid[ic + 1, jc + 1], Ch)
            add(dofid[ic, jc],     dofid[ic, jc + 1],     Cv)
            add(dofid[ic + 1, jc], dofid[ic + 1, jc + 1], Cv)

    b = np.zeros(n)
    A = A.tolil()
    fijos = {}

    # ψ = 0 : fondo + bordes laterales
    j_fondo = int(np.argmin(np.abs(m.z - c.z_imp)))
    for i in range(m.nx):
        if dofid[i, j_fondo] >= 0:
            fijos[dofid[i, j_fondo]] = 0.0
    for j in range(m.nz):
        for i in (0, m.nx - 1):
            if dofid[i, j] >= 0:
                fijos.setdefault(dofid[i, j], 0.0)

    # ψ = Q : pantalla
    i_tab = int(np.argmin(np.abs(m.x - c.x_tablestaca)))
    z_lecho_max = max(c.z_lecho_arriba, c.z_lecho_abajo)
    for j in range(m.nz):
        zj = m.z[j]
        if c.z_pie - 1e-9 <= zj <= z_lecho_max + 1e-9:
            if dofid[i_tab, j] >= 0:
                fijos[dofid[i_tab, j]] = Q

    for d, v in fijos.items():
        col = A[:, d].toarray().ravel()
        b -= col * v
        A[d, :] = 0; A[:, d] = 0; A[d, d] = 1; b[d] = v

    psi = spsolve(A.tocsr(), b)
    psi_nodo = np.full((m.nx, m.nz), np.nan)
    for i in range(m.nx):
        for j in range(m.nz):
            if dofid[i, j] >= 0:
                psi_nodo[i, j] = psi[dofid[i, j]]
    return psi_nodo, Q


def caudal_franja_bajo_pie(sol, psi_nodo, profundidad):
    """
    Caudal que cruza la franja vertical bajo el pie, desde z_pie hasta
    'profundidad' metros por debajo. q = ψ(z_pie) - ψ(z_corte), exacto.
    """
    m = sol.m
    c = m.contorno
    z_pie = c.z_pie
    z_max_prof = z_pie - c.z_imp
    if profundidad <= 0 or profundidad > z_max_prof + 1e-9:
        raise ValueError(
            f"profundidad en (0, {z_max_prof:.3f}] m (distancia pie-impermeable).")
    z_corte = z_pie - profundidad
    i_tab = int(np.argmin(np.abs(m.x - c.x_tablestaca)))
    psi_col = psi_nodo[i_tab, :]
    z = m.z
    psi_clean = np.where(np.isnan(psi_col), 0.0, psi_col)

    def psi_en(zq):
        return float(np.interp(zq, z, psi_clean))

    q = abs(psi_en(z_pie) - psi_en(z_corte))
    Q_total = abs(np.nanmax(psi_nodo))
    return dict(z_corte=z_corte, profundidad=profundidad, caudal=q,
                fraccion=q / Q_total if Q_total else float('nan'),
                Q_total=Q_total)
