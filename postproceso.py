"""
postproceso.py — Resultados de ingeniería a partir del campo de carga h.

Fase E del plan. Consume una Solucion de flujo.py y produce:
  - velocidades de Darcy y gradientes
  - caudal (ya disponible por balance nodal; aquí se complementa)
  - gradiente de salida i_exit aguas abajo y seguridad frente a sifonamiento
  - subpresiones (presion intersticial) y resultantes
  - [psi: funcion de corriente, en modulo aparte por su formulacion dual]

Convencion: cotas absolutas, z hacia arriba. u = gamma_w * (h - z).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from malla import Malla, TipoNodo
from flujo import Solucion, caudal_entrada


GAMMA_W = 9.81  # kN/m3


# --------------------------------------------------------------------------- #
#  Gradientes y velocidades de Darcy (por celda)                               #
# --------------------------------------------------------------------------- #
def campo_gradiente(sol: Solucion):
    """
    Gradiente hidraulico i = -grad(h) y velocidad de Darcy v = -k grad(h),
    evaluados en el centro de cada celda. Devuelve arrays (nx-1, nz-1):
    ix, iz (gradiente), vx, vz (velocidad), imod, vmod (modulos).
    """
    m = sol.m
    nx, nz = m.nx, m.nz
    ix = np.full((nx - 1, nz - 1), np.nan)
    iz = np.full((nx - 1, nz - 1), np.nan)
    vx = np.full((nx - 1, nz - 1), np.nan)
    vz = np.full((nx - 1, nz - 1), np.nan)

    for ic in range(nx - 1):
        dx = m.x[ic + 1] - m.x[ic]
        for jc in range(nz - 1):
            k_idx = m.capa_celda[ic, jc]
            if k_idx < 0:
                continue
            dz = m.z[jc + 1] - m.z[jc]
            # h en las 4 esquinas (usar valores de nodo; en pantalla, el lado
            # que corresponda segun la celda)
            h00 = _h_celda(sol, ic, jc, ic, jc)
            h10 = _h_celda(sol, ic, jc, ic + 1, jc)
            h01 = _h_celda(sol, ic, jc, ic, jc + 1)
            h11 = _h_celda(sol, ic, jc, ic + 1, jc + 1)
            if any(np.isnan(v) for v in (h00, h10, h01, h11)):
                continue
            # gradientes centrados en la celda
            dhdx = 0.5 * ((h10 - h00) + (h11 - h01)) / dx
            dhdz = 0.5 * ((h01 - h00) + (h11 - h10)) / dz
            ix[ic, jc] = -dhdx
            iz[ic, jc] = -dhdz
            cap = m.contorno.capas[k_idx]
            vx[ic, jc] = -cap.kx * dhdx
            vz[ic, jc] = -cap.kz * dhdz

    imod = np.sqrt(ix**2 + iz**2)
    vmod = np.sqrt(vx**2 + vz**2)
    return dict(ix=ix, iz=iz, vx=vx, vz=vz, imod=imod, vmod=vmod)


def _h_celda(sol, ic, jc, i, j):
    """h del nodo (i,j) visto desde la celda (ic,jc): si el nodo esta
    desdoblado, toma el lado segun la celda este a izq/der de la pantalla."""
    m = sol.m
    if sol.dm.is_split(i, j):
        lado = 'L' if (ic + 1 == sol.dm.i_tab) else (
               'R' if ic == sol.dm.i_tab else None)
        if lado == 'L':
            return sol.h_nodo_L[i, j]
        elif lado == 'R':
            return sol.h_nodo_R[i, j]
        return sol.h_nodo[i, j]
    return sol.h_nodo[i, j]


# --------------------------------------------------------------------------- #
#  Gradiente de salida y sifonamiento                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoSifonamiento:
    i_exit_max: float
    i_exit_medio: float
    i_critico: float | None
    FS: float | None
    metodo_icrit: str


def gradiente_salida(sol: Solucion, gamma_sat: float | None = None,
                     G_s: float | None = None, e: float | None = None,
                     gamma_w: float = GAMMA_W) -> ResultadoSifonamiento:
    """
    Gradiente de salida vertical en la cara de salida aguas abajo, junto a la
    tablestaca, y factor de seguridad frente a sifonamiento (Terzaghi).

    i_exit se evalua como el gradiente vertical en las celdas adyacentes al
    lecho de aguas abajo, en la franja junto a la pantalla (de ancho = la
    profundidad de empotramiento, criterio del prisma de Terzaghi).
    """
    m = sol.m
    c = m.contorno
    grad = campo_gradiente(sol)
    iz = grad['iz']

    # Lado de aguas abajo y su lecho
    lado_abajo_izq = not c.lado_arriba_izq
    x_lo, x_hi = c.x_abajo
    z_lecho = c.z_lecho_abajo
    s = z_lecho - c.z_pie   # empotramiento

    # franja de ancho s junto a la pantalla, en el lado de aguas abajo
    if lado_abajo_izq:
        # aguas abajo a la izquierda: franja [x_tab - s, x_tab]
        xf_lo, xf_hi = c.x_tablestaca - s, c.x_tablestaca
    else:
        xf_lo, xf_hi = c.x_tablestaca, c.x_tablestaca + s

    # celdas justo por debajo del lecho de salida en esa franja
    valores = []
    for ic in range(m.nx - 1):
        xc = 0.5 * (m.x[ic] + m.x[ic + 1])
        if not (xf_lo - 1e-9 <= xc <= xf_hi + 1e-9):
            continue
        # celda inmediatamente bajo el lecho
        for jc in range(m.nz - 1):
            zc = 0.5 * (m.z[jc] + m.z[jc + 1])
            if z_lecho - (m.z[jc + 1] - m.z[jc]) - 1e-9 <= zc < z_lecho:
                if not np.isnan(iz[ic, jc]):
                    # gradiente vertical ascendente = -iz si iz apunta... 
                    # i = -grad h; flujo ascendente -> componente iz positiva
                    valores.append(iz[ic, jc])
                break

    valores = np.array(valores)
    if len(valores) == 0:
        i_exit_max = i_exit_medio = float('nan')
    else:
        # el gradiente de salida es ascendente (magnitud del componente vertical)
        i_exit_max = float(np.max(np.abs(valores)))
        i_exit_medio = float(np.mean(np.abs(valores)))

    # gradiente critico
    i_crit = None
    metodo = "no evaluado"
    if G_s is not None and e is not None:
        i_crit = (G_s - 1.0) / (1.0 + e)
        metodo = f"(G_s-1)/(1+e) con G_s={G_s}, e={e}"
    elif gamma_sat is not None:
        gamma_sumergido = gamma_sat - gamma_w
        i_crit = gamma_sumergido / gamma_w
        metodo = f"gamma'/gamma_w con gamma_sat={gamma_sat}"

    FS = (i_crit / i_exit_max) if (i_crit and i_exit_max and
                                   not np.isnan(i_exit_max)) else None

    return ResultadoSifonamiento(
        i_exit_max=i_exit_max, i_exit_medio=i_exit_medio,
        i_critico=i_crit, FS=FS, metodo_icrit=metodo)


# --------------------------------------------------------------------------- #
#  Subpresiones                                                                #
# --------------------------------------------------------------------------- #
def presion_intersticial(sol: Solucion) -> np.ndarray:
    """u(x,z) = gamma_w * (h - z) en cada nodo. (nx, nz), NaN si FUERA."""
    m = sol.m
    u = np.full((m.nx, m.nz), np.nan)
    for i in range(m.nx):
        for j in range(m.nz):
            h = sol.h_nodo[i, j]
            if not np.isnan(h):
                u[i, j] = GAMMA_W * (h - m.z[j])
    return u


def lineas_corriente(sol, n_x=200, n_z=80):
    """
    Prepara los campos de velocidad interpolados a malla regular para trazar
    líneas de corriente por streamplot. Devuelve (xr, zr, VX, VZ).

    Las líneas de corriente así obtenidas son tangentes al campo de velocidad
    de Darcy v = -k·grad(h) y por tanto perpendiculares a las equipotenciales
    por construcción física (no dependen de resolver psi).
    """
    from scipy.interpolate import RegularGridInterpolator
    m = sol.m
    c = m.contorno
    grad = campo_gradiente(sol)
    xc = 0.5 * (m.x[:-1] + m.x[1:])
    zc = 0.5 * (m.z[:-1] + m.z[1:])
    vx = np.nan_to_num(grad["vx"].T)
    vz = np.nan_to_num(grad["vz"].T)
    fx = RegularGridInterpolator((zc, xc), vx, bounds_error=False, fill_value=0)
    fz = RegularGridInterpolator((zc, xc), vz, bounds_error=False, fill_value=0)
    xr = np.linspace(c.x_izq, c.x_der, n_x)
    zr = np.linspace(c.z_imp, c.z_sup, n_z)
    ZZ, XX = np.meshgrid(zr, xr, indexing="ij")
    pts = np.column_stack([ZZ.ravel(), XX.ravel()])
    VX = fx(pts).reshape(ZZ.shape)
    VZ = fz(pts).reshape(ZZ.shape)
    return xr, zr, VX, VZ


def subpresion_en_cota(sol: Solucion, z_base: float) -> dict:
    """
    Distribucion de presion intersticial u a lo largo de una cota horizontal
    z_base (p.ej. base de una estructura), y su resultante por integracion.
    Devuelve x[], u[], resultante [kN/m].
    """
    m = sol.m
    j = int(np.argmin(np.abs(m.z - z_base)))
    xs, us = [], []
    for i in range(m.nx):
        h = sol.h_nodo[i, j]
        if not np.isnan(h):
            xs.append(m.x[i])
            us.append(GAMMA_W * (h - m.z[j]))
    xs = np.array(xs); us = np.array(us)
    resultante = np.trapezoid(us, xs) if len(xs) > 1 else 0.0
    return dict(x=xs, u=us, resultante=resultante, z_base=m.z[j])
