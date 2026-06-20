"""
test_suite.py — Suite consolidada de validación del programa de flujo 2D
bajo tablestaca por diferencias finitas / volúmenes finitos.

Ejecutar:  pytest test_suite.py -v

Cubre las cinco fases del motor más la lógica de la app:
  A) Generador de malla y test geométrico
  B) Motor homogéneo: 1D exacto, conservación, simetría, analítica conforme
  C) Multicapa: k_eq serie (armónica) y paralelo (aritmética)
  D) Anisotropía: flujo H ve kx, V ve kz
  E) Post-proceso: subpresiones, sifonamiento, función de corriente (Q dual)
  F) Lógica de app: conversión de unidades, espesores->cotas, huella, validación
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import ellipk
import pytest

from malla import Capa, ContornoRect, generar_malla, verificar_malla, TipoNodo
from flujo import resolver_h, caudal_entrada
from postproceso import (campo_gradiente, gradiente_salida,
                         presion_intersticial, subpresion_en_cota)


# --------------------------------------------------------------------------- #
#  Utilidades comunes                                                          #
# --------------------------------------------------------------------------- #
def caso_simetrico(densidad=None, paso=None):
    capas = [Capa(0.0, 10.0, 1e-5, 1e-5, "Arena")]
    c = ContornoRect(x_izq=0.0, x_der=40.0, x_tablestaca=20.0,
        z_coronacion=13.0, z_pie=4.0, z_lecho_arriba=10.0, z_lecho_abajo=10.0,
        h1=12.0, h2=10.5, capas=capas)
    return generar_malla(c, densidad=densidad, paso=paso), c


def _ensamblar_bloque(x, z, coef_cell):
    """Ensamblador de bloque homogéneo/heterogéneo (mismo esquema que flujo.py)."""
    nx, nz = len(x), len(z)
    def idx(i, j): return i * nz + j
    A = sp.lil_matrix((nx * nz, nx * nz))
    def addc(a, b, C): A[a, a] -= C; A[a, b] += C; A[b, b] -= C; A[b, a] += C
    for ic in range(nx - 1):
        dx = x[ic + 1] - x[ic]
        for jc in range(nz - 1):
            dz = z[jc + 1] - z[jc]
            cc = coef_cell[ic, jc]
            addc(idx(ic, jc), idx(ic + 1, jc), cc * (dz / 2) / dx)
            addc(idx(ic, jc + 1), idx(ic + 1, jc + 1), cc * (dz / 2) / dx)
            addc(idx(ic, jc), idx(ic, jc + 1), cc * (dx / 2) / dz)
            addc(idx(ic + 1, jc), idx(ic + 1, jc + 1), cc * (dx / 2) / dz)
    return A.tocsr()


def _resolver_dir(A, dirn):
    n = A.shape[0]; b = np.zeros(n); A = A.tolil(); A0 = A.tocsr().copy()
    for d, v in dirn:
        col = A[:, d].toarray().ravel(); b -= col * v
        A[d, :] = 0; A[:, d] = 0; A[d, d] = 1; b[d] = v
    return spsolve(A.tocsr(), b), A0


# =========================================================================== #
#  FASE A — Malla                                                              #
# =========================================================================== #
def test_A_malla_geometrica():
    m, c = caso_simetrico(densidad="normal")
    assert not verificar_malla(m)

def test_A_escalon_lechos():
    """Lechos a distinta cota: nodos de agua marcados como FUERA."""
    capas = [Capa(0.0, 10.0, 1e-5, 1e-5, "Arena")]
    c = ContornoRect(x_izq=0.0, x_der=40.0, x_tablestaca=20.0,
        z_coronacion=13.0, z_pie=4.0, z_lecho_arriba=10.0, z_lecho_abajo=8.0,
        h1=12.0, h2=8.5, capas=capas)
    m = generar_malla(c, densidad="normal")
    assert not verificar_malla(m)
    assert np.sum(m.tipo_nodo == TipoNodo.FUERA) > 0

def test_A_multicapa_contactos_en_malla():
    capas = [Capa(0.0, 3.0, 1e-6, 1e-6, "Arcilla"),
             Capa(3.0, 7.0, 5e-5, 5e-5, "Arena"),
             Capa(7.0, 10.0, 1e-5, 1e-5, "Limo")]
    c = ContornoRect(x_izq=0.0, x_der=40.0, x_tablestaca=20.0,
        z_coronacion=13.0, z_pie=4.0, z_lecho_arriba=10.0, z_lecho_abajo=10.0,
        h1=12.0, h2=10.5, capas=capas)
    m = generar_malla(c, densidad="normal")
    assert not verificar_malla(m)
    for zc in (3.0, 7.0):
        assert np.any(np.abs(m.z - zc) < 1e-7)
    assert sorted(set(m.capa_celda[m.capa_celda >= 0].tolist())) == [0, 1, 2]


# =========================================================================== #
#  FASE B — Motor homogéneo                                                    #
# =========================================================================== #
def test_A_escalon_grande_resuelve():
    """Escalón de lechos grande: no debe dar matriz singular (regresión)."""
    capas = [Capa(0.0, 12.0, 5e-5, 5e-5, "Arena")]
    c = ContornoRect(x_izq=0.0, x_der=50.0, x_tablestaca=25.0,
        z_coronacion=15.0, z_pie=5.0, z_lecho_arriba=12.0, z_lecho_abajo=10.0,
        h1=14.5, h2=10.0, capas=capas)
    m = generar_malla(c, densidad="normal")
    sol = resolver_h(m)
    Q, Qh1, Qh2 = caudal_entrada(sol)
    assert not np.isnan(Q) and Q > 0
    assert abs(Qh1 + Qh2) / abs(Qh1) < 1e-6   # conservación
    assert np.nanmin(sol.h_nodo) >= c.h2 - 1e-6
    assert np.nanmax(sol.h_nodo) <= c.h1 + 1e-6


def test_A_contraste_extremo_resuelve():
    """Capa casi impermeable (contraste 1e6): el solver robusto debe resolver
    sin NaN y conservar masa (regresión del caso de matriz singular)."""
    from flujo import contraste_permeabilidad
    capas = [Capa(2.0, 12.0, 6e-5, 6e-5, "Arena"),
             Capa(0.0, 2.0, 6e-11, 6e-11, "Arcilla")]
    c = ContornoRect(x_izq=0.0, x_der=50.0, x_tablestaca=25.0,
        z_coronacion=15.0, z_pie=5.0, z_lecho_arriba=12.0, z_lecho_abajo=10.0,
        h1=14.5, h2=10.0, capas=capas)
    m = generar_malla(c, densidad="normal")
    diag = contraste_permeabilidad(m)
    assert diag["nivel"] == "extremo"
    assert len(diag["capas_debiles"]) >= 1
    sol = resolver_h(m)
    Q, Qh1, Qh2 = caudal_entrada(sol)
    assert np.isfinite(Q) and Q > 0
    assert abs(Qh1 + Qh2) / abs(Qh1) < 1e-6
    assert np.nanmin(sol.h_nodo) >= c.h2 - 1e-6
    assert np.nanmax(sol.h_nodo) <= c.h1 + 1e-6


def test_B_operador_1d_exacto():
    """Flujo 1D horizontal: Q = k dH B / L exacto."""
    L, B, k, H0 = 40.0, 10.0, 1e-5, 1.5
    x = np.linspace(0, L, 41); z = np.linspace(0, B, 11); nz = len(z); nx = len(x)
    def idx(i, j): return i * nz + j
    A = _ensamblar_bloque(x, z, np.full((nx - 1, nz - 1), k))
    dirn = [(idx(0, j), H0) for j in range(nz)] + [(idx(nx - 1, j), 0.0) for j in range(nz)]
    phi, A0 = _resolver_dir(A, dirn); q = A0 @ phi
    Q = abs(sum(q[idx(0, j)] for j in range(nz)))
    assert abs(Q - k * H0 * B / L) / (k * H0 * B / L) < 1e-9

def test_B_conservacion_masa():
    m, c = caso_simetrico(densidad="fino")
    sol = resolver_h(m)
    Q, Qh1, Qh2 = caudal_entrada(sol)
    assert abs(Qh1 + Qh2) / abs(Qh1) < 1e-6

def test_B_rango_fisico():
    m, c = caso_simetrico(densidad="normal")
    sol = resolver_h(m)
    assert np.nanmin(sol.h_nodo) >= c.h2 - 1e-9
    assert np.nanmax(sol.h_nodo) <= c.h1 + 1e-9

def test_B_simetria_precision_maquina():
    m, c = caso_simetrico(densidad="fino")
    sol = resolver_h(m)
    i_tab = int(np.argmin(np.abs(m.x - c.x_tablestaca)))
    hmed = (c.h1 + c.h2) / 2
    desv = max((abs(sol.h_nodo[i_tab, j] - hmed)
                for j in range(m.nz)
                if m.z[j] < c.z_pie - 1e-9 and not np.isnan(sol.h_nodo[i_tab, j])),
               default=0.0)
    assert desv < 1e-9

def test_B_convergencia_monotona():
    Qs = []
    for paso in (1.0, 0.5, 0.25):
        m, c = caso_simetrico(paso=paso)
        Q, _, _ = caudal_entrada(resolver_h(m))
        Qs.append(Q)
    assert Qs[0] < Qs[1] < Qs[2]
    assert abs(Qs[2] - Qs[1]) < abs(Qs[1] - Qs[0])

def test_B_validacion_analitica_conforme():
    """Número de forma contra solución por transformación conforme."""
    m, c = caso_simetrico(paso=0.125)
    Q, _, _ = caudal_entrada(resolver_h(m))
    forma_num = Q / (1e-5 * (c.h1 - c.h2))
    T, s = 10.0, 6.0
    m_mod = np.sin(np.pi / 2 * s / T); mp = np.sqrt(1 - m_mod**2)
    forma_teor = ellipk(mp**2) / (2 * ellipk(m_mod**2))
    assert abs(forma_num - forma_teor) / forma_teor < 0.01


# =========================================================================== #
#  FASE C — Multicapa                                                          #
# =========================================================================== #
def test_C_serie_armonica():
    """Flujo vertical a través de capas -> k_eq armónica."""
    L1, L2, L3 = 3.0, 4.0, 3.0; k1, k2, k3 = 1e-6, 1e-4, 1e-5; B = 5.0
    z = np.linspace(0, 10, 21); x = np.linspace(0, B, 6); nz = len(z); nx = len(x)
    def kc(jc):
        zc = 0.5 * (z[jc] + z[jc + 1])
        return k1 if zc < 3 else (k2 if zc < 7 else k3)
    coef = np.array([[kc(jc) for jc in range(nz - 1)] for _ in range(nx - 1)])
    A = _ensamblar_bloque(x, z, coef)
    def idx(i, j): return i * nz + j
    dirn = [(idx(i, nz - 1), 1.0) for i in range(nx)] + [(idx(i, 0), 0.0) for i in range(nx)]
    phi, A0 = _resolver_dir(A, dirn); q = A0 @ phi
    Q = abs(sum(q[idx(i, nz - 1)] for i in range(nx)))
    keq_num = Q * 10.0 / (1.0 * B)
    keq_teor = (L1 + L2 + L3) / (L1 / k1 + L2 / k2 + L3 / k3)
    assert abs(keq_num - keq_teor) / keq_teor < 1e-6

def test_C_paralelo_aritmetica():
    """Flujo horizontal a lo largo de capas -> k_eq aritmética."""
    L1, L2, L3 = 3.0, 4.0, 3.0; k1, k2, k3 = 1e-6, 1e-4, 1e-5; Lx = 10.0
    z = np.linspace(0, 10, 21); x = np.linspace(0, Lx, 11); nz = len(z); nx = len(x)
    def kc(jc):
        zc = 0.5 * (z[jc] + z[jc + 1])
        return k1 if zc < 3 else (k2 if zc < 7 else k3)
    coef = np.array([[kc(jc) for jc in range(nz - 1)] for _ in range(nx - 1)])
    A = _ensamblar_bloque(x, z, coef)
    def idx(i, j): return i * nz + j
    dirn = [(idx(0, j), 1.0) for j in range(nz)] + [(idx(nx - 1, j), 0.0) for j in range(nz)]
    phi, A0 = _resolver_dir(A, dirn); q = A0 @ phi
    Q = abs(sum(q[idx(0, j)] for j in range(nz)))
    keq_num = Q * Lx / (1.0 * 10.0)
    keq_teor = (k1 * L1 + k2 * L2 + k3 * L3) / (L1 + L2 + L3)
    assert abs(keq_num - keq_teor) / keq_teor < 1e-6


# =========================================================================== #
#  FASE D — Anisotropía                                                        #
# =========================================================================== #
def test_D_anisotropia_horizontal_vertical():
    kx, kz = 4e-5, 1e-5; L, B, H0 = 20.0, 10.0, 2.0
    x = np.linspace(0, L, 21); z = np.linspace(0, B, 11); nz = len(z); nx = len(x)
    def idx(i, j): return i * nz + j
    # flujo horizontal: ve kx
    nx_, nz_ = nx, nz
    cellx = np.full((nx - 1, nz - 1), 0.0)  # placeholder; ensamblamos aniso a mano
    # ensamblado anisótropo directo
    def ensamblar_aniso(x, z, kx, kz):
        A = sp.lil_matrix((len(x)*len(z), len(x)*len(z)))
        def addc(a, b, C): A[a,a]-=C; A[a,b]+=C; A[b,b]-=C; A[b,a]+=C
        for ic in range(len(x)-1):
            dx = x[ic+1]-x[ic]
            for jc in range(len(z)-1):
                dz = z[jc+1]-z[jc]
                addc(idx(ic,jc), idx(ic+1,jc), kx*(dz/2)/dx)
                addc(idx(ic,jc+1), idx(ic+1,jc+1), kx*(dz/2)/dx)
                addc(idx(ic,jc), idx(ic,jc+1), kz*(dx/2)/dz)
                addc(idx(ic+1,jc), idx(ic+1,jc+1), kz*(dx/2)/dz)
        return A.tocsr()
    A = ensamblar_aniso(x, z, kx, kz)
    dirn = [(idx(0,j), H0) for j in range(nz)] + [(idx(nx-1,j), 0.0) for j in range(nz)]
    phi, A0 = _resolver_dir(A, dirn); q = A0 @ phi
    Qh = abs(sum(q[idx(0,j)] for j in range(nz)))
    assert abs(Qh - kx*H0*B/L) / (kx*H0*B/L) < 1e-9
    # flujo vertical: ve kz
    A = ensamblar_aniso(x, z, kx, kz)
    dirn = [(idx(i,nz-1), H0) for i in range(nx)] + [(idx(i,0), 0.0) for i in range(nx)]
    phi, A0 = _resolver_dir(A, dirn); q = A0 @ phi
    Qv = abs(sum(q[idx(i,nz-1)] for i in range(nx)))
    assert abs(Qv - kz*H0*L/B) / (kz*H0*L/B) < 1e-9


# =========================================================================== #
#  FASE E — Post-proceso                                                       #
# =========================================================================== #
def test_E_presion_intersticial_exacta():
    """u en el lecho aguas arriba = gamma_w (h1 - z_lecho)."""
    m, c = caso_simetrico(densidad="fino")
    sol = resolver_h(m)
    u = presion_intersticial(sol)
    j_lecho = int(np.argmin(np.abs(m.z - 10.0)))
    u_teor = 9.81 * (c.h1 - c.z_lecho_arriba)
    # nodo aguas arriba en el lecho
    i = 5
    assert abs(u[i, j_lecho] - u_teor) < 1e-6

def test_E_sifonamiento_coherente():
    m, c = caso_simetrico(densidad="fino")
    sol = resolver_h(m)
    sif = gradiente_salida(sol, G_s=2.65, e=0.6)
    assert sif.i_exit_max > sif.i_exit_medio > 0    # max > medio
    assert sif.i_critico > 1.0                       # arena tipica
    assert sif.FS > 1.0

def test_E_perpendicularidad_contornos():
    """Equipotenciales perpendiculares a contornos impermeables:
    la componente del gradiente NORMAL al contorno impermeable es ~0."""
    m, c = caso_simetrico(densidad="fino")
    sol = resolver_h(m)
    grad = campo_gradiente(sol)
    ix, iz = grad["ix"], grad["iz"]
    # fondo impermeable (z=0): componente normal = vertical iz -> ~0
    iz_f, ix_f = iz[:, 0], ix[:, 0]
    rel_fondo = np.nanmean(np.abs(iz_f) / (np.abs(ix_f) + np.abs(iz_f) + 1e-30))
    # borde lateral izquierdo (x=0): componente normal = horizontal ix -> ~0
    ix_l, iz_l = ix[0, :], iz[0, :]
    rel_lat = np.nanmean(np.abs(ix_l) / (np.abs(ix_l) + np.abs(iz_l) + 1e-30))
    # tolerancia laxa: medimos a media celda del contorno, no sobre él
    assert rel_fondo < 0.10
    assert rel_lat < 0.10


def test_E_psi_ortogonalidad_y_caudal():
    """psi depurada: ortogonalidad con equipotenciales y caudal por franja
    completa = caudal total (isótropo)."""
    from corriente import resolver_psi, caudal_franja_bajo_pie
    m, c = caso_simetrico(densidad="fino")
    sol = resolver_h(m)
    Q_nodal, _, _ = caudal_entrada(sol)
    psi, Q = resolver_psi(sol)
    # rango [0, Q]
    assert abs(np.nanmin(psi)) < 1e-12 * max(Q, 1e-30)
    assert abs(np.nanmax(psi) - Q) / Q < 1e-9
    # caudal franja completa == Q total (exacto)
    r = caudal_franja_bajo_pie(sol, psi, c.z_pie - c.z_imp)
    assert abs(r["caudal"] - Q_nodal) / Q_nodal < 1e-6
    # ortogonalidad media buena (isótropo)
    H = sol.h_nodo
    i_tab = int(np.argmin(np.abs(m.x - c.x_tablestaca)))
    cs = []
    for i in range(1, m.nx - 1):
        if abs(i - i_tab) <= 1:
            continue
        for j in range(1, m.nz - 1):
            vals = [H[i, j], psi[i, j], H[i+1, j], H[i-1, j], H[i, j+1],
                    H[i, j-1], psi[i+1, j], psi[i-1, j], psi[i, j+1], psi[i, j-1]]
            if any(np.isnan(v) for v in vals):
                continue
            dhx = (H[i+1, j]-H[i-1, j])/(m.x[i+1]-m.x[i-1])
            dhz = (H[i, j+1]-H[i, j-1])/(m.z[j+1]-m.z[j-1])
            dpx = (psi[i+1, j]-psi[i-1, j])/(m.x[i+1]-m.x[i-1])
            dpz = (psi[i, j+1]-psi[i, j-1])/(m.z[j+1]-m.z[j-1])
            n1 = np.hypot(dhx, dhz); n2 = np.hypot(dpx, dpz)
            if n1 < 1e-15 or n2 < 1e-15:
                continue
            cs.append(abs(dhx*dpx + dhz*dpz)/(n1*n2))
    assert np.mean(cs) < 0.05


def test_E_caudal_franja_monotono():
    """El caudal por franja crece con la profundidad y se satura en Q."""
    from corriente import resolver_psi, caudal_franja_bajo_pie
    m, c = caso_simetrico(densidad="normal")
    sol = resolver_h(m)
    psi, Q = resolver_psi(sol)
    qs = [caudal_franja_bajo_pie(sol, psi, p)["caudal"]
          for p in (1.0, 2.0, 3.0, 4.0)]
    assert qs[0] < qs[1] < qs[2] < qs[3]
    assert abs(qs[3] - Q) / Q < 1e-6


def test_E_caudal_dual_psi():
    """Q por delta_psi coincide con balance nodal (canal 1D)."""
    k = 2e-5; L, B, H0 = 20.0, 10.0, 3.0
    x = np.linspace(0, L, 21); z = np.linspace(0, B, 11); nz = len(z); nx = len(x)
    def idx(i, j): return i * nz + j
    # h con coef=k
    A = _ensamblar_bloque(x, z, np.full((nx-1, nz-1), k))
    dirn = [(idx(0,j), H0) for j in range(nz)] + [(idx(nx-1,j), 0.0) for j in range(nz)]
    phih, A0 = _resolver_dir(A, dirn); qh = A0 @ phih
    Q_nodal = abs(sum(qh[idx(0,j)] for j in range(nz)))
    # psi con coef=1/k
    A = _ensamblar_bloque(x, z, np.full((nx-1, nz-1), 1.0/k))
    dirn = [(idx(i,0), 0.0) for i in range(nx)] + [(idx(i,nz-1), Q_nodal) for i in range(nx)]
    phipsi, _ = _resolver_dir(A, dirn)
    delta_psi = phipsi[idx(nx//2, nz-1)] - phipsi[idx(nx//2, 0)]
    assert abs(delta_psi - Q_nodal) / Q_nodal < 1e-6


# =========================================================================== #
#  FASE F — Lógica de la app                                                   #
# =========================================================================== #
def test_F_conversion_unidades():
    import app
    assert abs(app.k_a_si(1.0, "cm/s") - 0.01) < 1e-15
    assert abs(app.caudal_desde_si(1e-3, "l/s/m") - 1.0) < 1e-12
    assert abs(app.caudal_desde_si(1e-3, "l/min/m") - 60.0) < 1e-9

def test_F_espesores_a_cotas():
    import app
    datos = app._datos_por_defecto()
    datos["capas"] = [dict(nombre="Limo", espesor=3.0, kx=1e-5, kz=1e-5),
                      dict(nombre="Arena", espesor=4.0, kx=5e-5, kz=5e-5),
                      dict(nombre="Arcilla", espesor=3.0, kx=1e-6, kz=1e-6)]
    c = app.construir_contorno(datos)
    assert [cap.z_muro for cap in c.capas] == [0.0, 3.0, 7.0]
    assert c.capas[0].nombre == "Arcilla"   # la más profunda

def test_F_huella_detecta_cambios():
    import app
    d = app._datos_por_defecto()
    d2 = dict(d); d2["h1"] = 13.0
    assert app.huella_entradas(d) != app.huella_entradas(d2)

def test_F_validacion_pie_bajo_impermeable():
    import app
    d = app._datos_por_defecto(); d["z_pie"] = -1.0
    assert any("impermeable" in e for e in app.validar(d))

def test_F_calculo_end_to_end():
    import app
    res = app.calcular(app._datos_por_defecto(), "normal")
    assert res["Q"] > 0 and res["sif"].FS > 0 and not res["err_malla"]


if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, fn in tests:
        try:
            fn(); print(f"PASS {nombre}")
        except Exception as e:
            print(f"FAIL {nombre}: {e}"); fallos += 1
    print(f"\n{len(tests)-fallos}/{len(tests)} tests OK")
    sys.exit(1 if fallos else 0)
