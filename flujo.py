"""
flujo.py — Motor de cálculo de flujo 2D estacionario por volúmenes finitos.

Paso B del plan. Resuelve  div(c * grad(phi)) = 0  sobre la malla estructurada
de malla.py, con:
  - c = k  -> phi = h  (carga hidráulica)
  - c = 1/k -> phi = psi (función de corriente)  [se usará en Fase 4]

Esquema: volúmenes finitos vertex-centered. Cada CELDA aporta conductancias
a las 4 aristas de su contorno. Esto produce automáticamente:
  - media ARMÓNICA (serie) en flujo vertical a través de capas
    (vía el balance nodal en los nodos de interfaz), y
  - media ARITMÉTICA (paralelo) en flujo horizontal a lo largo de capas
    (vía la suma de aportaciones de las celdas a ambos lados de la arista).

Tablestaca de espesor nulo: los nodos de la pantalla en el tramo enterrado
(z_pie < z <= lecho) se DESDOBLAN en dos grados de libertad (lado 'L' y 'R'),
desconectados horizontalmente entre sí. En el pie (z = z_pie) y por debajo el
nodo es único: ahí el flujo rodea la punta.

Conductancia de arista
----------------------
Celda (ic,jc) de dimensiones dx x dz con coeficientes (cx, cz):
  - aporta a sus 2 aristas HORIZONTALES (conexiones en x): cx * (dz/2) / dx
  - aporta a sus 2 aristas VERTICALES   (conexiones en z): cz * (dx/2) / dz
La conductancia total de una conexión es la suma de aportaciones de las celdas
que la comparten.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from malla import Malla, TipoNodo


# --------------------------------------------------------------------------- #
#  Mapa de grados de libertad (DOF), con desdoblamiento de la tablestaca       #
# --------------------------------------------------------------------------- #
class DOFMap:
    """
    Asigna un índice de incógnita (dof) a cada nodo activo. Los nodos de la
    pantalla en el tramo enterrado tienen DOS dofs: lado 'L' (izquierda) y 'R'.

    side: None para nodos normales; 'L'/'R' para nodos desdoblados.
    """
    def __init__(self, m: Malla):
        self.m = m
        c = m.contorno
        self.i_tab = int(np.argmin(np.abs(m.x - c.x_tablestaca)))
        self.z_pie = c.z_pie
        self.z_lecho_max = max(c.z_lecho_arriba, c.z_lecho_abajo)

        # Lechos de cada lado de la pantalla. El lado L (x < x_tab) y R (x > x_tab)
        # tienen suelo cada uno hasta SU lecho. Con escalón de lechos, en el
        # tramo entre el lecho bajo y el alto solo un lado tiene suelo.
        if c.lado_arriba_izq:
            self.z_lecho_L = c.z_lecho_arriba   # izquierda = aguas arriba
            self.z_lecho_R = c.z_lecho_abajo
        else:
            self.z_lecho_L = c.z_lecho_abajo
            self.z_lecho_R = c.z_lecho_arriba

        # Clasificación de cada nodo de la columna de la pantalla en el tramo
        # enterrado (z_pie < z <= lecho de su lado):
        #   - desdoblado: ambos lados tienen suelo (z <= ambos lechos)
        #   - solo L / solo R: únicamente ese lado tiene suelo (zona de escalón)
        self._split = np.zeros((m.nx, m.nz), dtype=bool)
        self._solo_L = np.zeros((m.nx, m.nz), dtype=bool)
        self._solo_R = np.zeros((m.nx, m.nz), dtype=bool)
        for j in range(m.nz):
            zj = m.z[j]
            if zj <= self.z_pie + 1e-9:
                continue
            tiene_L = zj <= self.z_lecho_L + 1e-9
            tiene_R = zj <= self.z_lecho_R + 1e-9
            if tiene_L and tiene_R:
                self._split[self.i_tab, j] = True
            elif tiene_L:
                self._solo_L[self.i_tab, j] = True
            elif tiene_R:
                self._solo_R[self.i_tab, j] = True

        # Numeración
        self.dof_single = -np.ones((m.nx, m.nz), dtype=int)   # nodos normales
        self.dof_L = -np.ones((m.nx, m.nz), dtype=int)
        self.dof_R = -np.ones((m.nx, m.nz), dtype=int)
        n = 0
        for i in range(m.nx):
            for j in range(m.nz):
                if m.tipo_nodo[i, j] == TipoNodo.FUERA:
                    continue
                if self._split[i, j]:
                    self.dof_L[i, j] = n; n += 1
                    self.dof_R[i, j] = n; n += 1
                elif self._solo_L[i, j]:
                    self.dof_L[i, j] = n; n += 1   # solo lado L
                elif self._solo_R[i, j]:
                    self.dof_R[i, j] = n; n += 1   # solo lado R
                else:
                    self.dof_single[i, j] = n; n += 1
        self.ndof = n

    def is_split(self, i: int, j: int) -> bool:
        return bool(self._split[i, j])

    def is_solo_lado(self, i: int, j: int) -> str | None:
        """Devuelve 'L'/'R' si el nodo existe solo en ese lado (zona escalón)."""
        if self._solo_L[i, j]:
            return 'L'
        if self._solo_R[i, j]:
            return 'R'
        return None

    def dof(self, i: int, j: int, side: str | None = None) -> int:
        """dof del nodo (i,j). side 'L'/'R' relevante si está desdoblado o
        existe solo en un lado (zona de escalón)."""
        if self._split[i, j]:
            if side == 'L':
                return self.dof_L[i, j]
            elif side == 'R':
                return self.dof_R[i, j]
            else:
                raise ValueError(f"Nodo desdoblado ({i},{j}) requiere side.")
        if self._solo_L[i, j]:
            return self.dof_L[i, j]
        if self._solo_R[i, j]:
            return self.dof_R[i, j]
        return self.dof_single[i, j]


# --------------------------------------------------------------------------- #
#  Ensamblado genérico  div(c grad phi) = 0                                    #
# --------------------------------------------------------------------------- #
def _coef_celda(m: Malla, ic: int, jc: int, campo: str) -> tuple[float, float]:
    """Devuelve (cx, cz) de la celda (ic,jc) para el campo pedido.
    campo='h'  -> (kx, kz);  campo='psi' -> (1/kx, 1/kz). Celda fuera -> (0,0)."""
    k_idx = m.capa_celda[ic, jc]
    if k_idx < 0:
        return 0.0, 0.0
    cap = m.contorno.capas[k_idx]
    if campo == 'h':
        return cap.kx, cap.kz
    elif campo == 'psi':
        return 1.0 / cap.kx, 1.0 / cap.kz
    raise ValueError("campo debe ser 'h' o 'psi'.")


def _lado_de_celda(dm: DOFMap, ic: int) -> str:
    """Para una celda y la columna de la tablestaca: ¿la celda está a la
    izquierda ('L') o derecha ('R') de la pantalla? La celda (ic,*) ocupa
    [x[ic], x[ic+1]]. Si su borde derecho es la tablestaca (ic+1==i_tab) -> L.
    Si su borde izquierdo es la tablestaca (ic==i_tab) -> R."""
    if ic + 1 == dm.i_tab:
        return 'L'
    if ic == dm.i_tab:
        return 'R'
    return None  # no toca la tablestaca


def _resolver_dof(dm: DOFMap, i: int, j: int, lado_celda: str | None) -> int:
    """dof de un nodo (i,j) referenciado desde una celda cuyo lado es lado_celda."""
    if dm.is_split(i, j):
        # el nodo desdoblado toma el lado de la celda que lo referencia
        return dm.dof(i, j, side=lado_celda)
    return dm.dof(i, j)


def ensamblar(m: Malla, dm: DOFMap, campo: str = 'h'):
    """
    Ensambla la matriz A (ndof x ndof) del operador div(c grad phi).
    Devuelve A en formato lil para luego aplicar BC. Sin término fuente.
    """
    n = dm.ndof
    A = sp.lil_matrix((n, n))

    def add_conexion(da: int, db: int, C: float):
        """Añade conductancia C entre dofs da y db (laplaciano de conductancia)."""
        if da < 0 or db < 0 or C == 0.0:
            return
        A[da, da] -= C
        A[da, db] += C
        A[db, db] -= C
        A[db, da] += C

    for ic in range(m.nx - 1):
        dx = m.x[ic + 1] - m.x[ic]
        lado = _lado_de_celda(dm, ic)
        for jc in range(m.nz - 1):
            cx, cz = _coef_celda(m, ic, jc, campo)
            if cx == 0.0 and cz == 0.0:
                continue
            dz = m.z[jc + 1] - m.z[jc]

            # Conductancias de arista que aporta esta celda
            Ch = cx * (dz / 2.0) / dx   # aristas horizontales (conexión en x)
            Cv = cz * (dx / 2.0) / dz   # aristas verticales   (conexión en z)

            # Nodos esquina de la celda
            # bottom edge: (ic,jc)-(ic+1,jc) ; top: (ic,jc+1)-(ic+1,jc+1)
            # left edge:   (ic,jc)-(ic,jc+1) ; right: (ic+1,jc)-(ic+1,jc+1)
            def D(i, j):
                return _resolver_dof(dm, i, j, lado)

            # --- aristas horizontales (no cruzan la tablestaca: cada celda está
            #     enteramente a un lado) ---
            add_conexion(D(ic, jc),     D(ic + 1, jc),     Ch)  # bottom
            add_conexion(D(ic, jc + 1), D(ic + 1, jc + 1), Ch)  # top

            # --- aristas verticales ---
            add_conexion(D(ic, jc),     D(ic, jc + 1),     Cv)  # left
            add_conexion(D(ic + 1, jc), D(ic + 1, jc + 1), Cv)  # right

    return A


# --------------------------------------------------------------------------- #
#  Aplicación de Dirichlet y resolución                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Solucion:
    """Resultado del cálculo de carga h."""
    m: Malla
    dm: DOFMap
    h_nodo: np.ndarray        # (nx, nz) carga en cada nodo (NaN si FUERA)
    h_nodo_L: np.ndarray      # (nx, nz) lado L de nodos desdoblados (NaN si no)
    h_nodo_R: np.ndarray
    A_sin_bc: sp.spmatrix     # matriz antes de BC (para flujos de reacción)
    dof_dirichlet: np.ndarray # dofs fijados
    valores_dirichlet: np.ndarray


def _resolver_robusto(A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """
    Resuelve A·x = b de forma robusta frente a mal condicionamiento (p. ej.
    capas con contraste de permeabilidad extremo, que dejan regiones casi
    desconectadas y vuelven la matriz casi singular).

    Estrategia:
      1) Intento directo con spsolve.
      2) Si falla (NaN/inf o singular), regularización mínima: se añade un
         término diminuto a la diagonal (epsilon · máximo de la diagonal) que
         estabiliza el sistema sin alterar apreciablemente el resultado físico
         (en regiones que casi no conducen, la carga es casi indeterminada, y
         la regularización la fija a un valor suave y consistente).
      3) Si aún falla, solver iterativo (LGMRES) con precondicionador.
    """
    import warnings
    n = A.shape[0]

    # 1) intento directo
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            x = spsolve(A, b)
            if np.all(np.isfinite(x)):
                return x
        except Exception:
            pass

    # 2) regularización mínima de la diagonal
    diag = np.abs(A.diagonal())
    escala = diag[diag > 0].max() if np.any(diag > 0) else 1.0
    for eps in (1e-12, 1e-10, 1e-8, 1e-6):
        Areg = A + sp.eye(n, format="csr") * (eps * escala)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x = spsolve(Areg, b)
            if np.all(np.isfinite(x)):
                return x
        except Exception:
            continue

    # 3) iterativo como último recurso
    from scipy.sparse.linalg import lgmres
    Areg = A + sp.eye(n, format="csr") * (1e-8 * escala)
    x, info = lgmres(Areg, b, rtol=1e-10, maxiter=2000)
    return x


def contraste_permeabilidad(m: Malla) -> dict:
    """
    Evalúa el contraste de permeabilidad entre capas. Devuelve dict con el
    ratio máximo y un diagnóstico, para avisar de posibles problemas numéricos
    o de capas que conviene tratar como sustrato impermeable.
    """
    ks = []
    for cap in m.contorno.capas:
        ks.append(np.sqrt(cap.kx * cap.kz))
    ks = np.array(ks)
    k_max, k_min = ks.max(), ks.min()
    ratio = k_max / k_min if k_min > 0 else float('inf')

    # capa(s) muy poco permeables frente al resto (candidatas a impermeable)
    capas_debiles = []
    for i, cap in enumerate(m.contorno.capas):
        k = np.sqrt(cap.kx * cap.kz)
        if k_max / k >= 1e4:   # 4 órdenes de magnitud por debajo del máximo
            capas_debiles.append((i, cap.nombre, k))

    if ratio < 1e3:
        nivel = "ok"
    elif ratio < 1e4:
        nivel = "alto"          # resoluble con regularización, sin problema
    else:
        nivel = "extremo"       # conviene avisar / tratar como impermeable

    return dict(ratio=ratio, nivel=nivel, capas_debiles=capas_debiles,
                k_max=k_max, k_min=k_min)


def resolver_h(m: Malla) -> Solucion:
    """Resuelve el campo de carga h con las condiciones de contorno de la malla."""
    dm = DOFMap(m)
    A = ensamblar(m, dm, campo='h')
    n = dm.ndof
    b = np.zeros(n)

    # Recoger dofs Dirichlet con sus valores.
    # Nodos desdoblados en el lecho (coronación de la pantalla): cada lado
    # toma el valor del lecho de SU lado (L = lado de x menor, R = x mayor),
    # que puede ser h1 o h2 según en qué lado esté aguas arriba.
    c = m.contorno
    lado_L_es_arriba = c.lado_arriba_izq      # L (x<x_tab) coincide con aguas arriba?
    h_lado_L = c.h1 if lado_L_es_arriba else c.h2
    h_lado_R = c.h2 if lado_L_es_arriba else c.h1

    dir_dofs = []
    dir_vals = []
    for i in range(m.nx):
        for j in range(m.nz):
            t = m.tipo_nodo[i, j]
            if t in (TipoNodo.DIRICHLET_H1, TipoNodo.DIRICHLET_H2):
                if dm.is_split(i, j):
                    # nodo desdoblado en el lecho: valor distinto por lado
                    dL = dm.dof(i, j, side='L')
                    dR = dm.dof(i, j, side='R')
                    dir_dofs.append(dL); dir_vals.append(h_lado_L)
                    dir_dofs.append(dR); dir_vals.append(h_lado_R)
                else:
                    val = m.h_dirichlet[i, j]
                    d = dm.dof(i, j)
                    dir_dofs.append(d); dir_vals.append(val)
    dir_dofs = np.array(dir_dofs, dtype=int)
    dir_vals = np.array(dir_vals, dtype=float)

    A_sin_bc = A.tocsr().copy()

    # Imponer Dirichlet por eliminación de fila/columna (mantiene simetría)
    A = A.tolil()
    for d, v in zip(dir_dofs, dir_vals):
        # pasar columna al RHS
        col = A[:, d].toarray().ravel()
        b -= col * v
        A[d, :] = 0.0
        A[:, d] = 0.0
        A[d, d] = 1.0
        b[d] = v
    A = A.tocsr()

    phi = _resolver_robusto(A, b)

    # Reconstruir campos por nodo
    h_nodo = np.full((m.nx, m.nz), np.nan)
    h_L = np.full((m.nx, m.nz), np.nan)
    h_R = np.full((m.nx, m.nz), np.nan)
    for i in range(m.nx):
        for j in range(m.nz):
            if m.tipo_nodo[i, j] == TipoNodo.FUERA:
                continue
            if dm.is_split(i, j):
                h_L[i, j] = phi[dm.dof(i, j, 'L')]
                h_R[i, j] = phi[dm.dof(i, j, 'R')]
                h_nodo[i, j] = 0.5 * (h_L[i, j] + h_R[i, j])  # valor "medio"
            else:
                h_nodo[i, j] = phi[dm.dof(i, j)]

    return Solucion(m=m, dm=dm, h_nodo=h_nodo, h_nodo_L=h_L, h_nodo_R=h_R,
                    A_sin_bc=A_sin_bc, dof_dirichlet=dir_dofs,
                    valores_dirichlet=dir_vals)


# --------------------------------------------------------------------------- #
#  Caudal por balance de flujos nodales (reacciones en Dirichlet)              #
# --------------------------------------------------------------------------- #
def caudal_entrada(sol: Solucion):
    """
    Caudal total que entra por el contorno aguas arriba (Dirichlet h1),
    por balance de flujos nodales: q_dof = (A_sin_bc @ phi)[dof].
    Para nodos Dirichlet, esa cantidad es el flujo neto que sale del nodo
    hacia el dominio (la "reacción"). Sumado sobre h1 da el caudal total.

    Devuelve (Q, Qh1, Qh2): Q es |caudal| promedio por metro lineal
    [m^3/s/m]; Qh1, Qh2 los flujos en cada contorno (deben ser opuestos).
    """
    m, dm = sol.m, sol.dm
    # reconstruir vector phi completo
    phi = np.zeros(dm.ndof)
    for i in range(m.nx):
        for j in range(m.nz):
            if m.tipo_nodo[i, j] == TipoNodo.FUERA:
                continue
            if dm.is_split(i, j):
                phi[dm.dof(i, j, 'L')] = sol.h_nodo_L[i, j]
                phi[dm.dof(i, j, 'R')] = sol.h_nodo_R[i, j]
            else:
                phi[dm.dof(i, j)] = sol.h_nodo[i, j]

    q = sol.A_sin_bc @ phi   # flujo neto en cada dof

    Q_h1 = 0.0
    Q_h2 = 0.0
    for i in range(m.nx):
        for j in range(m.nz):
            t = m.tipo_nodo[i, j]
            if t == TipoNodo.DIRICHLET_H1:
                if dm.is_split(i, j):
                    Q_h1 += q[dm.dof(i, j, 'L')] + q[dm.dof(i, j, 'R')]
                else:
                    Q_h1 += q[dm.dof(i, j)]
            elif t == TipoNodo.DIRICHLET_H2:
                if dm.is_split(i, j):
                    Q_h2 += q[dm.dof(i, j, 'L')] + q[dm.dof(i, j, 'R')]
                else:
                    Q_h2 += q[dm.dof(i, j)]

    # Q_h1 debería ser ~ -Q_h2 (lo que entra sale). Devolvemos el promedio.
    return 0.5 * (abs(Q_h1) + abs(Q_h2)), Q_h1, Q_h2
