"""
malla.py — Generador de malla estructurada uniforme para el modelo de flujo 2D
por diferencias finitas / volúmenes finitos.

Paso A del plan de desarrollo. Esta pieza es el "contrato de datos" entre la
entrada (geometría rectangular + estratigrafía) y el motor de ensamblado.

Convenciones
------------
- Sistema de cotas ABSOLUTAS, z positivo hacia ARRIBA, datum único.
- El dominio exterior es un rectángulo: [x_izq, x_der] x [z_imp, z_sup].
- La malla es estructurada: producto cartesiano de dos vectores de líneas
  x[] (verticales) y z[] (horizontales). Los NODOS están en las
  intersecciones; las CELDAS son los rectángulos entre líneas.
- Toda discontinuidad geométrica (contacto de capa, lecho, pie de tablestaca,
  impermeable) DEBE caer sobre una línea de malla. Lo garantiza el sembrado
  de "líneas obligadas" y lo verifica el test geométrico.

Discretización: nodos en las líneas (vertex-centered). nx = len(x), nz = len(z).
El nodo (i, j) tiene coordenada (x[i], z[j]), i en [0, nx), j en [0, nz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- #
#  Tipos de nodo (etiquetado de contorno)                                      #
# --------------------------------------------------------------------------- #
class TipoNodo:
    """Etiquetas de nodo. Enteros para poder almacenarlos en un array."""
    INTERIOR = 0        # nodo activo, incógnita
    DIRICHLET_H1 = 1    # carga impuesta aguas arriba (lecho izquierdo)
    DIRICHLET_H2 = 2    # carga impuesta aguas abajo (lecho derecho)
    NEUMANN = 3         # contorno impermeable de flujo nulo (bordes, impermeable)
    FUERA = 4           # nodo fuera del dominio de suelo (zona de agua sobre lecho)


# --------------------------------------------------------------------------- #
#  Definición del contorno                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Capa:
    """Una capa de suelo. Cotas absolutas de muro (inferior) y techo (superior)."""
    z_muro: float
    z_techo: float
    kx: float
    kz: float
    nombre: str = ""

    def __post_init__(self):
        if self.z_techo <= self.z_muro:
            raise ValueError(
                f"Capa '{self.nombre}': z_techo ({self.z_techo}) debe ser "
                f"> z_muro ({self.z_muro})."
            )
        if self.kx <= 0 or self.kz <= 0:
            raise ValueError(
                f"Capa '{self.nombre}': permeabilidades deben ser > 0 "
                f"(kx={self.kx}, kz={self.kz})."
            )


@dataclass
class ContornoRect:
    """
    Definición geométrica completa del problema (contorno rectangular +
    tablestaca interna + escalón de lechos + estratigrafía).

    Toda la geometría cabe aquí. Las capas se dan ya en cotas absolutas
    (la conversión espesores->cotas se hace aguas arriba, en la capa de entrada).
    """
    # Rectángulo exterior
    x_izq: float
    x_der: float
    # z_imp (fondo) y z_sup (techo) se derivan de las capas, pero se guardan.

    # Tablestaca interna
    x_tablestaca: float
    z_coronacion: float
    z_pie: float

    # Escalón de lechos
    z_lecho_arriba: float   # lado aguas arriba (lecho más alto, carga h1)
    z_lecho_abajo: float    # lado aguas abajo (lecho más bajo, carga h2)
    lado_arriba_izq: bool = True   # True: aguas arriba a la IZQUIERDA de la tablestaca

    # Cargas hidráulicas (cotas absolutas de lámina de agua)
    h1: float = 0.0   # aguas arriba
    h2: float = 0.0   # aguas abajo

    # Estratigrafía (de muro a techo, sin huecos ni solapes)
    capas: list[Capa] = field(default_factory=list)

    # Derivados (rellenados en __post_init__)
    z_imp: float = field(init=False)
    z_sup: float = field(init=False)

    def __post_init__(self):
        if not self.capas:
            raise ValueError("Debe haber al menos una capa.")
        # Ordenar capas de muro a techo
        self.capas = sorted(self.capas, key=lambda c: c.z_muro)
        # Verificar que encadenan sin huecos ni solapes
        for a, b in zip(self.capas[:-1], self.capas[1:]):
            if not np.isclose(a.z_techo, b.z_muro):
                raise ValueError(
                    f"Capas no contiguas: techo {a.z_techo} != muro {b.z_muro}."
                )
        self.z_imp = self.capas[0].z_muro
        self.z_sup = self.capas[-1].z_techo

        # Validaciones de coherencia vertical
        if self.x_der <= self.x_izq:
            raise ValueError("x_der debe ser > x_izq.")
        if not (self.x_izq < self.x_tablestaca < self.x_der):
            raise ValueError("La tablestaca debe estar dentro del dominio.")
        if not (self.z_imp < self.z_pie):
            raise ValueError(
                f"El pie de la tablestaca ({self.z_pie}) debe estar por encima "
                f"del impermeable ({self.z_imp}). Si llega al impermeable, "
                f"no hay flujo bajo el pie (caso degenerado)."
            )
        z_lecho_max = max(self.z_lecho_arriba, self.z_lecho_abajo)
        if self.z_coronacion <= z_lecho_max:
            raise ValueError(
                "La coronación de la tablestaca debe superar el lecho más alto."
            )
        if self.z_pie >= min(self.z_lecho_arriba, self.z_lecho_abajo):
            raise ValueError(
                "El pie de la tablestaca debe estar por debajo de ambos lechos."
            )

    @property
    def x_arriba(self) -> tuple[float, float]:
        """Rango x del lado aguas arriba."""
        if self.lado_arriba_izq:
            return (self.x_izq, self.x_tablestaca)
        return (self.x_tablestaca, self.x_der)

    @property
    def x_abajo(self) -> tuple[float, float]:
        """Rango x del lado aguas abajo."""
        if self.lado_arriba_izq:
            return (self.x_tablestaca, self.x_der)
        return (self.x_izq, self.x_tablestaca)


# --------------------------------------------------------------------------- #
#  Sembrado de líneas: obligadas + relleno uniforme                            #
# --------------------------------------------------------------------------- #
def _sembrar_lineas(obligadas: list[float], paso_objetivo: float,
                    lo: float, hi: float) -> np.ndarray:
    """
    Construye un vector de líneas que:
      - contiene EXACTAMENTE todas las coordenadas obligadas,
      - rellena cada tramo entre obligadas consecutivas con paso ~ uniforme
        (paso_objetivo), de modo que las obligadas quedan como subconjunto.

    El paso real en cada tramo se ajusta para que un número entero de
    subdivisiones encaje exactamente entre las dos obligadas (uniforme por
    tramos, pero con las obligadas respetadas de forma exacta).
    """
    # Limpiar, recortar al rango y deduplicar las obligadas
    obl = sorted(set(round(v, 9) for v in obligadas if lo - 1e-9 <= v <= hi + 1e-9))
    obl = [lo] + [v for v in obl if lo < v < hi] + [hi]
    obl = sorted(set(round(v, 9) for v in obl))

    lineas = [obl[0]]
    for a, b in zip(obl[:-1], obl[1:]):
        tramo = b - a
        n_sub = max(1, int(round(tramo / paso_objetivo)))
        # nodos intermedios (excluye 'a', incluye 'b')
        for k in range(1, n_sub + 1):
            lineas.append(a + tramo * k / n_sub)
    return np.array(sorted(set(round(v, 9) for v in lineas)))


# --------------------------------------------------------------------------- #
#  Estructura de malla (el contrato de datos)                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Malla:
    """
    Malla estructurada generada. Es lo que consume el motor de ensamblado.

    Arrays principales
    ------------------
    x, z          : coordenadas de las líneas (1D). nx=len(x), nz=len(z).
    tipo_nodo     : (nx, nz) int, etiqueta TipoNodo de cada nodo.
    capa_celda    : (nx-1, nz-1) int, índice de capa de cada celda (-1 si fuera).
    h_dirichlet   : (nx, nz) float, valor de carga impuesta en nodos Dirichlet
                    (NaN donde no aplica).
    """
    contorno: ContornoRect
    x: np.ndarray
    z: np.ndarray
    tipo_nodo: np.ndarray
    capa_celda: np.ndarray
    h_dirichlet: np.ndarray

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def nz(self) -> int:
        return len(self.z)

    def idx(self, i: int, j: int) -> int:
        """Índice lineal del nodo (i, j) en numeración fila-mayor sobre z."""
        return i * self.nz + j

    # --- consultas útiles para el motor ---
    def cara_cortada_tablestaca(self, i: int, j: int) -> bool:
        """
        ¿La cara HORIZONTAL entre el nodo (i,j) y (i+1,j) está cortada por la
        tablestaca? (es decir, ¿esa cara vertical de celda coincide con la
        pantalla, en el tramo coronación..pie?). El flujo horizontal a través
        de ella debe anularse.
        """
        c = self.contorno
        x_cara = 0.5 * (self.x[i] + self.x[i + 1])
        # La tablestaca está sobre una línea de malla; la cara cortada es la que
        # tiene la línea de la tablestaca como uno de sus extremos... en realidad
        # el corte se modela entre las dos columnas adyacentes a la tablestaca.
        # Aquí marcamos: la cara horizontal cuyo punto medio cruza x_tablestaca
        # y cuya z está en [z_pie, z_coronacion].
        cruza_x = self.x[i] < c.x_tablestaca < self.x[i + 1]
        # El tramo enterrado de pantalla va del pie al lecho más bajo: por
        # encima del lecho ya no hay suelo a ambos lados. Usamos el lecho
        # correspondiente como tope superior del corte.
        z_tope = max(c.z_lecho_arriba, c.z_lecho_abajo)
        en_tramo = c.z_pie <= self.z[j] <= z_tope
        return cruza_x and en_tramo


# --------------------------------------------------------------------------- #
#  Generador principal                                                         #
# --------------------------------------------------------------------------- #
_DENSIDAD = {"grosero": 0.20, "normal": 0.08, "fino": 0.03}  # fracción de la
# menor dimensión característica usada como paso objetivo (ajustable)


def generar_malla(c: ContornoRect, densidad: str = "normal",
                  paso: float | None = None) -> Malla:
    """
    Genera la malla uniforme (por tramos) a partir del contorno.

    densidad : 'grosero' | 'normal' | 'fino' — define el paso objetivo como
               fracción de la menor dimensión del dominio.
    paso     : si se da, sobreescribe la densidad y fija el paso objetivo (m).
    """
    Lx = c.x_der - c.x_izq
    Lz = c.z_sup - c.z_imp
    if paso is None:
        frac = _DENSIDAD.get(densidad)
        if frac is None:
            raise ValueError(f"densidad debe ser uno de {list(_DENSIDAD)}.")
        paso = frac * min(Lx, Lz)

    # --- Líneas obligadas ---
    # NOTA: z_coronacion NO se siembra: la parte de tablestaca por encima del
    # lecho sobresale en el agua, fuera del medio poroso, y no se malla.
    # El tramo relevante de pantalla a efectos de flujo es [z_pie, z_lecho].
    z_obl = [c.z_imp, c.z_sup, c.z_pie,
             c.z_lecho_arriba, c.z_lecho_abajo]
    z_obl += [cap.z_muro for cap in c.capas] + [cap.z_techo for cap in c.capas]

    x_obl = [c.x_izq, c.x_der, c.x_tablestaca]

    x = _sembrar_lineas(x_obl, paso, c.x_izq, c.x_der)
    z = _sembrar_lineas(z_obl, paso, c.z_imp, c.z_sup)
    nx, nz = len(x), len(z)

    # --- Etiquetado de celdas: a qué capa pertenece cada una ---
    capa_celda = np.full((nx - 1, nz - 1), -1, dtype=int)
    for jc in range(nz - 1):
        z_centro = 0.5 * (z[jc] + z[jc + 1])
        # capa cuyo [z_muro, z_techo] contiene z_centro
        idx_capa = -1
        for k, cap in enumerate(c.capas):
            if cap.z_muro <= z_centro <= cap.z_techo:
                idx_capa = k
                break
        capa_celda[:, jc] = idx_capa

    # Marcar como "fuera" (zona de agua) las celdas por encima del lecho de cada lado
    xa_lo, xa_hi = c.x_arriba
    xb_lo, xb_hi = c.x_abajo
    for ic in range(nx - 1):
        x_centro = 0.5 * (x[ic] + x[ic + 1])
        for jc in range(nz - 1):
            z_centro = 0.5 * (z[jc] + z[jc + 1])
            en_arriba = xa_lo <= x_centro <= xa_hi
            en_abajo = xb_lo <= x_centro <= xb_hi
            lecho = c.z_lecho_arriba if en_arriba else (
                c.z_lecho_abajo if en_abajo else None)
            if lecho is not None and z_centro > lecho:
                capa_celda[ic, jc] = -1  # fuera del suelo

    # --- Etiquetado de nodos ---
    tipo = np.full((nx, nz), TipoNodo.INTERIOR, dtype=int)
    h_dir = np.full((nx, nz), np.nan)

    for i in range(nx):
        for j in range(nz):
            xi, zj = x[i], z[j]
            # ¿nodo fuera del suelo? (por encima del lecho de su lado)
            en_arriba = xa_lo <= xi <= xa_hi
            en_abajo = xb_lo <= xi <= xb_hi

            # Lecho aguas arriba -> Dirichlet h1
            if en_arriba and np.isclose(zj, c.z_lecho_arriba):
                tipo[i, j] = TipoNodo.DIRICHLET_H1
                h_dir[i, j] = c.h1
                continue
            # Lecho aguas abajo -> Dirichlet h2
            if en_abajo and np.isclose(zj, c.z_lecho_abajo):
                tipo[i, j] = TipoNodo.DIRICHLET_H2
                h_dir[i, j] = c.h2
                continue
            # Por encima del lecho -> fuera
            lecho = c.z_lecho_arriba if en_arriba else (
                c.z_lecho_abajo if en_abajo else c.z_sup)
            if zj > lecho + 1e-9:
                tipo[i, j] = TipoNodo.FUERA
                continue
            # Bordes impermeables: laterales y fondo
            if (np.isclose(xi, c.x_izq) or np.isclose(xi, c.x_der)
                    or np.isclose(zj, c.z_imp)):
                tipo[i, j] = TipoNodo.NEUMANN
                continue

    return Malla(contorno=c, x=x, z=z, tipo_nodo=tipo,
                 capa_celda=capa_celda, h_dirichlet=h_dir)


# --------------------------------------------------------------------------- #
#  TEST GEOMÉTRICO — el chequeo crítico contra bugs silenciosos                #
# --------------------------------------------------------------------------- #
def verificar_malla(m: Malla, tol: float = 1e-7) -> list[str]:
    """
    Verifica que la malla respeta la geometría. Devuelve lista de errores
    (vacía si todo correcto). ESTE es el test que evita resultados sutilmente
    erróneos por discontinuidades mal alineadas.
    """
    errores: list[str] = []
    c = m.contorno

    def en_lineas(valor: float, lineas: np.ndarray, nombre: str):
        if not np.any(np.abs(lineas - valor) < tol):
            errores.append(
                f"La cota singular {nombre}={valor} NO cae sobre ninguna "
                f"línea de malla.")

    # Cotas z obligadas
    en_lineas(c.z_imp, m.z, "z_impermeable")
    en_lineas(c.z_sup, m.z, "z_sup")
    en_lineas(c.z_pie, m.z, "z_pie")
    en_lineas(c.z_lecho_arriba, m.z, "z_lecho_arriba")
    en_lineas(c.z_lecho_abajo, m.z, "z_lecho_abajo")
    for k, cap in enumerate(c.capas):
        en_lineas(cap.z_muro, m.z, f"capa[{k}].z_muro")
        en_lineas(cap.z_techo, m.z, f"capa[{k}].z_techo")

    # Coordenadas x obligadas
    en_lineas(c.x_izq, m.x, "x_izq")
    en_lineas(c.x_der, m.x, "x_der")
    en_lineas(c.x_tablestaca, m.x, "x_tablestaca")

    # Monotonía estricta de las líneas
    if not np.all(np.diff(m.x) > 0):
        errores.append("Las líneas x no son estrictamente crecientes.")
    if not np.all(np.diff(m.z) > 0):
        errores.append("Las líneas z no son estrictamente crecientes.")

    # Toda celda dentro del suelo debe tener una capa asignada
    # (las de fuera tienen -1, eso es correcto). Verificamos que no haya
    # celdas de suelo sin capa por un fallo de cobertura.
    for ic in range(m.nx - 1):
        x_centro = 0.5 * (m.x[ic] + m.x[ic + 1])
        for jc in range(m.nz - 1):
            z_centro = 0.5 * (m.z[jc] + m.z[jc + 1])
            xa_lo, xa_hi = c.x_arriba
            xb_lo, xb_hi = c.x_abajo
            en_arriba = xa_lo <= x_centro <= xa_hi
            en_abajo = xb_lo <= x_centro <= xb_hi
            lecho = c.z_lecho_arriba if en_arriba else c.z_lecho_abajo
            es_suelo = z_centro < lecho and z_centro > c.z_imp
            if es_suelo and m.capa_celda[ic, jc] == -1:
                errores.append(
                    f"Celda de suelo ({ic},{jc}) en z={z_centro:.3f} sin capa "
                    f"asignada (fallo de cobertura estratigráfica).")
                break  # un aviso por columna basta

    return errores


# --------------------------------------------------------------------------- #
#  Resumen legible                                                             #
# --------------------------------------------------------------------------- #
def resumen(m: Malla) -> str:
    c = m.contorno
    n_dir1 = int(np.sum(m.tipo_nodo == TipoNodo.DIRICHLET_H1))
    n_dir2 = int(np.sum(m.tipo_nodo == TipoNodo.DIRICHLET_H2))
    n_neu = int(np.sum(m.tipo_nodo == TipoNodo.NEUMANN))
    n_int = int(np.sum(m.tipo_nodo == TipoNodo.INTERIOR))
    n_fuera = int(np.sum(m.tipo_nodo == TipoNodo.FUERA))
    return (
        f"Malla {m.nx} x {m.nz} = {m.nx * m.nz} nodos\n"
        f"  Dominio: x[{c.x_izq}, {c.x_der}]  z[{c.z_imp}, {c.z_sup}]\n"
        f"  Tablestaca en x={c.x_tablestaca}, pie z={c.z_pie}\n"
        f"  Nodos: {n_int} interior, {n_dir1} Dirichlet-h1, "
        f"{n_dir2} Dirichlet-h2, {n_neu} Neumann, {n_fuera} fuera\n"
        f"  Capas: {len(c.capas)}\n"
        f"  ΔH = h1 - h2 = {c.h1 - c.h2}"
    )
