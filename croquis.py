"""
croquis.py — Croquis esquemático de la sección, para la pestaña de geometría.

Dibuja el dominio, las capas con relleno, la tablestaca, los lechos (con su
escalón) y los niveles de agua a partir del diccionario de entrada. Robusto a
datos incompletos o aún inconsistentes (dibuja lo que puede).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np

# Paleta de rellenos por tipo de material (cíclica) y tramas geotécnicas
_COLORES = ["#E8D9A0", "#C9B97E", "#D6C28A", "#BFA86A", "#E0CFA0", "#CDBA82"]
_AGUA = "#AED6F1"


def fig_croquis(datos: dict):
    """Devuelve una figura matplotlib con el croquis de la sección."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    try:
        _dibujar(ax, datos)
    except Exception as ex:
        ax.text(0.5, 0.5, f"Croquis no disponible:\n{ex}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    return fig


def _dibujar(ax, d):
    x_izq = float(d["x_izq"]); x_der = float(d["x_der"])
    x_tab = float(d["x_tablestaca"])
    cota_arr = float(d["cota_arranque"])
    z_la = float(d["z_lecho_arriba"]); z_lb = float(d["z_lecho_abajo"])
    z_cor = float(d["z_coronacion"]); z_pie = float(d["z_pie"])
    h1 = float(d["h1"]); h2 = float(d["h2"])
    arriba_izq = bool(d["lado_arriba_izq"])
    capas = d.get("capas", [])

    # cotas derivadas
    esp_total = sum(float(c["espesor"]) for c in capas) if capas else (cota_arr - z_pie)
    z_imp = cota_arr - esp_total
    z_sup = cota_arr

    # --- capas con relleno (de arriba hacia abajo) ---
    z_techo = cota_arr
    for i, cap in enumerate(capas):
        esp = float(cap["espesor"])
        z_muro = z_techo - esp
        color = _COLORES[i % len(_COLORES)]
        ax.add_patch(Rectangle((x_izq, z_muro), x_der - x_izq, esp,
                               facecolor=color, edgecolor="#7A6A3A",
                               linewidth=0.8, zorder=1))
        # nombre de la capa a la izquierda
        ax.text(x_izq + 0.02 * (x_der - x_izq), 0.5 * (z_muro + z_techo),
                cap.get("nombre", f"Capa {i+1}"), fontsize=8, va="center",
                zorder=5, color="#3A2F15")
        z_techo = z_muro

    # --- sustrato impermeable (trama) ---
    ax.add_patch(Rectangle((x_izq, z_imp - 0.04*(z_sup-z_imp)), x_der-x_izq,
                           0.04*(z_sup-z_imp), facecolor="#888888",
                           hatch="xxx", edgecolor="black", zorder=2))
    ax.text(x_der - 0.02*(x_der-x_izq), z_imp - 0.06*(z_sup-z_imp),
            "Impermeable", ha="right", va="top", fontsize=7, style="italic")

    # --- agua aguas arriba y abajo ---
    xa = (x_izq, x_tab) if arriba_izq else (x_tab, x_der)
    xb = (x_tab, x_der) if arriba_izq else (x_izq, x_tab)
    # lámina de agua arriba (de lecho a h1)
    if h1 > z_la:
        ax.add_patch(Rectangle((xa[0], z_la), xa[1]-xa[0], h1-z_la,
                               facecolor=_AGUA, alpha=0.6, zorder=0))
        _nivel_agua(ax, xa, h1, "#1B4F72")
    # lámina de agua abajo (de lecho a h2)
    if h2 > z_lb:
        ax.add_patch(Rectangle((xb[0], z_lb), xb[1]-xb[0], h2-z_lb,
                               facecolor=_AGUA, alpha=0.6, zorder=0))
        _nivel_agua(ax, xb, h2, "#1B4F72")

    # --- lechos (líneas de terreno) ---
    ax.plot([xa[0], xa[1]], [z_la, z_la], color="saddlebrown", lw=2, zorder=3)
    ax.plot([xb[0], xb[1]], [z_lb, z_lb], color="saddlebrown", lw=2, zorder=3)

    # --- tablestaca ---
    ax.plot([x_tab, x_tab], [z_pie, z_cor], color="black", lw=4, zorder=6,
            solid_capstyle="butt")
    ax.text(x_tab, z_cor + 0.03*(z_sup-z_imp), "Tablestaca", ha="center",
            va="bottom", fontsize=8, fontweight="bold")

    # --- acotaciones clave ---
    # empotramiento s (lado aguas abajo)
    s = z_lb - z_pie
    ax.annotate("", xy=(x_tab + 0.06*(x_der-x_izq), z_pie),
                xytext=(x_tab + 0.06*(x_der-x_izq), z_lb),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(x_tab + 0.08*(x_der-x_izq), 0.5*(z_pie+z_lb),
            f"s={s:.1f}", fontsize=7, va="center", color="#555")
    # ΔH
    ax.annotate("", xy=(x_izq + 0.5*(x_tab-x_izq), h1),
                xytext=(x_izq + 0.5*(x_tab-x_izq), h2),
                arrowprops=dict(arrowstyle="<->", color="#1B4F72"))
    ax.text(x_izq + 0.5*(x_tab-x_izq), 0.5*(h1+h2),
            f"ΔH={h1-h2:.2f}", fontsize=7, va="center", ha="right",
            color="#1B4F72")

    # límites
    margen_z = 0.12 * (max(h1, h2, z_cor) - z_imp)
    ax.set_xlim(x_izq - 0.03*(x_der-x_izq), x_der + 0.03*(x_der-x_izq))
    ax.set_ylim(z_imp - margen_z, max(h1, h2, z_cor) + margen_z)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m), cotas absolutas")
    ax.set_title("Croquis de la sección")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def _nivel_agua(ax, xr, nivel, color):
    """Dibuja el símbolo de nivel de agua (triángulo) en el centro del tramo."""
    xc = 0.5 * (xr[0] + xr[1])
    ancho = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.get_xlim()[1] != ax.get_xlim()[0] else 0.4
    ax.plot([xr[0], xr[1]], [nivel, nivel], color=color, lw=1, zorder=4)
    ax.plot(xc, nivel, marker="v", color=color, markersize=7, zorder=5)
