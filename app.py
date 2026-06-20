"""
app.py — Interfaz Streamlit para el cálculo de flujo 2D bajo tablestaca
por diferencias finitas / volúmenes finitos.

Paso F del plan. Capa fina sobre el motor (malla.py + flujo.py + postproceso.py).

Arquitectura:
  - Entrada por pestañas, espesores convertidos a cotas en la frontera.
  - Permeabilidad: SI interno (m/s); la interfaz acepta cm/s y convierte.
  - Detección de resultados obsoletos por huella de entradas (session_state).
  - 7 pestañas: Geometría | Estratigrafía | Cálculo | Resultados |
    Red de flujo | Verificación | Exportación.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from malla import Capa, ContornoRect, generar_malla, verificar_malla, TipoNodo
from flujo import resolver_h, caudal_entrada, contraste_permeabilidad
from postproceso import (campo_gradiente, gradiente_salida,
                         presion_intersticial, subpresion_en_cota, GAMMA_W)


# =========================================================================== #
#  Utilidades de conversión de unidades (solo en la frontera)                  #
# =========================================================================== #
def k_a_si(valor: float, unidad: str) -> float:
    """Permeabilidad de la unidad de interfaz a m/s."""
    return valor * 0.01 if unidad == "cm/s" else valor


def k_desde_si(valor_si: float, unidad: str) -> float:
    return valor_si / 0.01 if unidad == "cm/s" else valor_si


def caudal_desde_si(q_si: float, unidad: str) -> float:
    """q_si en m3/s/m a la unidad pedida."""
    if unidad == "l/s/m":
        return q_si * 1000.0
    if unidad == "l/min/m":
        return q_si * 1000.0 * 60.0
    return q_si  # m3/s/m


# =========================================================================== #
#  Construcción del modelo desde los datos de entrada (espesores -> cotas)     #
# =========================================================================== #
def construir_contorno(datos: dict) -> ContornoRect:
    """
    Convierte los datos de entrada (con capas por espesor, ancladas arriba)
    en un ContornoRect en cotas absolutas. Frontera entrada->motor.
    """
    cota_arranque = datos["cota_arranque"]          # = z_sup (superficie/lecho alto)
    capas_in = datos["capas"]                       # lista de dicts: espesor, kx, kz, nombre

    # Apilar espesores de arriba hacia abajo
    capas = []
    z_techo = cota_arranque
    for cap in capas_in:
        z_muro = z_techo - cap["espesor"]
        capas.append(Capa(z_muro=z_muro, z_techo=z_techo,
                          kx=cap["kx"], kz=cap["kz"], nombre=cap["nombre"]))
        z_techo = z_muro
    # z_impermeable derivado = muro de la última capa
    capas = list(reversed(capas))  # de muro a techo

    return ContornoRect(
        x_izq=datos["x_izq"], x_der=datos["x_der"],
        x_tablestaca=datos["x_tablestaca"],
        z_coronacion=datos["z_coronacion"], z_pie=datos["z_pie"],
        z_lecho_arriba=datos["z_lecho_arriba"],
        z_lecho_abajo=datos["z_lecho_abajo"],
        lado_arriba_izq=datos["lado_arriba_izq"],
        h1=datos["h1"], h2=datos["h2"],
        capas=capas,
    )


def huella_entradas(datos: dict) -> str:
    """Hash de los datos de entrada para detectar resultados obsoletos."""
    blob = json.dumps(datos, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# =========================================================================== #
#  Validaciones cruzadas previas al cálculo                                    #
# =========================================================================== #
def validar(datos: dict) -> list[str]:
    """Devuelve lista de errores (vacía si todo correcto)."""
    e = []
    if datos["x_der"] <= datos["x_izq"]:
        e.append("x_der debe ser mayor que x_izq.")
    if not (datos["x_izq"] < datos["x_tablestaca"] < datos["x_der"]):
        e.append("La tablestaca debe estar dentro del dominio.")
    if datos["h1"] < datos["h2"]:
        e.append("h1 (aguas arriba) debería ser ≥ h2 (aguas abajo). "
                 "Si se invierte, el flujo cambia de sentido.")
    if not datos["capas"]:
        e.append("Debe definirse al menos una capa.")
    espesor_total = sum(c["espesor"] for c in datos["capas"])
    z_imp = datos["cota_arranque"] - espesor_total
    if datos["z_pie"] <= z_imp:
        e.append(f"El pie de la tablestaca (z={datos['z_pie']}) debe quedar por "
                 f"encima del impermeable (z={z_imp:.2f}).")
    z_lecho_min = min(datos["z_lecho_arriba"], datos["z_lecho_abajo"])
    if datos["z_pie"] >= z_lecho_min:
        e.append("El pie debe estar por debajo de ambos lechos.")
    for c in datos["capas"]:
        if c["kx"] <= 0 or c["kz"] <= 0:
            e.append(f"Capa '{c['nombre']}': permeabilidades deben ser > 0.")
        ratio = c["kx"] / c["kz"] if c["kz"] > 0 else 0
        if ratio > 100 or ratio < 0.01:
            e.append(f"Capa '{c['nombre']}': ratio kx/kz={ratio:.1f} fuera de "
                     f"rango físico habitual (¿error de unidades?).")
    return e


# =========================================================================== #
#  Cálculo completo                                                            #
# =========================================================================== #
def recortar_capa_impermeable(datos: dict, idx_capa: int) -> dict:
    """
    Devuelve una copia de los datos donde las capas desde idx_capa hacia abajo
    se eliminan del dominio de flujo: el techo de esa capa pasa a ser el nuevo
    sustrato impermeable. Útil cuando una capa es tan poco permeable que actúa
    como base impermeable.
    """
    import copy
    d = copy.deepcopy(datos)
    capas = d["capas"]
    # las capas están ordenadas de arriba (superficie) hacia abajo
    d["capas"] = capas[:idx_capa]
    d["capas_display"] = d.get("capas_display", capas)[:idx_capa]
    return d


def calcular(datos: dict, densidad: str) -> dict:
    c = construir_contorno(datos)
    m = generar_malla(c, densidad=densidad)
    err_malla = verificar_malla(m)
    contraste = contraste_permeabilidad(m)
    sol = resolver_h(m)
    Q, Qh1, Qh2 = caudal_entrada(sol)
    grad = campo_gradiente(sol)
    sif = gradiente_salida(
        sol,
        gamma_sat=datos.get("gamma_sat"),
        G_s=datos.get("G_s"), e=datos.get("e"))
    u = presion_intersticial(sol)
    subp = subpresion_en_cota(sol, c.z_pie)
    return dict(contorno=c, malla=m, sol=sol, Q=Q, Qh1=Qh1, Qh2=Qh2,
                grad=grad, sif=sif, u=u, subp=subp, err_malla=err_malla,
                contraste=contraste)


# =========================================================================== #
#  Visualizaciones (Plotly)                                                    #
# =========================================================================== #
def _factor_forma(res: dict) -> float:
    """
    Factor de forma Nf/Nd = Q/(k·ΔH) de la red de flujo. Para una sola capa
    es exacto; para multicapa usa una permeabilidad equivalente (media
    armónica ponderada por espesor, dominante en flujo bajo pantalla) como
    aproximación — los cuadrados serán aproximados, como en toda red de flujo
    en terreno estratificado.
    """
    c = res["contorno"]
    Q = res["Q"]
    dH = c.h1 - c.h2
    if dH <= 0:
        return 1.0
    if len(c.capas) == 1:
        k_ref = np.sqrt(c.capas[0].kx * c.capas[0].kz)
    else:
        # media armónica ponderada por espesor (flujo predominantemente vertical
        # bajo el pie atraviesa las capas en serie)
        esp = sum(cap.z_techo - cap.z_muro for cap in c.capas)
        denom = sum((cap.z_techo - cap.z_muro) / np.sqrt(cap.kx * cap.kz)
                    for cap in c.capas)
        k_ref = esp / denom if denom else 1.0
    return Q / (k_ref * dH) if k_ref else 1.0


def fig_red_flujo(res: dict, n_flujo: int = 5, n_equip: int | None = None,
                  modo: str = "clasico") -> go.Figure:
    """
    Red de flujo: equipotenciales (azul) + líneas de corriente (rojo, por psi).

    modo='clasico': n_flujo canales de flujo; n_equip se calcula para formar
                    cuadrados curvilíneos (Nd = Nf / factor_forma).
    modo='libre':   n_flujo y n_equip independientes (sin restricción de cuadrados).
    """
    m = res["malla"]; sol = res["sol"]; c = res["contorno"]
    H = sol.h_nodo.T
    X, Z = np.meshgrid(m.x, m.z)

    # número de equipotenciales
    if modo == "clasico":
        S = _factor_forma(res)
        nd = max(2, int(round(n_flujo / S))) if S > 0 else 10
    else:
        nd = n_equip if n_equip else 12

    fig = go.Figure()
    # equipotenciales: niveles equiespaciados en h (de h2 a h1)
    niveles_h = np.linspace(c.h2, c.h1, nd + 1)
    fig.add_trace(go.Contour(
        x=m.x, y=m.z, z=H, contours_coloring='lines',
        line_width=1.2, colorscale='Blues', showscale=False,
        contours=dict(start=c.h2, end=c.h1, size=(c.h1 - c.h2) / nd),
        name="Equipotenciales"))

    # líneas de corriente: niveles equiespaciados en psi (canales iguales)
    try:
        from corriente import resolver_psi
        psi, Q = resolver_psi(sol)
        PSI = psi.T
        fig.add_trace(go.Contour(
            x=m.x, y=m.z, z=PSI, contours_coloring='lines',
            line_width=1.2, colorscale=[[0, '#C0392B'], [1, '#C0392B']],
            showscale=False,
            contours=dict(start=0, end=Q, size=Q / n_flujo),
            name="Líneas de corriente"))
    except Exception:
        pass

    # tablestaca y lechos
    fig.add_trace(go.Scatter(
        x=[c.x_tablestaca, c.x_tablestaca], y=[c.z_pie, c.z_coronacion],
        mode="lines", line=dict(color="black", width=4), name="Tablestaca"))
    xa_lo, xa_hi = c.x_arriba; xb_lo, xb_hi = c.x_abajo
    fig.add_trace(go.Scatter(x=[xa_lo, xa_hi], y=[c.z_lecho_arriba]*2,
        mode="lines", line=dict(color="saddlebrown", width=2),
        name="Lecho arriba"))
    fig.add_trace(go.Scatter(x=[xb_lo, xb_hi], y=[c.z_lecho_abajo]*2,
        mode="lines", line=dict(color="saddlebrown", width=2),
        name="Lecho abajo"))
    titulo = (f"Red de flujo — {n_flujo} canales, {nd} saltos"
              + (" (cuadrados curvilíneos)" if modo == "clasico" else ""))
    fig.update_layout(title=titulo, xaxis_title="x (m)", yaxis_title="z (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1), height=500)
    return fig


def fig_mapa(res: dict, campo: str) -> go.Figure:
    m = res["malla"]; sol = res["sol"]; grad = res["grad"]; c = res["contorno"]
    if campo == "carga":
        Z = sol.h_nodo.T; x = m.x; y = m.z; titulo = "Carga hidráulica h (m)"
        cs = "Viridis"
    else:  # gradiente
        Z = grad["imod"].T
        x = 0.5 * (m.x[:-1] + m.x[1:]); y = 0.5 * (m.z[:-1] + m.z[1:])
        titulo = "Módulo del gradiente |i|"; cs = "Hot"
    fig = go.Figure(go.Heatmap(x=x, y=y, z=Z, colorscale=cs,
                               colorbar=dict(title="")))
    fig.add_trace(go.Scatter(
        x=[c.x_tablestaca, c.x_tablestaca], y=[c.z_pie, c.z_coronacion],
        mode="lines", line=dict(color="white", width=3), showlegend=False))
    fig.update_layout(title=titulo, xaxis_title="x (m)", yaxis_title="z (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1), height=500)
    return fig


def fig_subpresion(res: dict) -> go.Figure:
    subp = res["subp"]
    fig = go.Figure(go.Scatter(x=subp["x"], y=subp["u"], mode="lines+markers",
                               fill="tozeroy", name="u"))
    fig.update_layout(
        title=f"Subpresión en cota z={subp['z_base']:.2f} m "
              f"(resultante {subp['resultante']:.0f} kN/m)",
        xaxis_title="x (m)", yaxis_title="u (kN/m²)", height=350)
    return fig


# =========================================================================== #
#  INTERFAZ STREAMLIT                                                          #
# =========================================================================== #
def main():
    st.set_page_config(page_title="Flujo bajo tablestaca", layout="wide")
    st.title("Cálculo de flujo 2D bajo tablestaca — diferencias finitas")
    st.caption("Modelo 2D de sección por metro lineal · flujo confinado "
               "estacionario · multicapa anisótropo")

    if "datos" not in st.session_state:
        st.session_state.datos = _datos_por_defecto()
    if "resultado" not in st.session_state:
        st.session_state.resultado = None
        st.session_state.huella_calc = None

    datos = st.session_state.datos

    tabs = st.tabs([
        "1· Geometría", "2· Estratigrafía", "3· Cálculo",
        "4· Resultados", "5· Red de flujo", "6· Verificación", "7· Exportar"])

    # ---- Pestaña 1: Geometría ----
    with tabs[0]:
        st.subheader("Geometría del dominio y la tablestaca")
        col1, col2 = st.columns(2)
        with col1:
            datos["x_izq"] = st.number_input("x izquierdo (m)", value=datos["x_izq"])
            datos["x_der"] = st.number_input("x derecho (m)", value=datos["x_der"])
            datos["x_tablestaca"] = st.number_input(
                "x tablestaca (m)", value=datos["x_tablestaca"])
            datos["lado_arriba_izq"] = st.radio(
                "Aguas arriba está a la…", ["Izquierda", "Derecha"],
                index=0 if datos["lado_arriba_izq"] else 1) == "Izquierda"
        with col2:
            datos["cota_arranque"] = st.number_input(
                "Cota de superficie / arranque de capas (m)",
                value=datos["cota_arranque"])
            datos["z_lecho_arriba"] = st.number_input(
                "Cota lecho aguas arriba (m)", value=datos["z_lecho_arriba"])
            datos["z_lecho_abajo"] = st.number_input(
                "Cota lecho aguas abajo (m)", value=datos["z_lecho_abajo"])
            datos["z_coronacion"] = st.number_input(
                "Cota coronación tablestaca (m)", value=datos["z_coronacion"])
            datos["z_pie"] = st.number_input(
                "Cota pie tablestaca (m)", value=datos["z_pie"])
        st.markdown("**Cargas hidráulicas**")
        col3, col4 = st.columns(2)
        with col3:
            datos["h1"] = st.number_input("h1 aguas arriba (m)", value=datos["h1"])
        with col4:
            datos["h2"] = st.number_input("h2 aguas abajo (m)", value=datos["h2"])
        st.info(f"ΔH = h1 − h2 = **{datos['h1'] - datos['h2']:.3f} m**")

        # Croquis dinámico de la sección (se redibuja con cada cambio)
        st.markdown("**Croquis de la sección**")
        st.caption("Se actualiza automáticamente al modificar los datos. "
                   "Útil para verificar la geometría antes de calcular.")
        try:
            from croquis import fig_croquis
            st.pyplot(fig_croquis(datos))
        except Exception as ex:
            st.info(f"Croquis no disponible: {ex}")

    # ---- Pestaña 2: Estratigrafía ----
    with tabs[1]:
        st.subheader("Capas (de la superficie hacia abajo, por espesores)")
        datos["unidad_k"] = st.radio("Unidad de permeabilidad",
                                     ["m/s", "cm/s"], horizontal=True,
                                     index=0 if datos["unidad_k"] == "m/s" else 1)
        uk = datos["unidad_k"]
        df = pd.DataFrame(datos["capas_display"])
        df_edit = st.data_editor(df, num_rows="dynamic", use_container_width=True,
            column_config={
                "nombre": "Nombre",
                "espesor": st.column_config.NumberColumn("Espesor (m)", min_value=0.0),
                "kx": st.column_config.NumberColumn(f"kx ({uk})", format="%.2e"),
                "kz": st.column_config.NumberColumn(f"kz ({uk})", format="%.2e"),
            })
        datos["capas_display"] = df_edit.to_dict("records")
        # convertir a SI para el motor
        datos["capas"] = [
            dict(nombre=r["nombre"], espesor=float(r["espesor"]),
                 kx=k_a_si(float(r["kx"]), uk), kz=k_a_si(float(r["kz"]), uk))
            for r in datos["capas_display"]]
        esp_total = sum(c["espesor"] for c in datos["capas"])
        z_imp = datos["cota_arranque"] - esp_total
        st.info(f"Espesor total = {esp_total:.2f} m → "
                f"impermeable derivado en cota z = **{z_imp:.2f} m**")

    # ---- Pestaña 3: Cálculo ----
    with tabs[2]:
        st.subheader("Parámetros y ejecución")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sifonamiento (opcional)**")
            usar_sif = st.checkbox("Evaluar sifonamiento",
                                   value=datos.get("usar_sif", True))
            datos["usar_sif"] = usar_sif
            if usar_sif:
                metodo = st.radio("Método de i crítico",
                                  ["G_s y e", "γ_sat"], horizontal=True)
                if metodo == "G_s y e":
                    datos["G_s"] = st.number_input("G_s", value=datos.get("G_s") or 2.65)
                    datos["e"] = st.number_input("Índice de poros e",
                                                 value=datos.get("e") or 0.6)
                    datos["gamma_sat"] = None
                else:
                    datos["gamma_sat"] = st.number_input(
                        "γ_sat (kN/m³)", value=datos.get("gamma_sat") or 20.0)
                    datos["G_s"] = None; datos["e"] = None
            else:
                datos["G_s"] = datos["e"] = datos["gamma_sat"] = None
        with col2:
            st.markdown("**Control numérico**")
            datos["densidad"] = st.select_slider(
                "Densidad de malla", ["grosero", "normal", "fino"],
                value=datos.get("densidad", "normal"))
            datos["unidad_q"] = st.selectbox("Unidad de caudal",
                ["m³/s/m", "l/s/m", "l/min/m"],
                index=["m³/s/m", "l/s/m", "l/min/m"].index(
                    datos.get("unidad_q", "l/s/m")))

        errores = validar(datos)
        if errores:
            st.error("Corrige antes de calcular:")
            for er in errores:
                st.write("•", er)
        else:
            if st.button("▶ Calcular", type="primary"):
                with st.spinner("Resolviendo…"):
                    res = calcular(datos, datos["densidad"])
                st.session_state.resultado = res
                st.session_state.huella_calc = huella_entradas(datos)
                st.success(f"Cálculo completado · malla "
                           f"{res['malla'].nx}×{res['malla'].nz} = "
                           f"{res['malla'].nx*res['malla'].nz} nodos")
                # Aviso de contraste de permeabilidad extremo
                ct = res.get("contraste", {})
                if ct.get("nivel") == "extremo" and ct.get("capas_debiles"):
                    nombres = ", ".join(n for _, n, _ in ct["capas_debiles"])
                    st.warning(
                        f"⚠ Contraste de permeabilidad muy alto "
                        f"(ratio {ct['ratio']:.0e}). La(s) capa(s) «{nombres}» "
                        f"son varios órdenes menos permeables que el resto y "
                        f"apenas conducen flujo. El cálculo se ha resuelto de "
                        f"forma estable, pero si esa capa actúa como base "
                        f"impermeable, considera tratarla como sustrato "
                        f"impermeable (opción abajo) para un modelo más limpio.")
                    # ofrecer recorte de la capa más profunda débil
                    idx_deb = max(i for i, _, _ in ct["capas_debiles"])
                    if st.button("Tratar capas inferiores como sustrato "
                                 "impermeable y recalcular"):
                        datos_rec = recortar_capa_impermeable(datos, idx_deb)
                        st.session_state.datos = datos_rec
                        res = calcular(datos_rec, datos_rec["densidad"])
                        st.session_state.resultado = res
                        st.session_state.huella_calc = huella_entradas(datos_rec)
                        st.success("Recalculado tratando las capas inferiores "
                                   "como sustrato impermeable.")

    # ---- Detección de resultados obsoletos ----
    res = st.session_state.resultado
    obsoleto = (res is not None and
                st.session_state.huella_calc != huella_entradas(datos))

    def _guard():
        if res is None:
            st.warning("Aún no hay resultados. Ve a la pestaña «Cálculo».")
            return False
        if obsoleto:
            st.warning("⚠ Las entradas han cambiado desde el último cálculo. "
                       "Los resultados mostrados están **desactualizados**; "
                       "recalcula en la pestaña «Cálculo».")
        return True

    # ---- Pestaña 4: Resultados ----
    with tabs[3]:
        st.subheader("Resultados numéricos")
        if _guard():
            uq = datos["unidad_q"]
            Q = caudal_desde_si(res["Q"], uq)
            sif = res["sif"]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Caudal Q ({uq})", f"{Q:.4g}")
            c2.metric("i_exit máx", f"{sif.i_exit_max:.3f}")
            if sif.FS is not None:
                color = "normal" if sif.FS >= 3 else "inverse"
                c3.metric("FS sifonamiento", f"{sif.FS:.2f}",
                          delta="OK" if sif.FS >= 3 else "BAJO",
                          delta_color=color)
            st.divider()
            tabla = {
                "Magnitud": ["Caudal de filtración", "i_exit máximo",
                             "i_exit medio", "i crítico", "FS sifonamiento",
                             "Resultante subpresión (cota pie)"],
                "Valor": [f"{Q:.4g} {uq}", f"{sif.i_exit_max:.4f}",
                          f"{sif.i_exit_medio:.4f}",
                          f"{sif.i_critico:.4f}" if sif.i_critico else "—",
                          f"{sif.FS:.2f}" if sif.FS else "—",
                          f"{res['subp']['resultante']:.0f} kN/m"],
            }
            st.table(pd.DataFrame(tabla))
            st.caption(f"i crítico: {sif.metodo_icrit}")

            # --- Caudal por franja bajo el pie (función de corriente) ---
            st.divider()
            st.markdown("**Caudal a través de una franja bajo el pie**")
            cc = res["contorno"]
            prof_max = cc.z_pie - cc.z_imp
            st.caption(f"Distancia entre el pie de la pantalla y el "
                       f"impermeable: {prof_max:.2f} m. El caudal indica qué "
                       f"parte del flujo total cruza la franja vertical desde "
                       f"el pie hasta la profundidad elegida.")
            prof = st.slider("Profundidad bajo el pie (m)", 0.0, float(prof_max),
                             float(prof_max), step=prof_max / 20 if prof_max else 0.1)
            if prof > 0:
                try:
                    from corriente import resolver_psi, caudal_franja_bajo_pie
                    if (st.session_state.get("psi_huella") !=
                            st.session_state.huella_calc):
                        psi, Qpsi = resolver_psi(res["sol"])
                        st.session_state.psi_cache = psi
                        st.session_state.psi_huella = st.session_state.huella_calc
                    psi = st.session_state.psi_cache
                    rfr = caudal_franja_bajo_pie(res["sol"], psi, prof)
                    qfr = caudal_desde_si(rfr["caudal"], uq)
                    cc1, cc2 = st.columns(2)
                    cc1.metric(f"Caudal en la franja ({uq})", f"{qfr:.4g}")
                    cc2.metric("Fracción del caudal total",
                               f"{rfr['fraccion']*100:.1f} %")
                except Exception as ex:
                    st.info(f"No se pudo calcular el caudal por franja: {ex}")

    # ---- Pestaña 5: Red de flujo ----
    with tabs[4]:
        st.subheader("Red de flujo y campos")
        if _guard():
            colm1, colm2, colm3 = st.columns([1, 1, 1])
            with colm1:
                modo = st.radio("Modo de red de flujo",
                                ["Clásico (cuadrados)", "Libre"],
                                help="Clásico: las equipotenciales se calculan "
                                "para formar cuadrados curvilíneos. Libre: "
                                "controlas ambos números por separado.")
                modo_key = "clasico" if modo.startswith("Clásico") else "libre"
            with colm2:
                n_flujo = st.slider("Canales de flujo", 3, 20, 5)
            with colm3:
                if modo_key == "libre":
                    n_equip = st.slider("Equipotenciales", 4, 40, 12)
                else:
                    n_equip = None
                    S = _factor_forma(res)
                    nd_auto = max(2, int(round(n_flujo / S))) if S > 0 else 10
                    st.metric("Saltos equipotenciales", nd_auto)
            if modo_key == "clasico" and len(res["contorno"].capas) > 1:
                st.caption("⚠ En terreno estratificado los cuadrados son "
                           "aproximados (cada capa tiene su propia escala).")
            st.plotly_chart(
                fig_red_flujo(res, n_flujo=n_flujo, n_equip=n_equip,
                              modo=modo_key),
                use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_mapa(res, "carga"), use_container_width=True)
            with col2:
                st.plotly_chart(fig_mapa(res, "gradiente"), use_container_width=True)
            st.plotly_chart(fig_subpresion(res), use_container_width=True)

    # ---- Pestaña 6: Verificación ----
    with tabs[5]:
        st.subheader("Autoverificación del cálculo")
        if _guard():
            Qh1, Qh2 = res["Qh1"], res["Qh2"]
            residuo = abs(Qh1 + Qh2) / abs(Qh1) if Qh1 else 0
            c = res["contorno"]
            # test simetría si aplica
            simetrico = np.isclose(c.z_lecho_arriba, c.z_lecho_abajo)
            checks = [
                ("Conservación de masa (Q entrada = Q salida)",
                 f"residuo {residuo:.2e}", residuo < 1e-6),
                ("Malla geométricamente correcta",
                 "sin errores" if not res["err_malla"] else
                 f"{len(res['err_malla'])} errores", not res["err_malla"]),
                ("Rango físico de h dentro de [h2, h1]",
                 f"[{np.nanmin(res['sol'].h_nodo):.2f}, "
                 f"{np.nanmax(res['sol'].h_nodo):.2f}]",
                 np.nanmin(res['sol'].h_nodo) >= c.h2 - 1e-6 and
                 np.nanmax(res['sol'].h_nodo) <= c.h1 + 1e-6),
            ]
            if simetrico:
                i_tab = int(np.argmin(np.abs(res["malla"].x - c.x_tablestaca)))
                hmed = (c.h1 + c.h2) / 2
                desv = 0.0
                for j in range(res["malla"].nz):
                    if res["malla"].z[j] < c.z_pie - 1e-9:
                        h = res["sol"].h_nodo[i_tab, j]
                        if not np.isnan(h):
                            desv = max(desv, abs(h - hmed))
                checks.append(
                    ("Equipotencial media bajo el pie (geometría simétrica)",
                     f"desviación {desv:.2e} m", desv < 1e-6))
            for nombre, detalle, ok in checks:
                st.write(f"{'✅' if ok else '❌'} **{nombre}** — {detalle}")

    # ---- Pestaña 7: Exportar ----
    with tabs[6]:
        st.subheader("Exportación")
        if _guard():
            uq = datos["unidad_q"]

            # Metadatos opcionales para la portada del informe
            with st.expander("Datos para la portada del informe (opcional)"):
                col1, col2 = st.columns(2)
                with col1:
                    meta_obra = st.text_input("Obra", "")
                    meta_tramo = st.text_input("Tramo / Estructura", "")
                    meta_pet = st.text_input("Peticionario", "")
                with col2:
                    meta_autor = st.text_input("Autor", "")
                    meta_fecha = st.text_input("Fecha", "")
                st.caption("Si se dejan en blanco, el informe imprime líneas "
                           "para rellenar a mano.")
            metadatos = {k: v for k, v in {
                "obra": meta_obra, "tramo": meta_tramo,
                "peticionario": meta_pet, "autor": meta_autor,
                "fecha": meta_fecha}.items() if v}

            # --- construir Excel en memoria ---
            def _construir_excel() -> bytes:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as xl:
                    pd.DataFrame({
                        "Magnitud": ["Q", "i_exit_max", "i_exit_medio",
                                     "i_critico", "FS", "Resultante subpresión"],
                        "Valor": [caudal_desde_si(res["Q"], uq),
                                  res["sif"].i_exit_max, res["sif"].i_exit_medio,
                                  res["sif"].i_critico, res["sif"].FS,
                                  res["subp"]["resultante"]],
                        "Unidad": [uq, "-", "-", "-", "-", "kN/m"]
                    }).to_excel(xl, sheet_name="Resumen", index=False)
                    pd.DataFrame(res["sol"].h_nodo, index=res["malla"].x,
                                 columns=res["malla"].z).to_excel(
                                     xl, sheet_name="Carga_h")
                return buf.getvalue()

            # --- construir Word en memoria ---
            def _construir_word() -> bytes:
                import tempfile, os
                from informe import generar_informe
                with tempfile.TemporaryDirectory() as tmp:
                    ruta = os.path.join(tmp, "informe_calculo.docx")
                    generar_informe(datos, res, metadatos=metadatos or None,
                                    salida=ruta)
                    with open(ruta, "rb") as f:
                        return f.read()

            st.markdown("**Informe de cálculo completo (Word + Excel)**")
            if st.button("📄 Generar informe ZIP (Word + Excel)", type="primary"):
                with st.spinner("Generando informe…"):
                    import zipfile
                    word_bytes = _construir_word()
                    excel_bytes = _construir_excel()
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                        z.writestr("informe_calculo.docx", word_bytes)
                        z.writestr("resultados.xlsx", excel_bytes)
                    st.session_state.zip_bytes = zbuf.getvalue()
                st.success("Informe generado.")

            if st.session_state.get("zip_bytes"):
                st.download_button(
                    "📥 Descargar informe (ZIP)",
                    st.session_state.zip_bytes,
                    "informe_flujo_tablestaca.zip", "application/zip")

            st.divider()
            st.markdown("**Descargas sueltas**")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📊 Solo Excel", _construir_excel(),
                    "resultados.xlsx",
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet")
            with col2:
                if st.button("📄 Solo Word"):
                    st.session_state.word_bytes = _construir_word()
                if st.session_state.get("word_bytes"):
                    st.download_button("📥 Descargar Word",
                        st.session_state.word_bytes, "informe_calculo.docx",
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document")


def _datos_por_defecto() -> dict:
    return dict(
        x_izq=0.0, x_der=40.0, x_tablestaca=20.0,
        cota_arranque=10.0,
        z_lecho_arriba=10.0, z_lecho_abajo=10.0,
        z_coronacion=13.0, z_pie=4.0,
        lado_arriba_izq=True, h1=12.0, h2=10.5,
        unidad_k="m/s", unidad_q="l/s/m", densidad="normal",
        usar_sif=True, G_s=2.65, e=0.6, gamma_sat=None,
        capas_display=[dict(nombre="Arena", espesor=10.0, kx=1e-5, kz=1e-5)],
        capas=[dict(nombre="Arena", espesor=10.0, kx=1e-5, kz=1e-5)],
    )


if __name__ == "__main__":
    main()
