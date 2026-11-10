"""
Aplicación de Streamlit: riesgo de deforestación por zona.
Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia

Ejecutar:  streamlit run app.py
Requiere en la misma carpeta: modelo_final.joblib e info_modelo.json (se crean con entrenar_modelo.py).
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

CARPETA = Path(__file__).resolve().parent
st.set_page_config(page_title="Riesgo de deforestación", page_icon="🌳", layout="wide")


# ---------------------------------------------------------------------------
# Carga del modelo (el mismo pipeline del entrenamiento: preprocesamiento + SMOTE + regresión logística)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar():
    modelo = joblib.load(CARPETA / "modelo_final.joblib")
    info = json.loads((CARPETA / "info_modelo.json").read_text(encoding="utf-8"))
    return modelo, info


modelo, info = cargar()
DIST, CAT, NOM = info["distancias"], info["categoricas"], info["nombres"]
COLUMNAS = DIST + CAT
UMBRAL = info["umbral"]
CORTOS = {"disedge09": "Borde del bosque (2009)", "Distriv": "Ríos", "dist_roads": "Vías",
          "d_IR": "Resguardos indígenas", "d_II_III": "Áreas protegidas", "rec2001c1": "Cobertura 2001"}


# ---------------------------------------------------------------------------
# Verificación de coherencia entre las entradas y los datos de entrenamiento
# ---------------------------------------------------------------------------
def verificar(fila):
    """Devuelve (errores, avisos) para una zona."""
    errores, avisos = [], []
    for c in DIST:
        v, r = fila[c], info["rangos"][c]
        if pd.isna(v):
            avisos.append(f"**{NOM[c]}** está vacía: se imputará con la mediana del entrenamiento.")
        elif v < 0:
            errores.append(f"**{NOM[c]}** no puede ser negativa.")
        elif v < r["min"] or v > r["max"]:
            avisos.append(f"**{NOM[c]}** = {v:g} está fuera del rango de entrenamiento "
                          f"({r['min']:g} a {r['max']:g}): el modelo estaría extrapolando.")
    for c in CAT:
        v = fila[c]
        if pd.isna(v) or int(v) not in info["clases_rec2001c1"]:
            avisos.append(f"**{NOM[c]}** = {v} no apareció en el entrenamiento; se tratará como la clase de referencia.")
    return errores, avisos


def predecir(df):
    X = df[COLUMNAS].copy()
    for c in DIST:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return modelo.predict_proba(X)[:, 1]


def contribuciones(fila):
    """Aporte de cada variable a los log-odds: coeficiente × valor transformado (estandarizado)."""
    X = pd.DataFrame([fila])[COLUMNAS]
    z = modelo.named_steps["preprocesamiento"].transform(X)[0]
    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()
    aportes = {}
    for nombre, valor in zip(nombres, z):
        grupo, var = nombre.split("__", 1)
        base = var if grupo == "distancias" else var.rsplit("_", 1)[0]
        aportes[base] = aportes.get(base, 0.0) + info["coeficientes"][nombre] * float(valor)
    return pd.DataFrame({"Variable": [CORTOS.get(k, NOM[k]) for k in aportes], "Aporte": list(aportes.values())})


# ---------------------------------------------------------------------------
# Barra lateral: tarjeta del modelo
# ---------------------------------------------------------------------------
m, b = info["metricas_prueba"], info["metricas_bloques_20km"]
with st.sidebar:
    st.header("🌳 Tarjeta del modelo")
    st.markdown(f"**Modelo:** {info['modelo']}  \n**Umbral de alerta:** {UMBRAL:.3f}")
    st.caption(f"Elegido con probabilidades fuera de fold del entrenamiento para lograr recall ≥ {info['meta_recall']:.2f}.")
    st.subheader("Desempeño en prueba")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['recall']:.3f}")
    c2.metric("Precisión", f"{m['precision']:.3f}")
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    st.caption(f"Detectó {m['tp']} de {m['positivos']} zonas deforestadas en {m['registros']:,} zonas de prueba. "
               f"Con bloques espaciales de 20 km: ROC-AUC {b['roc_auc']:.3f}.".replace(",", "."))
    st.subheader("Limitaciones")
    st.markdown(f"- De cada 100 alertas, unas **{100 * m['precision']:.0f}** son deforestación real.\n"
                "- Falla más en zonas lejos de las vías y sin incendio.\n"
                "- Sirve para **priorizar el monitoreo**, no reemplaza la verificación en campo.")


# ---------------------------------------------------------------------------
# Contenido principal
# ---------------------------------------------------------------------------
st.title("Riesgo de deforestación por zona")
st.markdown("Estima la probabilidad de que una zona de la Amazonía colombiana se deforeste, a partir de sus "
            "distancias a vías, ríos, borde del bosque, resguardos y áreas protegidas, y de su cobertura en 2001. "
            "Reproduce, con datos colombianos, el objetivo de **Rosa et al. (2013)**.")

tab_una, tab_varias, tab_ayuda = st.tabs(["🔎 Una zona", "📄 Varias zonas (CSV)", "ℹ️ Cómo usar"])

# --- Pestaña 1: una zona -----------------------------------------------------
with tab_una:
    opciones = ["Valores personalizados"] + list(info["ejemplos"])
    elegido = st.selectbox("Cargar un ejemplo real del conjunto de prueba", opciones,
                           help="Zonas reales que el modelo no vio al entrenar.")
    if elegido != "Valores personalizados":
        base = info["ejemplos"][elegido]["valores"]
        real = info["ejemplos"][elegido]["real"]
        st.caption(f"Valor real de esta zona: **{'deforestada' if real == 1 else 'sin deforestación'}**.")
    else:
        base = {c: info["rangos"][c]["mediana"] for c in DIST}
        base[CAT[0]] = info.get("clase_frecuente", info["clases_rec2001c1"][0])

    with st.form("zona"):
        st.markdown("**Características de la zona** (en las unidades del conjunto de datos original)")
        cols = st.columns(3)
        fila = {}
        for i, c in enumerate(DIST):
            r = info["rangos"][c]
            fila[c] = cols[i % 3].number_input(
                NOM[c], min_value=0.0, value=float(base[c]), format="%.4f", key=f"{elegido}_{c}",
                help=f"Rango de entrenamiento: {r['min']:g} a {r['max']:g} · mediana {r['mediana']:g}")
        clases = info["clases_rec2001c1"]
        fila[CAT[0]] = cols[len(DIST) % 3].selectbox(
            NOM[CAT[0]], clases, index=clases.index(int(base[CAT[0]])) if int(base[CAT[0]]) in clases else 0,
            key=f"{elegido}_{CAT[0]}", help="Código de cobertura del conjunto de datos original.")
        enviar = st.form_submit_button("Predecir riesgo", type="primary")

    if enviar:
        errores, avisos = verificar(fila)
        for e in errores:
            st.error(e)
        for a in avisos:
            st.warning(a)
        if not errores:
            p = float(predecir(pd.DataFrame([fila]))[0])
            alerta = p >= UMBRAL
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Probabilidad de deforestación", f"{p:.1%}")
                st.progress(min(p, 1.0))
                if alerta:
                    st.error(f"⚠️ **ALERTA**: la probabilidad supera el umbral de {UMBRAL:.3f}. Zona prioritaria para monitoreo.")
                else:
                    st.success(f"✅ **Sin alerta**: la probabilidad está por debajo del umbral de {UMBRAL:.3f}.")
                st.caption("La probabilidad está calibrada sobre datos balanceados con SMOTE: úsela para ordenar zonas "
                           "por riesgo y compárela con el umbral, no como un porcentaje literal.")
            with c2:
                aportes = contribuciones(fila).sort_values("Aporte")
                aportes["Efecto"] = np.where(aportes["Aporte"] > 0, "Sube el riesgo", "Baja el riesgo")
                graf = alt.Chart(aportes).mark_bar().encode(
                    x=alt.X("Aporte:Q", title="Aporte al riesgo (log-odds)"),
                    y=alt.Y("Variable:N", sort=list(aportes["Variable"]), title=None,
                            axis=alt.Axis(labelOverlap=False, labelLimit=220, labelFontSize=13)),
                    color=alt.Color("Efecto:N", scale=alt.Scale(domain=["Sube el riesgo", "Baja el riesgo"],
                                                                range=["#D9622B", "#1F4D3A"]), legend=alt.Legend(orient="bottom")),
                    tooltip=["Variable", alt.Tooltip("Aporte:Q", format=".2f")]).properties(
                    height=300, title="¿Qué variables movieron esta predicción?")
                st.altair_chart(graf, width="stretch")
                mayor = aportes.iloc[aportes["Aporte"].abs().argmax()]
                st.caption(f"La variable que más pesó fue **{mayor['Variable']}**, que "
                           f"{'subió' if mayor['Aporte'] > 0 else 'bajó'} el riesgo de esta zona frente a una zona promedio.")

# --- Pestaña 2: varias zonas -------------------------------------------------
with tab_varias:
    st.markdown(f"Suba un CSV con estas columnas: `{', '.join(COLUMNAS)}`. "
                "Se devuelve el mismo archivo con la probabilidad y la alerta de cada zona.")
    plantilla = pd.DataFrame([e["valores"] for e in info["ejemplos"].values()])[COLUMNAS]
    st.download_button("Descargar plantilla de ejemplo", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_zonas.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        faltan = [c for c in COLUMNAS if c not in datos.columns]
        if faltan:
            st.error(f"Faltan estas columnas: {', '.join(faltan)}")
        else:
            problemas = [(i, e) for i, f in datos.iterrows() for e in verificar(f)[0]]
            if problemas:
                st.error(f"Hay {len(problemas)} valores inválidos (por ejemplo, distancias negativas). "
                         f"Primer caso, fila {problemas[0][0] + 1}: {problemas[0][1]}")
            else:
                n_avisos = sum(len(verificar(f)[1]) for _, f in datos.iterrows())
                if n_avisos:
                    st.warning(f"{n_avisos} valores están fuera del rango de entrenamiento o vacíos; revíselos.")
                datos["probabilidad"] = predecir(datos)
                datos["alerta"] = np.where(datos["probabilidad"] >= UMBRAL, "Sí", "No")
                c1, c2 = st.columns(2)
                c1.metric("Zonas evaluadas", f"{len(datos):,}".replace(",", "."))
                c2.metric("Zonas con alerta", f"{(datos['alerta'] == 'Sí').sum():,}".replace(",", "."))
                st.dataframe(datos.sort_values("probabilidad", ascending=False), width="stretch")
                st.download_button("Descargar resultados", datos.to_csv(index=False).encode("utf-8"),
                                   "resultados_riesgo.csv", "text/csv")

# --- Pestaña 3: ayuda --------------------------------------------------------
with tab_ayuda:
    st.markdown(f"""
**Qué hace la aplicación.** Recibe las características de una zona, les aplica exactamente el mismo
preprocesamiento del entrenamiento (imputación por mediana, log(1 + x) en las distancias, estandarización y
codificación de la cobertura) y la regresión logística entrenada, y devuelve la probabilidad de deforestación.
Si la probabilidad es mayor o igual a **{UMBRAL:.3f}**, se emite una alerta.

**Cómo leer el resultado.**
- *Probabilidad:* sirve para ordenar las zonas de mayor a menor riesgo.
- *Alerta:* el umbral se eligió para encontrar al menos 80 de cada 100 zonas deforestadas.
- *Gráfico de aportes:* barras naranjas suben el riesgo y verdes lo bajan, comparado con una zona promedio.

**Coherencia con el entrenamiento.** La aplicación avisa si un valor es negativo, está vacío o queda fuera del rango
visto en el entrenamiento, porque en esos casos el modelo estaría extrapolando.

**Variables que no usa.** `yloss`, `fire` y `yfire` (fuga de información), las distancias a pérdidas de 2010 a
2014, `rec2009c1` (medida después de muchas pérdidas) y la latitud y longitud (describen el lugar, no un mecanismo).

**Datos:** `bothgridandfire.csv` (16.759 zonas, 694 deforestadas). Semilla: {info['semilla']}.
""")
