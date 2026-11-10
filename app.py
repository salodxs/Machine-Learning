"""
Aplicación de Streamlit: riesgo de deforestación por zona (8.3 Despliegue básico).
Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia

Modelo final (8.1): regresión logística con SMOTE, 5 predictores (Modelo B, sin longitud),
validación espacial por bloques de 20 km y dos escenarios de umbral (8.2).

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
# Carga del modelo: el mismo pipeline del entrenamiento (log + z-score + SMOTE + regresión logística)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar():
    modelo = joblib.load(CARPETA / "modelo_final.joblib")
    info = json.loads((CARPETA / "info_modelo.json").read_text(encoding="utf-8"))
    return modelo, info


modelo, info = cargar()
COLUMNAS, NOM, ESC = info["predictores"], info["nombres"], info["escenarios"]
NOMBRES_ESC = list(ESC)
CORTOS = {"lat": "Latitud", "Distriv": "Ríos", "dist_roads": "Vías",
          "d_IR": "Resguardos indígenas", "d_II_III": "Áreas protegidas"}
UNIDAD = {"lat": "grados decimales"}


def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Verificación de coherencia entre las entradas y los datos de entrenamiento
# ---------------------------------------------------------------------------
def verificar(fila):
    """Devuelve (errores, avisos) para una zona."""
    errores, avisos = [], []
    for c in COLUMNAS:
        v, r = fila[c], info["rangos"][c]
        if pd.isna(v):
            errores.append(f"**{NOM[c]}** está vacía: el modelo necesita las 5 variables.")
        elif c != "lat" and v < 0:
            errores.append(f"**{NOM[c]}** no puede ser negativa.")
        elif v < r["min"] or v > r["max"]:
            avisos.append(f"**{NOM[c]}** = {v:g} está fuera del rango de entrenamiento "
                          f"({r['min']:g} a {r['max']:g}): el modelo estaría extrapolando.")
    return errores, avisos


def predecir(df):
    X = df[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    return modelo.predict_proba(X)[:, 1]


def contribuciones(fila):
    """Aporte de cada variable a los log-odds: coeficiente × valor transformado y estandarizado."""
    X = pd.DataFrame([fila])[COLUMNAS]
    paso_log = modelo.named_steps["transformacion_log"]
    z = modelo.named_steps["escalamiento"].transform(paso_log.transform(X))[0]
    nombres = paso_log.get_feature_names_out()
    return pd.DataFrame({"Variable": [CORTOS[n] for n in nombres],
                         "Aporte": [info["coeficientes"][n] * float(v) for n, v in zip(nombres, z)]})


# ---------------------------------------------------------------------------
# Barra lateral: escenario y tarjeta del modelo
# ---------------------------------------------------------------------------
m, ve = info["metricas_prueba"], info["validacion_espacial"]
with st.sidebar:
    st.header("🌳 Tarjeta del modelo")
    escenario = st.radio("Escenario de uso", NOMBRES_ESC,
                         help="Lo único que cambia entre escenarios es el umbral de alerta.")
    e = ESC[escenario]
    UMBRAL, pe = e["umbral"], e["prueba"]
    st.markdown(f"**Umbral de alerta:** {fmt(UMBRAL, 4 if UMBRAL != round(UMBRAL, 2) else 2)}")
    st.caption(e["descripcion"])

    st.subheader("Desempeño en prueba")
    st.caption(f"{m['registros']:,} zonas de {m['bloques_prueba']} bloques de 20 km que el modelo nunca vio "
               f"({m['positivos']} deforestadas).".replace(",", "."))
    c1, c2 = st.columns(2)
    c1.metric("Recall", fmt(pe["recall"]))
    c2.metric("Precisión", fmt(pe["precision"]))
    c1.metric("F1", fmt(pe["f1"]))
    c2.metric("Zonas señaladas", f"{fmt(pe['pct_senaladas'], 1)} %")
    st.caption(f"Encontró **{pe['tp']} de {m['positivos']}** zonas deforestadas y señaló "
               f"{pe['zonas_senaladas']:,} zonas en total.".replace(",", "."))

    with st.expander("Comparar los dos escenarios"):
        st.dataframe(pd.DataFrame({
            n: {"Umbral": fmt(v["umbral"], 4), "Recall": fmt(v["prueba"]["recall"]),
                "Precisión": fmt(v["prueba"]["precision"]), "F1": fmt(v["prueba"]["f1"]),
                "Señaladas": f"{v['prueba']['zonas_senaladas']} ({fmt(v['prueba']['pct_senaladas'], 1)} %)",
                "Detectadas": f"{v['prueba']['tp']} de {m['positivos']}"} for n, v in ESC.items()}),
            width="stretch")

    st.subheader("Validación espacial")
    st.caption(f"Bloques de 20 km, 5 folds (umbral 0,5): recall {fmt(ve['recall']['media'])} · "
               f"ROC-AUC {fmt(ve['roc_auc']['media'])} · PR-AUC {fmt(ve['pr_auc']['media'])}.")

    st.subheader("Limitaciones")
    st.markdown(f"- De cada 100 alertas, unas **{100 * pe['precision']:.0f}** son deforestación real.\n"
                "- Sirve para **priorizar el monitoreo**, no reemplaza la verificación en campo.\n"
                "- Solo se validó en la región y el periodo de los datos (Guaviare, 2000–2014).")


# ---------------------------------------------------------------------------
# Contenido principal
# ---------------------------------------------------------------------------
st.title("Riesgo de deforestación por zona")
st.markdown("Estima la probabilidad de que una zona del Guaviare (Amazonía colombiana) se deforeste, a partir de su "
            "latitud y de sus distancias a ríos, vías, resguardos indígenas y áreas protegidas. "
            "Reproduce, con datos colombianos, el objetivo de **Rosa et al. (2013)**.")
st.info(f"Escenario activo: **{escenario}** · umbral **{fmt(UMBRAL, 4)}**. Puede cambiarlo en la barra lateral.")

tab_una, tab_varias, tab_ayuda = st.tabs(["🔎 Una zona", "📄 Varias zonas (CSV)", "ℹ️ Cómo usar"])

# --- Pestaña 1: una zona -----------------------------------------------------
with tab_una:
    opciones = ["Valores típicos (medianas)"] + list(info["ejemplos"])
    elegido = st.selectbox("Cargar un ejemplo real del conjunto de prueba", opciones,
                           help="Zonas reales de bloques que el modelo no vio al entrenar.")
    if elegido in info["ejemplos"]:
        base = info["ejemplos"][elegido]["valores"]
        real = info["ejemplos"][elegido]["real"]
        st.caption(f"Valor real de esta zona: **{'deforestada' if real == 1 else 'sin deforestación'}**.")
    else:
        base = {c: info["rangos"][c]["mediana"] for c in COLUMNAS}

    with st.form("zona"):
        st.markdown("**Características de la zona** (en las unidades del conjunto de datos original)")
        cols = st.columns(3)
        fila = {}
        for i, c in enumerate(COLUMNAS):
            r = info["rangos"][c]
            fila[c] = cols[i % 3].number_input(
                NOM[c] + (f" ({UNIDAD[c]})" if c in UNIDAD else ""),
                min_value=None if c == "lat" else 0.0, value=float(base[c]), format="%.4f",
                key=f"{elegido}_{c}",
                help=f"Rango de entrenamiento: {r['min']:g} a {r['max']:g} · mediana {r['mediana']:g}")
        enviar = st.form_submit_button("Predecir riesgo", type="primary")

    if enviar:
        errores, avisos = verificar(fila)
        for err in errores:
            st.error(err)
        for a in avisos:
            st.warning(a)
        if not errores:
            p = float(predecir(pd.DataFrame([fila]))[0])
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Probabilidad de deforestación", f"{p:.1%}".replace(".", ","))
                st.progress(min(p, 1.0))
                for n, v in ESC.items():
                    alerta = p >= v["umbral"]
                    texto = (f"**{n}** (umbral {fmt(v['umbral'], 4)}): "
                             + ("⚠️ ALERTA, zona prioritaria." if alerta else "✅ sin alerta."))
                    if n == escenario:
                        (st.error if alerta else st.success)(texto)
                    else:
                        st.caption(texto)
                st.caption("El modelo se entrenó con SMOTE: use la probabilidad para ordenar zonas por riesgo y "
                           "compararla con el umbral, no como un porcentaje literal.")
            with c2:
                aportes = contribuciones(fila).sort_values("Aporte")
                aportes["Efecto"] = np.where(aportes["Aporte"] > 0, "Sube el riesgo", "Baja el riesgo")
                graf = alt.Chart(aportes).mark_bar().encode(
                    x=alt.X("Aporte:Q", title="Aporte al riesgo (log-odds)"),
                    y=alt.Y("Variable:N", sort=list(aportes["Variable"]), title=None,
                            axis=alt.Axis(labelOverlap=False, labelLimit=220, labelFontSize=13)),
                    color=alt.Color("Efecto:N", scale=alt.Scale(domain=["Sube el riesgo", "Baja el riesgo"],
                                                                range=["#D9622B", "#1F4D3A"]),
                                    legend=alt.Legend(orient="bottom")),
                    tooltip=["Variable", alt.Tooltip("Aporte:Q", format=".2f")]).properties(
                    height=280, title="¿Qué variables movieron esta predicción?")
                st.altair_chart(graf, width="stretch")
                mayor = aportes.iloc[aportes["Aporte"].abs().argmax()]
                st.caption(f"La variable que más pesó fue **{mayor['Variable']}**, que "
                           f"{'subió' if mayor['Aporte'] > 0 else 'bajó'} el riesgo de esta zona frente a una zona promedio.")

# --- Pestaña 2: varias zonas -------------------------------------------------
with tab_varias:
    st.markdown(f"Suba un CSV con estas columnas: `{', '.join(COLUMNAS)}`. "
                "Se devuelve el mismo archivo con la probabilidad, la alerta de cada escenario y el orden de prioridad.")
    plantilla = pd.DataFrame([v["valores"] for v in info["ejemplos"].values()])[COLUMNAS]
    st.download_button("Descargar plantilla de ejemplo", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_zonas.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        faltan = [c for c in COLUMNAS if c not in datos.columns]
        if faltan:
            st.error(f"Faltan estas columnas: {', '.join(faltan)}")
        else:
            datos[COLUMNAS] = datos[COLUMNAS].apply(pd.to_numeric, errors="coerce")
            revision = [verificar(f) for _, f in datos.iterrows()]
            problemas = [(i, err) for i, (errs, _) in enumerate(revision) for err in errs]
            if problemas:
                st.error(f"Hay {len(problemas)} valores inválidos (vacíos o distancias negativas). "
                         f"Primer caso, fila {problemas[0][0] + 1}: {problemas[0][1]}")
            else:
                n_avisos = sum(len(a) for _, a in revision)
                if n_avisos:
                    st.warning(f"{n_avisos} valores están fuera del rango de entrenamiento; revíselos.")
                datos["probabilidad"] = predecir(datos)
                for n, v in ESC.items():
                    datos[f"alerta_{n}"] = np.where(datos["probabilidad"] >= v["umbral"], "Sí", "No")
                datos["prioridad"] = datos["probabilidad"].rank(ascending=False, method="first").astype(int)
                c1, c2, c3 = st.columns(3)
                c1.metric("Zonas evaluadas", f"{len(datos):,}".replace(",", "."))
                for col, (n, v) in zip([c2, c3], ESC.items()):
                    col.metric(f"Alertas · {n}", f"{(datos[f'alerta_{n}'] == 'Sí').sum():,}".replace(",", "."))
                st.dataframe(datos.sort_values("prioridad"), width="stretch")
                st.download_button("Descargar resultados", datos.to_csv(index=False).encode("utf-8"),
                                   "resultados_riesgo.csv", "text/csv")

# --- Pestaña 3: ayuda --------------------------------------------------------
with tab_ayuda:
    e1, e2 = ESC[NOMBRES_ESC[0]], ESC[NOMBRES_ESC[1]]
    st.markdown(f"""
**Qué hace la aplicación.** Recibe las 5 variables de una zona, les aplica exactamente el mismo
preprocesamiento del entrenamiento (log(1 + x) en las distancias a resguardos y a áreas protegidas, y
estandarización z-score) y la regresión logística final, y devuelve la probabilidad de deforestación.

**Los dos escenarios.** El modelo es el mismo; solo cambia el umbral:
- *{NOMBRES_ESC[0]}* (umbral {fmt(e1['umbral'], 2)}): se eligió para encontrar al menos 80 de cada 100 zonas
  deforestadas. En prueba encontró {e1['prueba']['tp']} de {m['positivos']}, pero señala el
  {fmt(e1['prueba']['pct_senaladas'], 1)} % de las zonas.
- *{NOMBRES_ESC[1]}* (umbral {fmt(e2['umbral'], 4)}): señala solo el 20 % de zonas con mayor probabilidad, para
  cuando no se pueden visitar todas. En prueba encontró {e2['prueba']['tp']} de {m['positivos']}.

Ambos umbrales se eligieron con probabilidades fuera de fold del entrenamiento (validación espacial), nunca con
los datos de prueba.

**Cómo leer el resultado.**
- *Probabilidad:* sirve para ordenar las zonas de mayor a menor riesgo.
- *Alerta:* se emite si la probabilidad es mayor o igual al umbral del escenario activo.
- *Gráfico de aportes:* barras naranjas suben el riesgo y verdes lo bajan, comparado con una zona promedio.

**Coherencia con el entrenamiento.** La aplicación bloquea valores vacíos o distancias negativas, y avisa si un
valor queda fuera del rango visto en el entrenamiento, porque ahí el modelo estaría extrapolando.

**Variables que no usa.** `yloss`, `fire` y `yfire` (fuga de información), las distancias a pérdidas de 2010 a
2014, `rec2009c1` y `disedge09` (medidas después de muchas pérdidas) y la longitud (muy correlacionada con la
distancia a vías).

**Datos:** `bothgridandfire.csv` (Armenteras et al., 2017). Partición espacial: {m['bloques_prueba']} bloques de
20 km para prueba.
""")
