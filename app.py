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
DIST_KM, KM = info["distancias"], info["km_por_grado"]


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
m, ve, va = info["metricas_prueba"], info["validacion_espacial"], info["validacion_aleatoria"]
RECOMENDADO = next((n for n in NOMBRES_ESC if "recomendado" in n), NOMBRES_ESC[0])


def miles(x):
    return f"{x:,.0f}".replace(",", ".")


with st.sidebar:
    st.header("🌳 Tarjeta del modelo")
    escenario = st.radio("Escenario de uso", NOMBRES_ESC, index=NOMBRES_ESC.index(RECOMENDADO),
                         help="El modelo es el mismo; lo único que cambia entre escenarios es el umbral de alerta.")
    e = ESC[escenario]
    UMBRAL, pe = e["umbral"], e["prueba"]
    st.markdown(f"**Umbral de alerta:** {fmt(UMBRAL, 4)}")
    st.caption(e["descripcion"])

    st.subheader("¿Qué significa en campo?")
    st.caption(f"Resultado en {miles(m['registros'])} zonas de {m['bloques_prueba']} bloques de 20 km que el modelo "
               f"nunca vio ({m['positivos']} deforestadas).")
    st.markdown(f"#### {miles(pe['zonas_senaladas'])} inspecciones")
    st.caption(f"{fmt(pe['pct_senaladas'], 1)} % del territorio")
    st.markdown(f"#### {pe['tp']} de {m['positivos']} deforestadas halladas")
    st.caption(f"{fmt(100 * pe['recall'], 0)} % de la deforestación")
    st.markdown(f"- Por cada zona deforestada encontrada se revisan **~{pe['falsas_por_acierto']:.0f} zonas** que no "
                f"lo estaban.\n"
                f"- Revisando al azar el mismo número de zonas se encontrarían **~{pe['al_azar']:.0f}**: el modelo "
                f"encuentra **{fmt(pe['veces_mejor_que_azar'], 1)} veces más**.")
    with st.expander("Métricas técnicas"):
        st.markdown(f"Recall {fmt(pe['recall'])} · Precisión {fmt(pe['precision'])} · F1 {fmt(pe['f1'])}  \n"
                    f"ROC-AUC {fmt(m['roc_auc'])} · PR-AUC {fmt(m['pr_auc'])} (prevalencia {fmt(m['prevalencia'])})")

    st.subheader("¿Memoriza vecinos?")
    st.caption(f"No: se evaluó en bloques de 20 km completos. Si se mezclan zonas vecinas (validación aleatoria) el "
               f"ROC-AUC sale {fmt(va['roc_auc']['media'])}; separando por bloques baja a "
               f"{fmt(ve['roc_auc']['media'])} y en los bloques de prueba a {fmt(m['roc_auc'])}. Reportamos los "
               f"números bajos, que son los honestos.")

    st.subheader("Limitaciones")
    st.markdown("- Sirve para **priorizar** el monitoreo, no reemplaza la verificación en campo.\n"
                "- Solo se validó dentro del Guaviare y con datos de 2000–2014.")


# ---------------------------------------------------------------------------
# Contenido principal
# ---------------------------------------------------------------------------
st.title("¿Dónde vigilar primero?")
st.markdown("**Si solo podemos inspeccionar una fracción del territorio, ¿dónde concentrar la vigilancia para "
            "detectar más deforestación?** Esta herramienta estima el riesgo de deforestación de cada zona del "
            "Guaviare a partir de su latitud y de sus distancias a ríos, vías, resguardos indígenas y áreas "
            "protegidas, y la ordena para priorizar las visitas de campo.")
st.info(f"Escenario activo: **{escenario}** · umbral **{fmt(UMBRAL, 4)}**. Puede cambiarlo en la barra lateral.")

tab_esc, tab_una, tab_varias, tab_ayuda = st.tabs(
    ["📊 Escenarios", "🔎 Una zona", "📄 Varias zonas (CSV)", "ℹ️ Cómo usar"])

# --- Pestaña 0: escenarios de negocio -----------------------------------------
with tab_esc:
    st.markdown(f"Resultados en **{miles(m['registros'])} zonas** de bloques que el modelo nunca vio, con "
                f"**{m['positivos']} zonas deforestadas**. Al azar, 1 de cada {1 / m['prevalencia']:.0f} zonas "
                "revisadas estaría deforestada.")
    filas = []
    for n, v in ESC.items():
        q = v["prueba"]
        filas.append({"Escenario": n, "Umbral": fmt(v["umbral"], 4),
                      "Inspecciones": f"{miles(q['zonas_senaladas'])} ({fmt(q['pct_senaladas'], 1)} %)",
                      "Deforestadas halladas": f"{q['tp']} de {m['positivos']} ({fmt(100 * q['recall'], 0)} %)",
                      "Se escapan": q["fn"],
                      "Visitas vacías": miles(q["fp"]),
                      "Visitas vacías por acierto": f"~{q['falsas_por_acierto']:.0f}",
                      "Al azar hallaría": f"~{q['al_azar']:.0f}"})
    st.dataframe(pd.DataFrame(filas).set_index("Escenario"), width="stretch")
    st.caption("Visitas vacías: zonas inspeccionadas que no estaban deforestadas. ★ = escenario recomendado.")

    CORTO_ESC = {n: n.replace("Capacidad de inspección limitada (recomendado)", "Capacidad limitada ★") for n in ESC}
    graf_df = pd.DataFrame([{"Escenario": CORTO_ESC[n], "Tipo": t, "Zonas": v["prueba"][k]}
                            for n, v in ESC.items()
                            for t, k in [("Deforestadas halladas", "tp"), ("Visitas vacías", "fp")]])
    graf = alt.Chart(graf_df).mark_bar().encode(
        y=alt.Y("Escenario:N", sort=list(CORTO_ESC.values()), title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=13, labelOverlap=False)),
        x=alt.X("Zonas:Q", title="Zonas inspeccionadas"),
        color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Deforestadas halladas", "Visitas vacías"],
                                                  range=["#D9622B", "#C9CFC9"]), legend=alt.Legend(orient="bottom")),
        order=alt.Order("Tipo:N", sort="ascending"),
        tooltip=["Escenario", "Tipo", "Zonas"]).properties(height=240, title="¿Cuánto cuesta cada escenario en inspecciones?")
    st.altair_chart(graf, width="stretch")

    r = ESC[RECOMENDADO]
    st.success(f"**Recomendación: {RECOMENDADO}** (umbral {fmt(r['umbral'], 4)}). Revisando el "
               f"{fmt(r['prueba']['pct_senaladas'], 0)} % del territorio se encuentra el "
               f"{fmt(100 * r['prueba']['recall'], 0)} % de la deforestación, "
               f"{fmt(r['prueba']['veces_mejor_que_azar'], 1)} veces más que revisando al azar. En validación espacial "
               "fue el punto más eficiente: con menos inspecciones se pierde mucha detección y con más, cada "
               "inspección extra rinde menos.")

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
        base = {c: (info["rangos_km"][c]["mediana"] / KM if c in DIST_KM else info["rangos"][c]["mediana"])
                for c in COLUMNAS}

    with st.form("zona"):
        st.markdown("**Características de la zona** · latitud en grados decimales y distancias en **kilómetros**")
        cols = st.columns(3)
        fila, avisos_km = {}, []
        for i, c in enumerate(COLUMNAS):
            if c in DIST_KM:
                r = info["rangos_km"][c]
                km = cols[i % 3].number_input(
                    f"{NOM[c]} (km)", min_value=0.0, value=round(float(base[c]) * KM, 2), step=1.0, format="%.2f",
                    key=f"{elegido}_{c}",
                    help=f"Rango de entrenamiento: {r['min']:.1f} a {r['max']:.1f} km · mediana {r['mediana']:.1f} km")
                fila[c] = km / KM          # el modelo trabaja en grados, como el CSV original
                if km < r["min"] or km > r["max"]:
                    avisos_km.append(f"**{NOM[c]}** = {km:g} km está fuera del rango de entrenamiento "
                                     f"({r['min']:.1f} a {r['max']:.1f} km): el modelo estaría extrapolando.")
            else:
                r = info["rangos"][c]
                fila[c] = cols[i % 3].number_input(
                    f"{NOM[c]} (grados decimales)", value=float(base[c]), step=0.01, format="%.4f",
                    key=f"{elegido}_{c}",
                    help=f"Rango de entrenamiento: {r['min']:.3f} a {r['max']:.3f} · mediana {r['mediana']:.3f}")
        enviar = st.form_submit_button("Predecir riesgo", type="primary")
    st.caption(f"Las distancias se convierten a grados (1 grado ≈ {KM:g} km), que es la unidad del conjunto de datos. "
               "Si escribe distancias medidas hoy, el resultado es el riesgo estimado con las condiciones actuales: "
               "el modelo aprendió con datos de 2000 a 2014 y no se ha validado en años recientes.")

    if enviar:
        errores, avisos = verificar(fila)
        avisos = [a for a in avisos if not any(NOM[c] in a for c in DIST_KM)] + avisos_km
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
    st.markdown(f"Suba un CSV con estas columnas: `{', '.join(COLUMNAS)}`, **en las mismas unidades del conjunto de "
                "datos original** (distancias en grados, no en km), para poder evaluar filas del CSV directamente. "
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
                cols_m = st.columns(len(ESC) + 1)
                cols_m[0].metric("Zonas evaluadas", f"{len(datos):,}".replace(",", "."))
                for col, (n, v) in zip(cols_m[1:], ESC.items()):
                    col.metric(f"Alertas · {n}", f"{(datos[f'alerta_{n}'] == 'Sí').sum():,}".replace(",", "."))
                st.dataframe(datos.sort_values("prioridad"), width="stretch")
                st.download_button("Descargar resultados", datos.to_csv(index=False).encode("utf-8"),
                                   "resultados_riesgo.csv", "text/csv")

# --- Pestaña 3: ayuda --------------------------------------------------------
with tab_ayuda:
    lista_esc = "\n".join(f"- *{n}* (umbral {fmt(v['umbral'], 4)}): {v['descripcion']} En prueba encontró "
                          f"{v['prueba']['tp']} de {m['positivos']} revisando el {fmt(v['prueba']['pct_senaladas'], 1)} % de las zonas."
                          for n, v in ESC.items())
    st.markdown(f"""
**Qué hace la aplicación.** Recibe las 5 variables de una zona (en la pestaña *Una zona*, las distancias en km;
la app las pasa a grados, la unidad del conjunto de datos, con 1 grado ≈ {KM:g} km), les aplica exactamente el mismo
preprocesamiento del entrenamiento (log(1 + x) en las distancias a resguardos y a áreas protegidas, y
estandarización z-score) y la regresión logística final, y devuelve la probabilidad de deforestación.

**Los tres escenarios.** El modelo es el mismo; solo cambia el umbral:
{lista_esc}

Los umbrales se eligieron con probabilidades fuera de fold del entrenamiento (validación espacial), nunca con
los datos de prueba.

**Cómo leer el resultado.**
- *Probabilidad:* sirve para ordenar las zonas de mayor a menor riesgo.
- *Alerta:* se emite si la probabilidad es mayor o igual al umbral del escenario activo.
- *Gráfico de aportes:* barras naranjas suben el riesgo y verdes lo bajan, comparado con una zona promedio.

**Coherencia con el entrenamiento.** La aplicación bloquea valores vacíos o distancias negativas, y avisa si un
valor queda fuera del rango visto en el entrenamiento, porque ahí el modelo estaría extrapolando.

**Sobre el año.** El modelo aprendió con datos de 2000 a 2014. Si se escriben distancias medidas hoy, el resultado
es una estimación del riesgo con las condiciones actuales, pero no se validó en años recientes.

**Variables que no usa.** `yloss`, `fire` y `yfire` (fuga de información), las distancias a pérdidas de 2010 a
2014, `rec2009c1` y `disedge09` (medidas después de muchas pérdidas) y la longitud (muy correlacionada con la
distancia a vías).

**Datos:** `bothgridandfire.csv` (Armenteras et al., 2017). Partición espacial: {m['bloques_prueba']} bloques de
20 km para prueba.
""")
