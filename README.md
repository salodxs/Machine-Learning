# Riesgo de deforestación por zona — Despliegue básico (8.3)

Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia

Aplicación en **Streamlit** que recibe las 5 variables de una zona del Guaviare (Amazonía colombiana), les aplica el
mismo preprocesamiento del entrenamiento y la regresión logística final (8.1), y devuelve la **probabilidad de
deforestación** y una **alerta** según el escenario de uso elegido (8.2).

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación de Streamlit |
| `entrenar_modelo.py` | Reproduce el modelo final del notebook del equipo y genera `modelo_final.joblib` e `info_modelo.json` |
| `modelo_final.joblib` | Pipeline completo guardado: log(1 + x), estandarización z-score, SMOTE y regresión logística |
| `info_modelo.json` | Umbrales de los dos escenarios, variables, rangos de entrenamiento, métricas, coeficientes y ejemplos |
| `bothgridandfire.csv` | Datos originales (Armenteras et al., 2017; Dryad, doi:10.5061/dryad.1925k) |
| `requirements.txt` | Librerías (scikit-learn 1.6.1, la misma de Google Colab con que se entrenó) |

## Cómo ejecutarla

1. Instalar Python 3.11 o superior.
2. Abrir una terminal en esta carpeta e instalar las librerías:
   ```bash
   pip install -r requirements.txt
   ```
3. (Opcional) Volver a entrenar el modelo:
   ```bash
   python entrenar_modelo.py bothgridandfire.csv
   ```
   Debe imprimir: entrenamiento 13.452 zonas, prueba 3.307 zonas (95 deforestadas), umbrales 0,46 y 0,6423.
4. Abrir la aplicación:
   ```bash
   streamlit run app.py
   ```
   Se abre en el navegador, en `http://localhost:8501`.

> **Importante:** use scikit-learn **1.6.1**. Con otra versión, la partición espacial `StratifiedGroupKFold` reparte
> los bloques de otra forma y los números cambian; además, un modelo guardado con `joblib` solo se abre bien con la
> misma versión con que se guardó.

## Cómo usarla

- **Barra lateral:** elija el escenario (*Máxima detección* o *Capacidad de inspección limitada*) y vea sus métricas.
- **Una zona:** escriba la latitud y las cuatro distancias, o cargue uno de los tres ejemplos reales del conjunto de
  prueba. Pulse **Predecir riesgo**. La app muestra la alerta de los dos escenarios y resalta el activo.
- **Varias zonas (CSV):** descargue la plantilla, llénela, súbala y descargue los resultados con la probabilidad, la
  alerta de cada escenario y el orden de prioridad.
- **Cómo usar:** explicación dentro de la misma app.

## Modelo desplegado (igual al notebook del equipo)

- **Regresión logística** con SMOTE dentro del pipeline (semilla 42).
- **Predictores (5, Modelo B):** `lat`, `Distriv`, `dist_roads`, `d_IR`, `d_II_III`. A `d_IR` y `d_II_III` se les
  aplica log(1 + x); luego todas se estandarizan. Se excluyó `long` por su alta correlación con `dist_roads`.
- **Partición espacial:** coordenadas proyectadas a metros (EPSG:3116), bloques de 20 km (48 bloques); prueba = primer
  fold de `StratifiedGroupKFold` (9 bloques, 3.307 zonas, 95 deforestadas).
- **Validación espacial (5 folds, umbral 0,5):** recall 0,757 · precisión 0,129 · F1 0,215 · ROC-AUC 0,781 · PR-AUC 0,157.

| Escenario | Umbral | Recall | Precisión | F1 | Zonas señaladas | Detectadas |
|---|---|---|---|---|---|---|
| Máxima detección | 0,46 | 0,811 | 0,057 | 0,107 | 1.350 (40,8 %) | 77 de 95 |
| Capacidad de inspección limitada | 0,6423 | 0,484 | 0,063 | 0,111 | 731 (22,1 %) | 46 de 95 |

Los dos umbrales se eligieron con probabilidades fuera de fold del entrenamiento (validación espacial), no con la prueba:
el primero es el que logra recall ≥ 0,80 con la mayor precisión; el segundo señala el 20 % de zonas con mayor probabilidad.

**Coeficientes (log-odds por desviación estándar):** distancia a vías −1,078 · distancia a resguardos 0,842 ·
distancia a ríos −0,634 · distancia a áreas protegidas −0,370 · latitud −0,166.

## Cómo leer el resultado

- **Probabilidad:** sirve para ordenar zonas de mayor a menor riesgo. Como el modelo se entrenó con SMOTE, no debe
  leerse como un porcentaje literal.
- **Alerta:** se emite si la probabilidad es mayor o igual al umbral del escenario.
- **Gráfico de aportes:** cuánto empuja cada variable el riesgo de esa zona frente a una zona promedio
  (coeficiente × valor estandarizado). Naranja sube el riesgo; verde lo baja.

## Coherencia entre entrenamiento y uso

- La app usa **el mismo pipeline guardado**: las entradas pasan por las mismas transformaciones del entrenamiento.
- Las columnas de entrada son exactamente las del entrenamiento.
- La app **bloquea** valores vacíos y distancias negativas, y **advierte** cuando un valor queda fuera del rango visto
  en el entrenamiento (el modelo estaría extrapolando).

## Limitaciones

- La precisión es baja (unas 6 de cada 100 alertas son deforestación real): sirve para priorizar el monitoreo, no
  reemplaza la verificación en campo.
- Solo se validó en la región y el periodo de los datos (Guaviare, 2000–2014).
