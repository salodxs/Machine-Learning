# Riesgo de deforestación por zona — Despliegue básico (8.3)

Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia

Aplicación en **Streamlit** que recibe las características de una zona de la Amazonía colombiana, les aplica el mismo
preprocesamiento del entrenamiento y la regresión logística final, y devuelve la **probabilidad de deforestación** y una
**alerta** si supera el umbral.

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación de Streamlit |
| `entrenar_modelo.py` | Entrena el modelo final y genera `modelo_final.joblib` e `info_modelo.json` |
| `modelo_final.joblib` | Pipeline completo guardado: imputación, log(1 + x), estandarización, codificación, SMOTE y regresión logística |
| `info_modelo.json` | Umbral, variables, rangos de entrenamiento, métricas, coeficientes y ejemplos |
| `bothgridandfire (1).csv` | Datos originales (Armenteras et al., 2017; Dryad, doi:10.5061/dryad.1925k) |
| `requirements.txt` | Versiones exactas de las librerías |

## Cómo ejecutarla

1. Instalar Python 3.11 o superior.
2. Abrir una terminal en esta carpeta e instalar las librerías:
   ```bash
   pip install -r requirements.txt
   ```
3. (Recomendado) Volver a entrenar el modelo para que quede guardado con las librerías instaladas en su equipo:
   ```bash
   python entrenar_modelo.py "bothgridandfire (1).csv"
   ```
   Debe imprimir el umbral (0,451) y las métricas de prueba.
4. Abrir la aplicación:
   ```bash
   streamlit run app.py
   ```
   Se abre en el navegador, en `http://localhost:8501`.

> **Importante:** un modelo guardado con `joblib` solo se puede abrir con la misma versión de scikit-learn con que se
> guardó. Si aparece un error al cargar `modelo_final.joblib`, ejecute el paso 3 para regenerarlo.

## Cómo usarla

- **Una zona:** escriba las distancias y la clase de cobertura (en las unidades del conjunto de datos original), o
  cargue uno de los tres ejemplos reales del conjunto de prueba. Pulse **Predecir riesgo**.
- **Varias zonas (CSV):** descargue la plantilla, llénela con sus zonas, súbala y descargue los resultados con la
  probabilidad y la alerta de cada fila.
- **Cómo usar:** explicación de la aplicación dentro de la misma app.

## Cómo leer el resultado

- **Probabilidad:** sirve para ordenar zonas de mayor a menor riesgo. Como el modelo se entrenó con SMOTE, no debe
  leerse como un porcentaje literal.
- **Alerta:** se emite si la probabilidad es mayor o igual a **0,451**, umbral elegido con probabilidades fuera de fold
  del entrenamiento para encontrar al menos 80 de cada 100 zonas deforestadas.
- **Gráfico de aportes:** cuánto empuja cada variable el riesgo de esa zona frente a una zona promedio
  (coeficiente × valor estandarizado). Naranja sube el riesgo; verde lo baja.

## Coherencia entre entrenamiento y uso

- La app usa **el mismo pipeline guardado** del entrenamiento: las entradas pasan por las mismas transformaciones.
- Las columnas de entrada son exactamente las del entrenamiento.
- La app **bloquea** distancias negativas y **advierte** cuando un valor está vacío (se imputa con la mediana), queda
  fuera del rango visto en el entrenamiento (el modelo estaría extrapolando) o es una clase de cobertura nueva.

## Modelo desplegado

- **Regresión logística** con SMOTE, semilla 42, partición estratificada 70/30.
- **Predictores (6):** `disedge09`, `Distriv`, `dist_roads`, `d_IR`, `d_II_III` y `rec2001c1` (categórica).
- **Excluidos:** `yloss`, `fire`, `yfire` (fuga de información); `dist_10los` a `dist_los14` (temporalidad desconocida);
  `rec2009c1` (medida después de muchas pérdidas); `lat` y `long` (describen el lugar; se usan solo para la validación
  por bloques espaciales).
- **Prueba (5.028 zonas, 208 deforestadas):** recall 0,832 · precisión 0,109 · ROC-AUC 0,846 · PR-AUC 0,221.
- **Validación por bloques espaciales de 20 km:** ROC-AUC 0,815.

## Limitaciones

- De cada 100 alertas, unas 11 son deforestación real: sirve para priorizar el monitoreo, no reemplaza la verificación
  en campo.
- Falla más en zonas lejos de las vías y sin incendio.
- No se validó en años posteriores al periodo de los datos.
