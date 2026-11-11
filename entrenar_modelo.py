"""
Entrena el modelo final del equipo (8.1) y guarda lo que necesita la aplicación de Streamlit.
Reproduce exactamente el notebook del equipo: regresión logística con 5 predictores (Modelo B),
partición espacial por bloques de 20 km y dos escenarios de umbral.

Uso:
    python entrenar_modelo.py "bothgridandfire.csv"

Salidas (en la misma carpeta):
    modelo_final.joblib   -> pipeline completo (log + z-score + SMOTE + regresión logística)
    info_modelo.json      -> umbrales, variables, rangos, métricas, coeficientes y ejemplos
"""
import json
import sys
import warnings
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pyproj import Transformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore", message="Unknown solver options")  # aviso inofensivo de SciPy reciente

# ---------------------------------------------------------------------------
# Configuración (igual al notebook del equipo)
# ---------------------------------------------------------------------------
OBJETIVO = "def"
PREDICTORES = ["lat", "Distriv", "dist_roads", "d_IR", "d_II_III"]   # Modelo B (sin long)
VARIABLES_LOG = ["d_IR", "d_II_III"]                                  # log(1 + x)
TAM_BLOQUE_M = 20_000                                                 # bloques de 20 km
SEMILLA_PARTICION = 42                                                # partición entrenamiento/prueba
SEMILLA_CV = 40                                                       # validación espacial definitiva
META_RECALL = 0.80                                                    # escenario 1
CAPACIDAD = 0.20                                                      # escenario 2: 20 % de zonas (recomendado)
CAPACIDAD_MINIMA = 0.10                                               # escenario 3: presupuesto mínimo, 10 %
DISTANCIAS = ["Distriv", "dist_roads", "d_IR", "d_II_III"]            # en grados en el CSV
KM_POR_GRADO = 111.0                                                  # 1 grado ≈ 111 km cerca del ecuador

NOMBRES = {
    "lat": "Latitud",
    "Distriv": "Distancia a ríos",
    "dist_roads": "Distancia a vías",
    "d_IR": "Distancia a resguardos indígenas",
    "d_II_III": "Distancia a áreas protegidas (categorías II y III)",
}


def crear_modelo():
    return ImbPipeline([
        ("transformacion_log", ColumnTransformer(
            transformers=[("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), VARIABLES_LOG)],
            remainder="passthrough", verbose_feature_names_out=False)),
        ("escalamiento", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clasificador", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def evaluar(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"umbral": float(umbral), "recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "zonas_senaladas": int(tp + fp), "pct_senaladas": float((tp + fp) / len(y) * 100),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            # Traducción a recursos
            "falsas_por_acierto": float(fp / tp) if tp else None,          # zonas sanas revisadas por cada deforestada hallada
            "al_azar": float((tp + fp) * y.mean()),                         # deforestadas esperadas revisando al azar
            "veces_mejor_que_azar": float((tp / y.sum()) / ((tp + fp) / len(y))) if tp + fp else None}


def main(ruta_csv):
    df = pd.read_csv(ruta_csv)
    X, y = df[PREDICTORES], df[OBJETIVO].astype(int)

    # 1. Bloques de 20 km con coordenadas proyectadas a metros (EPSG:3116, MAGNA-SIRGAS Bogotá)
    transformador = Transformer.from_crs("EPSG:4326", "EPSG:3116", always_xy=True)
    x_m, y_m = transformador.transform(df["long"].values, df["lat"].values)
    grupos = pd.Series(np.floor(x_m / TAM_BLOQUE_M).astype(int).astype(str) + "_" +
                       np.floor(y_m / TAM_BLOQUE_M).astype(int).astype(str), index=df.index)

    # 2. Partición espacial: el primer fold de StratifiedGroupKFold es la prueba
    separador = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA_PARTICION)
    train_idx, test_idx = next(separador.split(X, y, groups=grupos))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    grupos_train = grupos.iloc[train_idx]

    # 3. Probabilidades fuera de fold con validación espacial (semilla 40)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA_CV)
    prob_oof = np.zeros(len(X_train))
    folds = []
    for tr, va in cv.split(X_train, y_train, groups=grupos_train):
        p_va = crear_modelo().fit(X_train.iloc[tr], y_train.iloc[tr]).predict_proba(X_train.iloc[va])[:, 1]
        prob_oof[va] = p_va
        y_va, pred_va = y_train.iloc[va], (p_va >= 0.5).astype(int)
        folds.append({"recall": recall_score(y_va, pred_va), "precision": precision_score(y_va, pred_va, zero_division=0),
                      "f1": f1_score(y_va, pred_va, zero_division=0), "roc_auc": roc_auc_score(y_va, p_va),
                      "pr_auc": average_precision_score(y_va, p_va)})
    folds = pd.DataFrame(folds)
    validacion_espacial = {m: {"media": float(folds[m].mean()), "desv": float(folds[m].std())} for m in folds}

    # Comparación: validación aleatoria (mezcla zonas vecinas entre entrenamiento y validación)
    auc_aleatoria = []
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA_CV).split(X_train, y_train):
        p_va = crear_modelo().fit(X_train.iloc[tr], y_train.iloc[tr]).predict_proba(X_train.iloc[va])[:, 1]
        auc_aleatoria.append(roc_auc_score(y_train.iloc[va], p_va))
    validacion_aleatoria = {"roc_auc": {"media": float(np.mean(auc_aleatoria)), "desv": float(np.std(auc_aleatoria, ddof=1))}}

    # 4. Escenario 1: recall >= 0,80 con la mayor precisión (rejilla de 0,01 a 0,99)
    filas = []
    for u in np.arange(0.01, 1.00, 0.01):
        pred = (prob_oof >= u).astype(int)
        filas.append((u, recall_score(y_train, pred, zero_division=0), precision_score(y_train, pred, zero_division=0)))
    tabla = pd.DataFrame(filas, columns=["umbral", "recall", "precision"])
    umbral_1 = float(round(tabla[tabla.recall >= META_RECALL].sort_values("precision", ascending=False).iloc[0].umbral, 2))

    # 5. Escenario 2: señalar aproximadamente el 20 % de las zonas con mayor probabilidad
    umbral_2 = float(round(1 - np.quantile(1 - prob_oof, CAPACIDAD), 4))
    umbral_3 = float(round(1 - np.quantile(1 - prob_oof, CAPACIDAD_MINIMA), 4))

    # 6. Modelo final: se ajusta con todo el entrenamiento y se evalúa una vez en prueba
    modelo = crear_modelo().fit(X_train, y_train)
    prob_test = modelo.predict_proba(X_test)[:, 1]

    nombres_transf = list(modelo.named_steps["transformacion_log"].get_feature_names_out())
    coefs = modelo.named_steps["clasificador"].coef_.ravel()

    rangos = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                  "mediana": float(X_train[c].median())} for c in PREDICTORES}
    # Rangos en km (1 grado ≈ 111 km), solo con las filas cuyas distancias están en grados
    en_grados = X_train[(X_train[VARIABLES_LOG] <= 2).all(axis=1)]
    rangos_km = {c: {"min": float(en_grados[c].min() * KM_POR_GRADO), "max": float(en_grados[c].max() * KM_POR_GRADO),
                     "mediana": float(en_grados[c].median() * KM_POR_GRADO)} for c in DISTANCIAS}

    # 7. Ejemplos reales del conjunto de prueba para la demostración
    prueba = X_test.copy()
    prueba["real"], prueba["prob"] = y_test.values, prob_test
    # Solo zonas con distancias en grados (algunas filas del CSV traen d_IR y d_II_III en metros)
    prueba = prueba[(prueba[VARIABLES_LOG] <= 2).all(axis=1)]
    pos, neg = prueba[prueba.real == 1], prueba[prueba.real == 0]
    ejemplos = {
        "Zona deforestada con riesgo alto": pos.sort_values("prob", ascending=False).iloc[2],
        "Zona deforestada con riesgo intermedio": pos.iloc[(pos.prob - (umbral_1 + umbral_2) / 2).abs().argsort().iloc[0]],
        "Zona sin deforestación (riesgo bajo)": neg.sort_values("prob").iloc[len(neg) // 10],
    }
    ejemplos = {k: {"valores": {c: float(v[c]) for c in PREDICTORES}, "real": int(v["real"])}
                for k, v in ejemplos.items()}

    info = {
        "modelo": "Regresión logística con SMOTE (Modelo B, 5 predictores)",
        "predictores": PREDICTORES,
        "variables_log": VARIABLES_LOG,
        "nombres": NOMBRES,
        "escenarios": {
            "Máxima detección": {
                "umbral": umbral_1,
                "descripcion": "Prioriza encontrar la mayor cantidad de zonas deforestadas (recall ≥ 0,80 en validación espacial).",
                "entrenamiento_oof": evaluar(y_train, prob_oof, umbral_1),
                "prueba": evaluar(y_test, prob_test, umbral_1)},
            "Capacidad de inspección limitada (recomendado)": {
                "umbral": umbral_2,
                "descripcion": "Señala el 20 % de zonas con mayor probabilidad. Es el punto más eficiente en validación "
                               "espacial: más inspecciones casi no mejoran la eficiencia y menos pierden mucha detección.",
                "entrenamiento_oof": evaluar(y_train, prob_oof, umbral_2),
                "prueba": evaluar(y_test, prob_test, umbral_2)},
            "Presupuesto mínimo": {
                "umbral": umbral_3,
                "descripcion": "Señala solo el 10 % de zonas con mayor probabilidad, para reducir al máximo las "
                               "falsas alarmas cuando el presupuesto es muy bajo.",
                "entrenamiento_oof": evaluar(y_train, prob_oof, umbral_3),
                "prueba": evaluar(y_test, prob_test, umbral_3)},
        },
        "validacion_espacial": validacion_espacial,
        "validacion_aleatoria": validacion_aleatoria,
        "rangos": rangos,
        "distancias": DISTANCIAS,
        "km_por_grado": KM_POR_GRADO,
        "rangos_km": rangos_km,
        "coeficientes": dict(zip(nombres_transf, map(float, coefs))),
        "metricas_prueba": {"registros": int(len(y_test)), "positivos": int(y_test.sum()),
                            "bloques_prueba": int(grupos.iloc[test_idx].nunique()),
                            "roc_auc": float(roc_auc_score(y_test, prob_test)),
                            "pr_auc": float(average_precision_score(y_test, prob_test)),
                            "prevalencia": float(y_test.mean())},
        "ejemplos": ejemplos,
        "versiones": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                      "imbalanced-learn": imblearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }

    carpeta = Path(__file__).resolve().parent
    joblib.dump(modelo, carpeta / "modelo_final.joblib")
    (carpeta / "info_modelo.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Entrenamiento: {len(y_train)} zonas | Prueba: {len(y_test)} zonas ({int(y_test.sum())} deforestadas)")
    for nombre, e in info["escenarios"].items():
        p = e["prueba"]
        print(f"{nombre}: umbral {e['umbral']} | recall {p['recall']:.4f} | precisión {p['precision']:.4f} | "
              f"señaladas {p['zonas_senaladas']} ({p['pct_senaladas']:.1f} %) | TP {p['tp']} FN {p['fn']} FP {p['fp']} | "
              f"falsas por acierto {p['falsas_por_acierto']:.1f} | {p['veces_mejor_que_azar']:.2f} veces mejor que al azar")
    print("AUC validación aleatoria:", round(validacion_aleatoria["roc_auc"]["media"], 4))
    print("Validación espacial (umbral 0,5):", {k: round(v["media"], 4) for k, v in validacion_espacial.items()})
    m = info["metricas_prueba"]
    print(f"ROC-AUC prueba {m['roc_auc']:.4f} | PR-AUC prueba {m['pr_auc']:.4f}")
    print("Coeficientes:", {k: round(v, 4) for k, v in info["coeficientes"].items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bothgridandfire.csv")
