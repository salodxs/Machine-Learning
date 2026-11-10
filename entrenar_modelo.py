"""
Entrena el modelo final y guarda todo lo que necesita la aplicación de Streamlit.

Uso:
    python entrenar_modelo.py "bothgridandfire (1).csv"

Salidas (en la misma carpeta):
    modelo_final.joblib   -> pipeline completo (preprocesamiento + SMOTE + regresión logística)
    info_modelo.json      -> umbral, variables, rangos de entrenamiento, métricas, coeficientes y ejemplos
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuración (si el equipo cambia el modelo final en la 8.1, se ajusta aquí)
# ---------------------------------------------------------------------------
SEMILLA = 42
OBJETIVO = "def"
DISTANCIAS = ["disedge09", "Distriv", "dist_roads", "d_IR", "d_II_III"]   # log(1 + x) + z-score
CATEGORICAS = ["rec2001c1"]                                               # códigos de cobertura en 2001
# Excluidas: yloss y fire/yfire (fuga), dist_10los...dist_los14 (temporalidad desconocida),
# lat/long (describen el lugar; solo se usan para los bloques espaciales) y rec2009c1 (medida en 2009,
# después de buena parte de las pérdidas).
META_RECALL = 0.80
TAM_BLOQUE_KM = 20

NOMBRES = {
    "disedge09": "Distancia al borde del bosque en 2009",
    "Distriv": "Distancia a ríos",
    "dist_roads": "Distancia a vías",
    "d_IR": "Distancia a resguardos indígenas",
    "d_II_III": "Distancia a áreas protegidas (categorías II y III)",
    "rec2001c1": "Clase de cobertura en 2001",
}


def crear_pipeline():
    distancias = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("logaritmo", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalamiento", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preprocesamiento = ColumnTransformer([
        ("distancias", distancias, DISTANCIAS),
        ("categoricas", categoricas, CATEGORICAS),
    ], remainder="drop")
    return ImbPipeline([
        ("preprocesamiento", preprocesamiento),
        ("balanceo", SMOTE(sampling_strategy="auto", random_state=SEMILLA, k_neighbors=5)),
        ("modelo", LogisticRegression(max_iter=2000, random_state=SEMILLA)),
    ])


def main(ruta_csv):
    df = pd.read_csv(ruta_csv)
    predictores = DISTANCIAS + CATEGORICAS
    X, y = df[predictores], df[OBJETIVO].astype(int)

    # 1. Partición estratificada 70/30 (la misma de la segunda entrega)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEMILLA, stratify=y)

    # 2. Umbral con probabilidades fuera de fold del entrenamiento (sin tocar la prueba)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    prob_oof = cross_val_predict(crear_pipeline(), X_train, y_train, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
    precisiones, recalls, umbrales = precision_recall_curve(y_train, prob_oof)
    cumple = recalls[:-1] >= META_RECALL
    umbral = float(umbrales[cumple][np.argmax(precisiones[:-1][cumple])])

    # 3. Validación cruzada por bloques espaciales de 20 km (solo como referencia de generalización)
    coord = df.loc[X_train.index, ["lat", "long"]]
    grupos = (np.floor(coord["lat"] * 111 / TAM_BLOQUE_KM).astype(int).astype(str) + "_" +
              np.floor(coord["long"] * 111 / TAM_BLOQUE_KM).astype(int).astype(str))
    r_bloques = cross_validate(crear_pipeline(), X_train, y_train, groups=grupos,
                               cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA),
                               scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"}, n_jobs=-1)

    # 4. Modelo final: se ajusta con todo el entrenamiento y se evalúa una vez en prueba
    modelo = crear_pipeline().fit(X_train, y_train)
    prob_test = modelo.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()

    # 5. Coeficientes (para explicar cada predicción en la app)
    nombres_transf = list(modelo.named_steps["preprocesamiento"].get_feature_names_out())
    coefs = modelo.named_steps["modelo"].coef_.ravel()

    # 6. Rangos de entrenamiento (para verificar la coherencia de las entradas)
    rangos = {}
    for c in DISTANCIAS:
        s = X_train[c]
        rangos[c] = {"min": float(s.min()), "max": float(s.max()), "p01": float(s.quantile(.01)),
                     "p99": float(s.quantile(.99)), "mediana": float(s.median())}
    clases = sorted(int(v) for v in X_train["rec2001c1"].dropna().unique())

    # 7. Ejemplos reales del conjunto de prueba para la demostración
    prueba = X_test.copy()
    prueba["real"], prueba["prob"] = y_test.values, prob_test
    ejemplos = {
        "Zona deforestada detectada (riesgo alto)":
            prueba[(prueba.real == 1)].sort_values("prob", ascending=False).iloc[0],
        "Zona deforestada que el modelo no detectó":
            prueba[(prueba.real == 1) & (prueba.prob < umbral)].sort_values("prob").iloc[0],
        "Zona sin deforestación (riesgo bajo)":
            prueba[(prueba.real == 0)].sort_values("prob").iloc[len(prueba[prueba.real == 0]) // 20],
    }
    ejemplos = {k: {"valores": {c: (int(v[c]) if c in CATEGORICAS else float(v[c])) for c in predictores},
                    "real": int(v["real"])} for k, v in ejemplos.items()}

    info = {
        "modelo": "Regresión logística con SMOTE",
        "semilla": SEMILLA,
        "distancias": DISTANCIAS,
        "categoricas": CATEGORICAS,
        "nombres": NOMBRES,
        "umbral": umbral,
        "meta_recall": META_RECALL,
        "rangos": rangos,
        "clases_rec2001c1": clases,
        "clase_frecuente": int(X_train["rec2001c1"].mode()[0]),
        "coeficientes": dict(zip(nombres_transf, map(float, coefs))),
        "intercepto": float(modelo.named_steps["modelo"].intercept_[0]),
        "metricas_prueba": {
            "registros": int(len(y_test)), "positivos": int(y_test.sum()),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "recall": float(recall_score(y_test, pred_test)),
            "precision": float(precision_score(y_test, pred_test)),
            "roc_auc": float(roc_auc_score(y_test, prob_test)),
            "pr_auc": float(average_precision_score(y_test, prob_test)),
            "prevalencia": float(y_test.mean()),
        },
        "metricas_bloques_20km": {
            "roc_auc": float(r_bloques["test_roc_auc"].mean()), "roc_auc_sd": float(r_bloques["test_roc_auc"].std()),
            "pr_auc": float(r_bloques["test_pr_auc"].mean()), "bloques": int(grupos.nunique()),
        },
        "ejemplos": ejemplos,
        "versiones": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                      "imbalanced-learn": imblearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__},
    }

    carpeta = Path(__file__).resolve().parent
    joblib.dump(modelo, carpeta / "modelo_final.joblib")
    (carpeta / "info_modelo.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    m = info["metricas_prueba"]
    print(f"Umbral: {umbral:.3f}")
    print(f"Prueba: {m['tp']} de {m['positivos']} detectadas | recall {m['recall']:.3f} | "
          f"precisión {m['precision']:.3f} | ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}")
    b = info["metricas_bloques_20km"]
    print(f"Bloques de 20 km: ROC-AUC {b['roc_auc']:.3f} ± {b['roc_auc_sd']:.3f} | PR-AUC {b['pr_auc']:.3f}")
    print("Guardados: modelo_final.joblib, info_modelo.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bothgridandfire (1).csv")
