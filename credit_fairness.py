#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de IA avanzada para riesgo crediticio con auditoría de equidad (Fairness)
================================================================================
Unidad 2 - Aprendizaje Basado en Retos (ABR)

Problema     : predecir si un solicitante de crédito será "mal pagador" (clase 1 = bad)
Dataset      : Statlog German Credit (OpenML 'credit-g', v1) - 1000 solicitudes, 20 atributos
Modelos      : Regresión Logística (línea base) y XGBoost (modelo avanzado, hiperparámetros afinados)
Particiones  : Train 60 % / Validation 20 % / Test 20 % (estratificadas por clase y grupo etario)
Fairness     : auditoría por edad (<25 vs >=25) y sexo; mitigación con
               (a) Equidad por desconocimiento, (b) Reweighing (Kamiran & Calders, 2012),
               (c) Post-procesamiento ThresholdOptimizer / Equalized Odds (Hardt et al., 2016)

Uso:  python credit_fairness.py [--out outputs] [--seed 42] [--n-iter 60]
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, false_positive_rate,
                               selection_rate, true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# 1. Datos
# --------------------------------------------------------------------------- #
def load_data(cache_dir: Path) -> pd.DataFrame:
    """Descarga German Credit desde OpenML y deriva los atributos protegidos."""
    raw = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(cache_dir))
    df = raw.frame.copy()
    df["target"] = (df["class"] == "bad").astype(int)  # 1 = incumplimiento (resultado adverso)
    df = df.drop(columns="class")
    # Atributos protegidos derivados
    df["sex"] = np.where(df["personal_status"].str.contains("female"), "female", "male")
    df["age_group"] = np.where(df["age"] < 25, "young(<25)", "adult(>=25)")
    return df


def prepare_columns(df: pd.DataFrame, drop_sensitive: bool):
    """Devuelve listas de columnas numéricas y categóricas."""
    feats = df.drop(columns=["target", "sex", "age_group"]).copy()
    if drop_sensitive:  # equidad por desconocimiento: elimina age y personal_status (proxy de sexo)
        feats = feats.drop(columns=["age", "personal_status"])
    num = feats.select_dtypes(include=["number"]).columns.tolist()
    cat = feats.select_dtypes(exclude=["number"]).columns.tolist()
    return feats.columns.tolist(), num, cat


def build_preprocessor(num, cat) -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


class Float64Proba(BaseEstimator, ClassifierMixin):
    """Envoltorio: convierte predict_proba a float64 (compatibilidad fairlearn/pandas con XGBoost float32)."""
    def __init__(self, est):
        self.est = est
        self.classes_ = est.classes_
    def fit(self, *a, **k):
        return self
    def predict_proba(self, X):
        return self.est.predict_proba(X).astype("float64")
    def predict(self, X):
        return self.est.predict(X)


# --------------------------------------------------------------------------- #
# 2. Métricas
# --------------------------------------------------------------------------- #
def clf_metrics(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, proba),
                balanced_acc=balanced_accuracy_score(y, pred), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def best_threshold(y, proba) -> float:
    """Umbral que maximiza F1 en el conjunto de VALIDACIÓN (nunca en test)."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([f1_score(y, (proba >= t).astype(int)) for t in grid])])


def fairness_report(y, pred, sens) -> dict:
    """Métricas de equidad. Resultado favorable = crédito aprobado (pred = 0)."""
    approve = 1 - pred
    mf = MetricFrame(metrics={"approval_rate": lambda yt, yp: 1 - np.mean(yp),
                              "TPR": true_positive_rate, "FPR": false_positive_rate,
                              "accuracy": accuracy_score},
                     y_true=y, y_pred=pred, sensitive_features=sens)
    rates = mf.by_group["approval_rate"]
    return dict(
        by_group=mf.by_group.round(4).to_dict(orient="index"),
        approval_rate_ratio_DI=float(rates.min() / rates.max()),            # Disparate Impact (regla 80 %)
        demographic_parity_diff=float(demographic_parity_difference(y, pred, sensitive_features=sens)),
        equalized_odds_diff=float(equalized_odds_difference(y, pred, sensitive_features=sens)),
        tpr_gap=float(mf.difference()["TPR"]), fpr_gap=float(mf.difference()["FPR"]),
    )


def bootstrap_ci(y, pred, sens, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx_all = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx_all, len(y), replace=True)
        if len(set(sens[i])) < 2:
            continue
        vals.append(fn(y[i], pred[i], sens[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reweighing_weights(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Kamiran & Calders (2012): w(g,y) = P(g)·P(y) / P(g,y)."""
    w = np.ones(len(y))
    for gv in np.unique(g):
        for yv in np.unique(y):
            m = (g == gv) & (y == yv)
            if m.any():
                w[m] = (np.mean(g == gv) * np.mean(y == yv)) / np.mean(m)
    return w


# --------------------------------------------------------------------------- #
# 3. Programa principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=60, help="configuraciones de búsqueda aleatoria")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    np.random.seed(seed)

    df = load_data(out / "_cache")
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas | tasa de impago = {df.target.mean():.3f}")
    print("Valores faltantes totales:", int(df.isna().sum().sum()))

    # ---------- EDA de sesgo en los datos ----------
    eda = {}
    for a in ["age_group", "sex"]:
        t = df.groupby(a).agg(n=("target", "size"), bad_rate=("target", "mean")).round(4)
        eda[a] = t.to_dict(orient="index")
        print(f"\nTasa base de impago por {a}:\n{t}")

    # ---------- Partición Train / Val / Test (60/20/20) ----------
    strat = df["target"].astype(str) + "_" + df["age_group"]
    idx_tr, idx_tmp = train_test_split(df.index, test_size=0.40, stratify=strat, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=strat.loc[idx_tmp], random_state=seed)
    tr, va, te = df.loc[idx_tr], df.loc[idx_va], df.loc[idx_te]
    print(f"\nPartición -> train={len(tr)}, val={len(va)}, test={len(te)}")
    split_info = {k: dict(n=len(d), bad_rate=float(d.target.mean()),
                          young_share=float((d.age_group == 'young(<25)').mean()))
                  for k, d in dict(train=tr, validation=va, test=te).items()}

    ytr, yva, yte = tr.target.values, va.target.values, te.target.values

    # ---------- Búsqueda de hiperparámetros (entrena en TRAIN, selecciona en VALIDATION) ----------
    cols_all, num_all, cat_all = prepare_columns(df, drop_sensitive=False)
    cols_ns, num_ns, cat_ns = prepare_columns(df, drop_sensitive=True)

    def make_xgb(params):
        return XGBClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                             learning_rate=params["learning_rate"], subsample=params["subsample"],
                             colsample_bytree=params["colsample_bytree"], min_child_weight=params["min_child_weight"],
                             reg_lambda=params["reg_lambda"], gamma=params["gamma"],
                             eval_metric="logloss", random_state=seed, n_jobs=2, tree_method="hist")

    space = dict(n_estimators=[100, 200, 300, 500], max_depth=[2, 3, 4, 5, 6],
                 learning_rate=[0.01, 0.03, 0.05, 0.1, 0.2], subsample=[0.6, 0.8, 1.0],
                 colsample_bytree=[0.5, 0.7, 0.9, 1.0], min_child_weight=[1, 3, 5, 10],
                 reg_lambda=[0.5, 1, 5, 10], gamma=[0, 0.5, 1, 3])
    trials = []
    best = (-1, None)
    pre_tmp = build_preprocessor(num_all, cat_all).fit(tr[cols_all])
    Xtr_t, Xva_t = pre_tmp.transform(tr[cols_all]), pre_tmp.transform(va[cols_all])
    for p in ParameterSampler(space, n_iter=args.n_iter, random_state=seed):
        m = make_xgb(p).fit(Xtr_t, ytr)
        auc_v = roc_auc_score(yva, m.predict_proba(Xva_t)[:, 1])
        trials.append({**p, "val_auc": auc_v})
        if auc_v > best[0]:
            best = (auc_v, p)
    best_params = best[1]
    pd.DataFrame(trials).sort_values("val_auc", ascending=False).to_csv(out / "hyperparameter_search.csv", index=False)
    print(f"\nMejores hiperparámetros XGBoost (AUC val = {best[0]:.4f}): {best_params}")

    # Regresión Logística: búsqueda sobre C
    best_lr = (-1, None)
    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        lr = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xtr_t, ytr)
        a = roc_auc_score(yva, lr.predict_proba(Xva_t)[:, 1])
        if a > best_lr[0]:
            best_lr = (a, C)
    print(f"Mejor C regresión logística: {best_lr[1]} (AUC val = {best_lr[0]:.4f})")

    # ---------- Definición de los sistemas a comparar ----------
    def pipe(clf, num, cat):
        return Pipeline([("pre", build_preprocessor(num, cat)), ("clf", clf)])

    systems = {}
    # M0: línea base logística (con atributos sensibles)
    m0 = pipe(LogisticRegression(C=best_lr[1], max_iter=2000, random_state=seed), num_all, cat_all)
    m0.fit(tr[cols_all], ytr); systems["M0 Modelo simple"] = (m0, cols_all)
    # M1: XGBoost afinado (con atributos sensibles)
    m1 = pipe(make_xgb(best_params), num_all, cat_all)
    m1.fit(tr[cols_all], ytr); systems["M1 Modelo avanzado"] = (m1, cols_all)
    # M2: XGBoost sin atributos sensibles (fairness through unawareness)
    m2 = pipe(make_xgb(best_params), num_ns, cat_ns)
    m2.fit(tr[cols_ns], ytr); systems["M2 Sin edad ni sexo"] = (m2, cols_ns)
    # M3: XGBoost con Reweighing sobre edad
    w = reweighing_weights(ytr, tr.age_group.values)
    m3 = pipe(make_xgb(best_params), num_all, cat_all)
    m3.fit(tr[cols_all], ytr, clf__sample_weight=w); systems["M3 Pesos por grupo"] = (m3, cols_all)

    # ---------- Evaluación Train / Val / Test ----------
    results, preds_test, probas_test, thresholds = {}, {}, {}, {}
    for name, (m, cols) in systems.items():
        pv = m.predict_proba(va[cols])[:, 1]
        thr = best_threshold(yva, pv)
        thresholds[name] = thr
        results[name] = {"threshold": thr}
        for split, d, y in [("train", tr, ytr), ("validation", va, yva), ("test", te, yte)]:
            p = m.predict_proba(d[cols])[:, 1]
            results[name][split] = clf_metrics(y, p, thr)
        probas_test[name] = m.predict_proba(te[cols])[:, 1]
        preds_test[name] = (probas_test[name] >= thr).astype(int)

    # M4: post-procesamiento ThresholdOptimizer (Equalized Odds) sobre M1, ajustado con VALIDATION
    to = ThresholdOptimizer(estimator=Float64Proba(m1), constraints="equalized_odds", objective="balanced_accuracy_score",
                            prefit=True, predict_method="predict_proba")
    to.fit(va[cols_all], yva, sensitive_features=va.age_group)
    name4 = "M4 Umbral por grupo"
    rs = np.random.RandomState(seed)
    pred4 = to.predict(te[cols_all], sensitive_features=te.age_group, random_state=rs).astype(int)
    preds_test[name4] = pred4
    tn, fp, fn_, tp = confusion_matrix(yte, pred4, labels=[0, 1]).ravel()
    results[name4] = {"threshold": None, "test": dict(
        accuracy=accuracy_score(yte, pred4), precision=precision_score(yte, pred4, zero_division=0),
        recall=recall_score(yte, pred4), f1=f1_score(yte, pred4), auc=float("nan"),
        balanced_acc=balanced_accuracy_score(yte, pred4), tn=int(tn), fp=int(fp), fn=int(fn_), tp=int(tp))}

    # ---------- Auditoría de equidad en TEST ----------
    fairness = {}
    for name, pred in preds_test.items():
        fairness[name] = {}
        for a in ["age_group", "sex"]:
            fairness[name][a] = fairness_report(yte, pred, te[a].values)
        # IC 95 % bootstrap para diferencia de paridad demográfica por edad
        fairness[name]["age_group"]["demographic_parity_diff_CI95"] = bootstrap_ci(
            yte, pred, te.age_group.values,
            lambda y, p, s: demographic_parity_difference(y, p, sensitive_features=s), seed=seed)

    # ---------- Persistencia de resultados ----------
    summary = dict(dataset=dict(n=len(df), bad_rate=float(df.target.mean())), eda_bias=eda, split=split_info,
                   best_xgb_params=best_params, best_xgb_val_auc=best[0], best_lr_C=best_lr[1],
                   n_trials=args.n_iter, results=results, fairness=fairness)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    # Tabla resumen
    rows = []
    for name in results:
        r = results[name]["test"]
        fa, fs = fairness[name]["age_group"], fairness[name]["sex"]
        rows.append(dict(modelo=name, accuracy=r["accuracy"], precision=r["precision"], recall=r["recall"],
                         f1=r["f1"], auc=r["auc"], DI_edad=fa["approval_rate_ratio_DI"],
                         DPD_edad=fa["demographic_parity_diff"], EOD_edad=fa["equalized_odds_diff"],
                         DI_sexo=fs["approval_rate_ratio_DI"], EOD_sexo=fs["equalized_odds_diff"]))
    table = pd.DataFrame(rows).round(3)
    table.to_csv(out / "summary_test.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n=== RESULTADOS EN TEST ===\n", table.to_string(index=False))

    # ---------- Figuras ----------
    # (1) ROC
    plt.figure(figsize=(5.5, 4.5))
    for name in ["M0 Modelo simple", "M1 Modelo avanzado", "M2 Sin edad ni sexo", "M3 Pesos por grupo"]:
        fpr, tpr, _ = roc_curve(yte, probas_test[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, probas_test[name]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curvas ROC en el conjunto de prueba"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out / "fig1_roc.png", dpi=200); plt.close()

    # (2) Matriz de confusión del modelo M1
    cm = confusion_matrix(yte, preds_test["M1 Modelo avanzado"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred. bueno", "Pred. malo"]); ax.set_yticks([0, 1], ["Real bueno", "Real malo"])
    ax.set_title("Matriz de confusión - M1 (test)"); plt.tight_layout(); plt.savefig(out / "fig2_confusion.png", dpi=200); plt.close()

    # (3) Tasa de aprobación y TPR/FPR por edad
    names = list(preds_test)
    groups = ["adult(>=25)", "young(<25)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, title in zip(axes, ["approval_rate", "TPR", "FPR"],
                                 ["Tasa de aprobación", "TPR (malos detectados)", "FPR (buenos rechazados)"]):
        x = np.arange(len(names)); wdt = .38
        for k, g in enumerate(groups):
            ax.bar(x + (k - .5) * wdt, [fairness[n]["age_group"]["by_group"][g][metric] for n in names], wdt, label=g)
        ax.set_xticks(x, [n.split(" ", 1)[1] for n in names], rotation=30, ha="right", fontsize=7); ax.set_title(title); ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(out / "fig3_fairness_age.png", dpi=200); plt.close()

    # (4) Trade-off desempeño vs. equidad
    plt.figure(figsize=(5.5, 4.2))
    for name in names:
        r = results[name]["test"]
        plt.scatter(fairness[name]["age_group"]["equalized_odds_diff"], r["balanced_acc"], s=70)
        dx, dy, ha = {"M0": (-8, 6, "right"), "M3": (6, -13, "left"), "M4": (-8, 8, "right")}.get(name.split(" ")[0], (6, 5, "left"))
        plt.annotate(name, (fairness[name]["age_group"]["equalized_odds_diff"], r["balanced_acc"]),
                     textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9)
    plt.xlabel("Diferencia de Equalized Odds por edad (menor = más equitativo)"); plt.ylabel("Balanced accuracy (test)")
    plt.title("Compromiso desempeño-equidad"); plt.tight_layout(); plt.savefig(out / "fig4_tradeoff.png", dpi=200); plt.close()

    # (5) Importancia por permutación de M1 (en validación)
    pi = permutation_importance(m1, va[cols_all], yva, scoring="roc_auc", n_repeats=20, random_state=seed)
    imp = pd.Series(pi.importances_mean, index=cols_all).sort_values(ascending=True).tail(10)
    imp.to_csv(out / "permutation_importance_top10.csv")
    plt.figure(figsize=(6, 4)); plt.barh(imp.index, imp.values); plt.xlabel("Caída de AUC al permutar")
    plt.title("Importancia por permutación - M1 (validación)"); plt.tight_layout()
    plt.savefig(out / "fig5_importance.png", dpi=200); plt.close()

    # (6) Búsqueda de hiperparámetros
    plt.figure(figsize=(5.5, 3.8)); plt.plot(sorted([t["val_auc"] for t in trials]), marker="o", ms=3)
    plt.xlabel("Configuración (ordenada)"); plt.ylabel("AUC en validación"); plt.title("Búsqueda aleatoria de hiperparámetros")
    plt.tight_layout(); plt.savefig(out / "fig6_search.png", dpi=200); plt.close()

    print(f"\nListo. Artefactos en: {out.resolve()}")


if __name__ == "__main__":
    main()
