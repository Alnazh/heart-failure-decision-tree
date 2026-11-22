"""Pelatihan model Decision Tree untuk klasifikasi risiko gagal jantung."""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")

BASE = Path(__file__).parent
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"
SEED = 42


def load_data():
    df = pd.read_csv(BASE / "data" / "heart.csv")
    df = df[df["RestingBP"] > 0].copy()  # tekanan darah 0 tidak mungkin terjadi
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)  # 0 berarti tidak tercatat
    return df


def build_pipeline(**tree_params):
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        sparse_threshold=0,
    )
    return Pipeline([("prep", prep),
                     ("tree", DecisionTreeClassifier(random_state=SEED, **tree_params))])


def evaluate(model, X, y):
    pred, proba = model.predict(X), model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, proba),
    }


def save_figures(out, df, numeric, target, y_test, proba, cm, importance, depth_curve):
    """Simpan grafik evaluasi dan EDA (palet biru) sebagai PNG ke folder out."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import roc_auc_score, roc_curve

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ink, blue, sky, pink, green = "#111111", "#2f6bff", "#7cc7ff", "#ff5c7c", "#2ed3a0"
    blues = LinearSegmentedColormap.from_list("nb_blue", ["#ffffff", sky, blue])
    diverge = LinearSegmentedColormap.from_list("nb_div", [pink, "#ffffff", blue])
    cm = np.array(cm)
    with plt.style.context("default"):

        def finish(fig, ax, title, name):
            ax.set_title(title, fontsize=13, fontweight="bold", color=ink, pad=12)
            for spine in ax.spines.values():
                spine.set_color(ink)
                spine.set_linewidth(2)
            fig.savefig(out / name, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(cm, cmap=blues)
        ax.grid(False)
        ax.set_xticks([0, 1], ["Sehat", "Berisiko"])
        ax.set_yticks([0, 1], ["Sehat", "Berisiko"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=26, fontweight="bold",
                        color="white" if cm[i][j] > cm.max() / 2 else ink)
        finish(fig, ax, "Confusion Matrix", "confusion_matrix.png")

        fpr, tpr, _ = roc_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.fill_between(fpr, tpr, color=sky, alpha=0.35)
        ax.plot(fpr, tpr, color=blue, lw=3, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
        ax.plot([0, 1], [0, 1], "--", color=ink, lw=1.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", edgecolor=ink)
        finish(fig, ax, "Kurva ROC", "roc_curve.png")

        items = [(k, v) for k, v in importance.items() if v > 0][::-1]
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.barh([k for k, _ in items], [v * 100 for _, v in items], color=blue, edgecolor=ink, linewidth=2)
        for y, (_, v) in enumerate(items):
            ax.text(v * 100 + 1, y, f"{v * 100:.1f}%", va="center", fontweight="bold")
        ax.set_xlim(0, max(v for _, v in items) * 118)
        ax.set_xlabel("Kontribusi (%)")
        finish(fig, ax, "Kepentingan Fitur", "feature_importance.png")

        depth = [d["depth"] for d in depth_curve]
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.plot(depth, [d["train"] for d in depth_curve], "o-", color=pink, lw=3, mec=ink, label="Train")
        ax.plot(depth, [d["test"] for d in depth_curve], "o-", color=blue, lw=3, mec=ink, label="Test")
        ax.set_xticks(depth)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("Akurasi")
        ax.grid(alpha=0.25)
        ax.legend(edgecolor=ink)
        finish(fig, ax, "Akurasi terhadap max_depth", "depth_curve.png")

        cols = list(numeric) + [target]
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.imshow(corr, cmap=diverge, vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)), cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color=ink)
        finish(fig, ax, "Korelasi Fitur Numerik", "correlation.png")

        counts = df[target].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        bars = ax.bar(["Sehat", "Berisiko"], counts.values, color=[green, pink], edgecolor=ink, linewidth=2)
        ax.bar_label(bars, fontweight="bold", padding=3)
        ax.set_ylim(0, counts.max() * 1.15)
        ax.set_ylabel("Jumlah pasien")
        finish(fig, ax, "Distribusi Kelas Target", "class_distribution.png")


def main():
    df = load_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    grid = GridSearchCV(
        build_pipeline(),
        {"tree__max_depth": [2, 3, 4, 5, 6, 7, 8],
         "tree__criterion": ["gini", "entropy"],
         "tree__min_samples_leaf": [1, 5, 10, 20]},
        scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    scores = evaluate(model, X_test, y_test)

    # Pengaruh kedalaman pohon terhadap akurasi (parameter lain mengikuti model terbaik)
    best = {k.split("__")[1]: v for k, v in grid.best_params_.items() if k != "tree__max_depth"}
    depth_curve = []
    for depth in range(1, 11):
        m = build_pipeline(max_depth=depth, **best).fit(X_train, y_train)
        depth_curve.append({"depth": depth,
                            "train": m.score(X_train, y_train),
                            "test": m.score(X_test, y_test)})

    # Kepentingan fitur dijumlahkan per kolom asli (gabungan hasil one-hot)
    names = model.named_steps["prep"].get_feature_names_out()
    importance = {}
    for name, value in zip(names, model.named_steps["tree"].feature_importances_):
        col = name.split("__")[1]
        col = col.rsplit("_", 1)[0] if name.startswith("cat__") else col
        importance[col] = importance.get(col, 0) + float(value)
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    metrics = {
        **scores,
        "cv_f1": grid.best_score_,
        "best_params": {k.split("__")[1]: v for k, v in grid.best_params_.items()},
        "confusion": confusion_matrix(y_test, model.predict(X_test)).tolist(),
        "importance": importance,
        "depth_curve": depth_curve,
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
        "n_total": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "n_positive": int(y.sum()),
        "sklearn_version": sklearn.__version__,
        "medians": {c: float(df[c].median()) for c in NUMERIC},
        "target_corr": {c: float(df[c].corr(df[TARGET])) for c in NUMERIC},
    }

    save_figures(BASE / "static" / "img", df, NUMERIC, TARGET, y_test,
                 model.predict_proba(X_test)[:, 1], metrics["confusion"], importance, depth_curve)
    joblib.dump(model, BASE / "model" / "model.joblib")
    (BASE / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Parameter terbaik:", metrics["best_params"])
    print({k: round(v, 4) for k, v in scores.items()}, "| CV F1:", round(grid.best_score_, 4))


if __name__ == "__main__":
    main()
