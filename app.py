"""HeartTree: aplikasi Flask prediksi risiko gagal jantung dengan Decision Tree."""
import json
import math
import os

import joblib
import pandas as pd
import sklearn
from flask import Flask, render_template, request

import train_model
from train_model import BASE, CATEGORICAL, NUMERIC

APP_NAME = "HeartTree"
FEATURES = NUMERIC + CATEGORICAL
NAV = [("index", "Beranda"), ("guide_page", "Panduan"), ("predict_page", "Prediksi"), ("performance_page", "Performa"),
       ("dataset_page", "Dataset")]
GROUPS = [("Profil pasien", "pink"), ("Hasil pemeriksaan", "blue"), ("Tes latihan", "green")]


def num(name, label, unit, lo, hi, step, hint, why, group):
    return {"name": name, "label": label, "unit": unit, "kind": "number", "min": lo, "max": hi,
            "step": step, "hint": hint, "why": why, "group": group}


def sel(name, label, options, hint, why, group):
    return {"name": name, "label": label, "unit": "", "kind": "select", "options": options,
            "hint": hint, "why": why, "group": group}


FIELDS = [
    num("Age", "Usia", "tahun", 18, 100, 1, "Data latih mencakup usia 28 sampai 77 tahun.",
        "Risiko penyakit jantung cenderung naik seiring bertambahnya usia.", 0),
    sel("Sex", "Jenis kelamin", [("M", "Laki-laki"), ("F", "Perempuan")], "Sesuai data klinis pasien.",
        "Pada data latih, proporsi pasien berisiko lebih tinggi pada laki-laki.", 0),
    sel("ChestPainType", "Tipe nyeri dada",
        [("ASY", "Tanpa gejala (ASY)"), ("ATA", "Angina atipikal (ATA)"),
         ("NAP", "Nyeri non-angina (NAP)"), ("TA", "Angina tipikal (TA)")],
        "ASY berarti tidak ada keluhan nyeri dada.",
        "Menariknya, pasien tanpa keluhan (ASY) justru paling banyak berisiko pada data ini.", 0),
    num("RestingBP", "Tekanan darah istirahat", "mmHg", 60, 250, 1, "Tekanan sistolik saat istirahat.",
        "Tekanan darah tinggi membebani jantung dalam jangka panjang.", 1),
    num("Cholesterol", "Kolesterol serum", "mg/dl", 80, 700, 1, "Kadar kolesterol total dalam darah.",
        "Kolesterol tinggi dapat menumpuk di pembuluh darah jantung.", 1),
    sel("FastingBS", "Gula darah puasa", [("0", "≤ 120 mg/dl"), ("1", "> 120 mg/dl")],
        "Apakah gula darah puasa di atas 120 mg/dl.", "Gula darah tinggi berkaitan dengan kerusakan pembuluh darah.", 1),
    sel("RestingECG", "EKG istirahat",
        [("Normal", "Normal"), ("ST", "Kelainan gelombang ST-T"), ("LVH", "Hipertrofi ventrikel kiri (LVH)")],
        "Hasil elektrokardiogram saat istirahat.", "EKG merekam aktivitas listrik jantung.", 1),
    num("MaxHR", "Detak jantung maksimum", "bpm", 40, 230, 1, "Detak tertinggi yang tercapai saat tes.",
        "Detak maksimum yang rendah dapat menandakan jantung kurang responsif.", 2),
    sel("ExerciseAngina", "Nyeri dada saat olahraga", [("N", "Tidak"), ("Y", "Ya")],
        "Apakah olahraga memicu nyeri dada.", "Nyeri saat aktivitas fisik adalah tanda klasik gangguan jantung.", 2),
    num("Oldpeak", "Oldpeak (depresi ST)", "", -3, 7, 0.1, "Penurunan segmen ST akibat olahraga.",
        "Nilai tinggi menandakan jantung kekurangan aliran darah saat beraktivitas.", 2),
    sel("ST_Slope", "Kemiringan segmen ST", [("Up", "Naik (Up)"), ("Flat", "Datar (Flat)"), ("Down", "Turun (Down)")],
        "Kemiringan segmen ST pada puncak latihan.", "Faktor paling menentukan pada model ini.", 2),
]
BY_NAME = {f["name"]: f for f in FIELDS}
SAMPLES = {
    "sehat": {"Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 190,
              "FastingBS": "0", "RestingECG": "Normal", "MaxHR": 170, "ExerciseAngina": "N",
              "Oldpeak": 0, "ST_Slope": "Up"},
    "berisiko": {"Age": 60, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 260,
                 "FastingBS": "1", "RestingECG": "ST", "MaxHR": 105, "ExerciseAngina": "Y",
                 "Oldpeak": 2, "ST_Slope": "Flat"},
}

app = Flask(__name__)


def load_artifacts():
    """Muat model; latih ulang otomatis jika versi scikit-learn berbeda dari saat pelatihan."""
    def read():
        return (joblib.load(BASE / "model" / "model.joblib"),
                json.loads((BASE / "model" / "metrics.json").read_text(encoding="utf-8")))
    try:
        model, metrics = read()
        if metrics.get("sklearn_version") == sklearn.__version__ and "target_corr" in metrics:
            return model, metrics
    except Exception:
        pass
    train_model.main()
    return read()


MODEL, METRICS = load_artifacts()
PREP, TREE = MODEL.named_steps["prep"], MODEL.named_steps["tree"]
FEATURE_NAMES = PREP.get_feature_names_out()
SHORT = {"Age": "Usia", "Sex": "Kelamin", "ChestPainType": "Nyeri dada", "RestingBP": "Tek. darah",
         "Cholesterol": "Kolesterol", "FastingBS": "Gula puasa", "RestingECG": "EKG",
         "MaxHR": "Detak maks.", "ExerciseAngina": "Angina olahraga", "Oldpeak": "Oldpeak",
         "ST_Slope": "Slope ST"}


@app.template_filter("pct")
def pct(value, digits=1):
    return f"{value * 100:.{digits}f}".replace(".", ",")


@app.template_filter("fmt")
def fmt(value, digits=1):
    return (f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0").replace(".", ",")


def validate(form):
    """Kembalikan (nilai, galat per kolom, peringatan per kolom)."""
    values, errors, warnings = {}, {}, {}
    for f in FIELDS:
        name, raw = f["name"], (form.get(f["name"]) or "").strip().replace(",", ".")
        if f["kind"] == "select":
            if raw in [v for v, _ in f["options"]]:
                values[name] = int(raw) if name == "FastingBS" else raw
            else:
                errors[name] = "Pilih salah satu opsi."
            continue
        try:
            value = float(raw)
        except ValueError:
            errors[name] = "Isi dengan angka, misalnya 120."
            continue
        if math.isnan(value) or not f["min"] <= value <= f["max"]:
            errors[name] = f"Nilai harus antara {f['min']} dan {f['max']} {f['unit']}.".replace(" .", ".")
            continue
        low, high = METRICS["ranges"][name]
        if not low <= value <= high:
            warnings[name] = f"Di luar rentang data latih ({fmt(low)} sampai {fmt(high)}), hasil kurang andal."
        values[name] = value
    return values, errors, warnings


def reference(name, value):
    """Kategori acuan umum (bukan diagnosis) untuk tekanan darah dan kolesterol."""
    scales = {"RestingBP": [(120, "Normal", "ok"), (140, "Di atas normal", "warn")],
              "Cholesterol": [(200, "Diinginkan", "ok"), (240, "Batas tinggi", "warn")]}
    for limit, text, tone in scales.get(name, []):
        if value < limit:
            return text, tone
    return ("Tinggi", "high") if name in scales else (None, "none")


def predict(values):
    row = pd.DataFrame([values])[FEATURES]
    proba = float(MODEL.predict_proba(row)[0][1])
    x, t = PREP.transform(row), TREE.tree_
    path = TREE.decision_path(x).indices
    steps = []
    for node in path:
        if t.children_left[node] == -1:
            continue
        kind, col = FEATURE_NAMES[t.feature[node]].split("__")
        left = x[0, t.feature[node]] <= t.threshold[node]
        if kind == "cat":
            col, val = col.rsplit("_", 1)
            shown = dict(BY_NAME[col]["options"]).get(val, val)
            steps.append(f"{BY_NAME[col]['label']} pasien {'bukan' if left else 'adalah'} {shown}")
        elif col == "FastingBS":
            steps.append("Gula darah puasa " + ("≤ 120 mg/dl" if left else "> 120 mg/dl"))
        else:
            f = BY_NAME[col]
            steps.append(f"{f['label']} pasien {fmt(values[col])} {f['unit']} "
                         f"{'≤' if left else '>'} {fmt(t.threshold[node])}".replace("  ", " "))
    rows = []
    for f in FIELDS:
        v = values[f["name"]]
        shown = dict(f["options"])[str(v)] if f["kind"] == "select" else f"{fmt(v)} {f['unit']}".strip()
        median = METRICS["medians"].get(f["name"])
        category, tone = reference(f["name"], v) if f["kind"] == "number" else (None, "none")
        rows.append({"label": f["label"], "shown": shown, "category": category, "tone": tone,
                     "median": None if median is None or f["name"] == "FastingBS" else f"{fmt(median)} {f['unit']}".strip()})
    return {"risky": proba >= 0.5, "percent": round(proba * 100, 1), "steps": steps, "rows": rows,
            "leaf_n": int(t.n_node_samples[path[-1]]), "path": [int(n) for n in path]}


def build_insights():
    """Susun penjelasan tiap grafik dari angka hasil training agar selalu sesuai gambar."""
    m = METRICS
    (tn, fp), (fn, tp) = m["confusion"]
    total, auc = tn + fp + fn + tp, m["auc"]
    leaves = int(TREE.tree_.n_leaves)
    kind, root = FEATURE_NAMES[TREE.tree_.feature[0]].split("__")
    root = root.rsplit("_", 1)[0] if kind == "cat" else root
    items = [(k, v) for k, v in m["importance"].items() if v > 0]
    unused = [k for k, v in m["importance"].items() if v == 0]
    rest = ", ".join(f"{k} ({pct(v)}%)" for k, v in items[2:])
    curve, chosen = m["depth_curve"], m["best_params"]["max_depth"]
    msl = m["best_params"]["min_samples_leaf"]
    best = max(d["test"] for d in curve)
    first_best = next(d["depth"] for d in curve if d["test"] == best)
    flat = all(d["test"] == best for d in curve if d["depth"] >= first_best)
    at = next(d for d in curve if d["depth"] == chosen)
    gap = at["train"] - at["test"]
    n, pos = m["n_total"], m["n_positive"]
    share = min(pos, n - pos) / n
    corr = sorted(m["target_corr"].items(), key=lambda kv: kv[1], reverse=True)
    signed = lambda v: f"{v:+.2f}".replace(".", ",")
    weakest = min(corr, key=lambda kv: abs(kv[1]))
    quality = "sangat baik" if auc >= 0.9 else "baik" if auc >= 0.8 else "cukup" if auc >= 0.7 else "kurang"
    return {
        "cm": (f"{tn} pasien sehat dan {tp} pasien berisiko diprediksi benar, jadi {tp + tn} dari {total} prediksi tepat (diagonal biru pekat). "
               f"Sisanya {fp} False Positive dan {fn} False Negative, sehingga recall {pct(tp / (tp + fn))}% dan precision {pct(tp / (tp + fp))}%. "
               f"False Negative lebih berbahaya karena pasien berisiko yang terlewat bisa tidak segera ditangani."),
        "roc": (f"Kurva berada di atas garis putus-putus (tebakan acak) dengan AUC {fmt(auc, 3)}, kategori {quality}. "
                f"Artinya model memberi skor lebih tinggi pada pasien berisiko daripada pasien sehat dengan peluang {pct(auc)}%. "
                f"Pada ambang 50%, True Positive Rate {pct(tp / (tp + fn))}% dan False Positive Rate {pct(fp / (fp + tn))}%."),
        "importance": (f"{items[0][0]} ({pct(items[0][1])}%) dan {items[1][0]} ({pct(items[1][1])}%) menentukan "
                       f"{pct(items[0][1] + items[1][1])}% keputusan pohon, dan akar pohon memakai {root}."
                       + (f" Fitur lain berperan kecil: {rest}." if rest else "")
                       + (f" {len(unused)} fitur bernilai 0% tidak dipakai pohon, bukan berarti tidak berkaitan dengan penyakit jantung."
                          if unused else "")),
        "depth": (f"Akurasi uji naik dari {pct(curve[0]['test'])}% (depth 1) ke {pct(at['test'])}% di depth {chosen} yang dipakai"
                  + (f", lalu mendatar di {pct(best)}% mulai depth {first_best} karena min_samples_leaf={msl}." if flat else ".")
                  + f" Akurasi latih {pct(at['train'])}% berbanding uji {pct(at['test'])}%, "
                  + ("selisihnya kecil sehingga tidak ada tanda overfitting." if gap < 0.05 else "selisihnya besar, tanda overfitting.")
                  + " Kedalaman dipilih lewat validasi silang pada data latih, bukan dari data uji."),
        "classes": (f"{pos} pasien berisiko ({pct(pos / n)}%) dan {n - pos} pasien sehat ({pct((n - pos) / n)}%). "
                    + ("Selisihnya kecil, sehingga kelas cukup seimbang dan SMOTE tidak diperlukan." if share >= 0.35
                       else "Kelas tidak seimbang, sehingga perlu penanganan khusus seperti class_weight atau SMOTE.")),
        "corr": (f"{corr[0][0]} ({signed(corr[0][1])}) dan {corr[1][0]} ({signed(corr[1][1])}) paling searah dengan risiko, "
                 f"sedangkan {corr[-1][0]} ({signed(corr[-1][1])}) berlawanan arah: nilai yang lebih rendah berkaitan dengan "
                 f"risiko lebih tinggi. {weakest[0]} hampir tidak berkorelasi ({signed(weakest[1])}). "
                 f"Korelasi hanya mengukur hubungan linear."),
    }


def build_tree_layout():
    """Hitung posisi simpul dan garis pohon keputusan untuk digambar sebagai SVG."""
    t = TREE.tree_
    w, h, gap_x, gap_y, pad = 130, 56, 146, 108, 16
    center, leaves = {}, [0]

    def place(node, depth):
        if t.children_left[node] == -1:
            x = leaves[0] * gap_x
            leaves[0] += 1
        else:
            left, right = place(t.children_left[node], depth + 1), place(t.children_right[node], depth + 1)
            x = (center[left][0] + center[right][0]) / 2
        center[node] = (x, depth * gap_y)
        return node

    place(0, 0)
    nodes, edges, yes_side = [], [], {}
    for node, (cx, cy) in center.items():
        x, y = cx + pad + w / 2, cy + pad
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            share = float(counts[1] / counts.sum())
            lines = ["Berisiko" if share >= 0.5 else "Sehat", f"n={int(t.n_node_samples[node])} · {pct(share, 0)}%"]
            nodes.append({"id": node, "x": x - w / 2, "y": y, "lines": lines, "leaf": True, "risky": share >= 0.5})
            continue
        kind, col = FEATURE_NAMES[t.feature[node]].split("__")
        if kind == "cat":
            col, val = col.rsplit("_", 1)
            line, yes_side[node] = f"= {val}", "right"
        else:
            line = "≤ 120 mg/dl" if col == "FastingBS" else f"≤ {fmt(t.threshold[node])}"
            yes_side[node] = "left"
        nodes.append({"id": node, "x": x - w / 2, "y": y, "lines": [SHORT[col], line], "leaf": False, "risky": False})
    for node in center:
        if t.children_left[node] == -1:
            continue
        for side, child in (("left", t.children_left[node]), ("right", t.children_right[node])):
            x1, y1 = center[node][0] + pad + w / 2, center[node][1] + pad + h
            x2, y2 = center[child][0] + pad + w / 2, center[child][1] + pad
            ym = (y1 + y2) / 2
            edges.append({"child": child, "label": "Ya" if side == yes_side[node] else "Tidak",
                          "d": f"M{x1:.1f} {y1:.1f} C{x1:.1f} {ym:.1f} {x2:.1f} {ym:.1f} {x2:.1f} {y2:.1f}",
                          "lx": (x1 + x2) / 2, "ly": ym})
    return {"width": leaves[0] * gap_x - (gap_x - w) + 2 * pad, "height": (max(c[1] for c in center.values()) + h + 2 * pad),
            "w": w, "h": h, "nodes": nodes, "edges": edges}


TREE_LAYOUT = build_tree_layout()
INSIGHTS = build_insights()


@app.context_processor
def shared():
    return {"app_name": APP_NAME, "nav": NAV, "active": request.endpoint, "m": METRICS, "tree": TREE_LAYOUT,
            "fields": FIELDS, "ins": INSIGHTS}


@app.route("/")
def index():
    return render_template("index.html", fields=FIELDS, importance=METRICS["importance"])


@app.route("/prediksi", methods=["GET", "POST"])
def predict_page():
    ctx = {"fields": FIELDS, "groups": GROUPS, "form": request.form, "samples": SAMPLES,
           "errors": {}, "warnings": {}, "result": None}
    if request.method == "POST":
        values, ctx["errors"], ctx["warnings"] = validate(request.form)
        if not ctx["errors"]:
            ctx["result"] = predict(values)
    return render_template("predict.html", **ctx)


@app.route("/panduan")
def guide_page():
    sample = {**SAMPLES["berisiko"], "FastingBS": int(SAMPLES["berisiko"]["FastingBS"])}
    return render_template("guide.html", example=predict(sample))


@app.route("/performa")
def performance_page():
    return render_template("performance.html")


@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html", fields=FIELDS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
