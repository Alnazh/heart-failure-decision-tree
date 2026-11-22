# HeartTree - Prediksi Risiko Gagal Jantung

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?style=flat-square&logo=scikit-learn)
![Akurasi](https://img.shields.io/badge/Akurasi-83.2%25-green?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-0.896-green?style=flat-square)

Aplikasi web berbasis Flask untuk mengklasifikasikan **risiko gagal jantung** dari data klinis pasien menggunakan algoritma **Decision Tree**. Hasil berupa label **Berisiko** atau **Risiko Rendah**, probabilitasnya, dan jalur keputusan pohon yang menjelaskan prediksi. Proyek ini adalah UTS mata kuliah Pembelajaran Mesin.

---

## Fitur Aplikasi

- **Lima halaman** - Beranda (edukasi dan indikator), Panduan (cara pakai, cara membaca pohon, glosarium, tanya jawab), Prediksi, Performa, dan Dataset
- **Prediksi Risiko** - isi 11 data klinis dan dapatkan label, probabilitas, dan saran tindak lanjut
- **Diagram Pohon Berwarna** - garis dan kotak jalur yang dilalui pasien disorot (merah untuk berisiko, hijau untuk rendah), plus daftar aturan dan tabel nilai pasien dibanding median data latih dan kategori acuan (tekanan darah, kolesterol)
- **Validasi Input** - galat per kolom, mendukung koma desimal, dan peringatan jika nilai di luar rentang data latih
- **Contoh Data** - tombol pengisi otomatis pasien sehat dan pasien berisiko
- **Performa Model** - metrik, serta grafik hasil training (confusion matrix, kurva ROC, kepentingan fitur, pengaruh `max_depth`) dan diagram pohon
- **Kamus Kode** - arti ASY, M, Up, dan kode lain di bawah setiap diagram pohon
- **Latih Ulang Otomatis** - jika versi scikit-learn berbeda dari saat model dilatih, aplikasi melatih ulang saat dijalankan

---

## Teknologi

| Komponen | Teknologi |
|---|---|
| Backend | Flask 3.1, Gunicorn |
| Machine Learning | scikit-learn 1.7 (DecisionTreeClassifier, Pipeline, ColumnTransformer) |
| Data Processing | Pandas, NumPy |
| Visualisasi | Diagram pohon SVG buatan sendiri (tanpa library grafik) |
| Penyimpanan Model | Joblib |
| Frontend | HTML, CSS, JavaScript (desain neobrutalism bernuansa biru) |

---

## Dataset

- **Sumber:** [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Jumlah data:** 918 baris (917 setelah pembersihan)
- **Fitur input (11):** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- **Target:** HeartDisease (1 = berisiko, 0 = sehat)
- **Proporsi kelas:** 55,4% berisiko, 44,6% sehat (cukup seimbang)
- **Pembersihan:** 1 baris `RestingBP = 0` dibuang, `Cholesterol = 0` (172 baris) dianggap tidak tercatat dan diisi median

---

## Performa Model

Evaluasi pada 184 data uji (20% data, pembagian berstrata).

| Metrik | Nilai |
|---|---|
| Algoritma | Decision Tree (`criterion=gini`, `max_depth=3`, `min_samples_leaf=20`) |
| Accuracy | 83,2% |
| Precision | 86,6% |
| **Recall** | **82,4%** |
| F1-Score | 84,4% (validasi silang 5 lipatan: 85,8%) |
| AUC | 0,896 |
| Split Data | 80% Training / 20% Testing |
| Pencarian parameter | GridSearchCV, 5 lipatan, skor F1 |

Fitur paling menentukan adalah `ST_Slope` (72%) dan `ChestPainType` (16%).

---

## Instalasi Lokal

```bash
git clone https://github.com/Alnazh/heart-failure-decision-tree.git
cd heart-failure-decision-tree
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Buka browser di `http://127.0.0.1:5000`

---

## Struktur Project

```
heartrisk/
├── app.py              # Aplikasi Flask (rute, validasi, prediksi)
├── train_model.py      # Script pelatihan model
├── heart_failure_decision_tree.ipynb  # Notebook Google Colab
├── Procfile            # Konfigurasi Gunicorn
├── requirements.txt
├── data/heart.csv
├── model/              # model.joblib dan metrics.json
├── templates/          # base, index, guide, predict, performance, dataset
└── static/             # style.css, app.js, img/ (grafik hasil training)
```

---

## Training Ulang Model

```bash
python train_model.py
```

Script membersihkan data, membagi data 80:20, mencari parameter terbaik, lalu menyimpan `model.joblib`, `metrics.json`, dan grafik di `static/img/`. Gunakan versi library yang sama dengan `requirements.txt` agar model kompatibel. Script yang sama bisa dijalankan di Google Colab.

---

## Cara Penggunaan

1. Buka halaman utama dan pergi ke bagian **Prediksi**
2. Isi data pasien, atau klik **Pasien sehat** / **Pasien berisiko** untuk contoh
3. Klik **Prediksi sekarang**
4. Aplikasi menampilkan label, probabilitas, dan jalur keputusan

> Aplikasi ini untuk pembelajaran dan bukan pengganti diagnosis dokter.
