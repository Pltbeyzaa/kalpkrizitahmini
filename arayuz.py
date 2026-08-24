# ============================================================
#  KALP HASTALIĞI RİSK TAHMİN ARAYÜZÜ  (Gradio + SHAP)
#  Bu hücreyi Colab'da tek başına çalıştırabilirsiniz.
# ============================================================
# !pip install gradio shap xgboost   <- Colab'da en üste ekleyin

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import gradio as gr

# --- 1) Modeli eğit (heart.csv aynı klasörde olmalı) ---
df = pd.read_csv("heart.csv").drop_duplicates().reset_index(drop=True)
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9,
                      eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
FEATURES = list(X.columns)

TR_ADLAR = {  # SHAP grafiğinde okunabilir Türkçe etiketler
    "age": "Yaş", "sex": "Cinsiyet", "cp": "Göğüs ağrısı tipi",
    "trestbps": "Kan basıncı", "chol": "Kolesterol", "fbs": "Açlık kan şekeri",
    "restecg": "Dinlenme EKG", "thalach": "Maks. kalp hızı", "exang": "Egzersiz anjinası",
    "oldpeak": "ST depresyonu", "slope": "ST eğimi", "ca": "Damar sayısı", "thal": "Talasemi"
}

# --- 2) Tahmin + açıklama fonksiyonu ---
def tahmin_et(age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal):
    girdi = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]], columns=FEATURES)
    risk = float(model.predict_proba(girdi)[0, 1])

    # Metinsel sonuç
    if risk >= 0.5:
        sonuc = f"## 🔴 Yüksek Risk\n### Tahmini kalp hastalığı olasılığı: **%{risk*100:.1f}**"
    else:
        sonuc = f"## 🟢 Düşük Risk\n### Tahmini kalp hastalığı olasılığı: **%{risk*100:.1f}**"
    sonuc += "\n\n*Bu araç yalnızca eğitim amaçlıdır, tıbbi tanı aracı değildir.*"

    # SHAP lokal açıklama grafiği
    sv = explainer.shap_values(girdi)[0]
    katki = pd.Series(sv, index=[TR_ADLAR[f] for f in FEATURES])
    katki = katki.reindex(katki.abs().sort_values(ascending=True).index).tail(8)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    renkler = ["#e76f51" if v > 0 else "#2a9d8f" for v in katki.values]
    ax.barh(katki.index, katki.values, color=renkler)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Bu tahmini hangi faktörler etkiledi?\n(kırmızı: riski artırdı, yeşil: azalttı)")
    ax.set_xlabel("SHAP katkısı")
    plt.tight_layout()
    return sonuc, fig

# --- 3) Arayüzü kur ---
with gr.Blocks(title="Kalp Hastalığı Risk Tahmini") as demo:
    gr.Markdown("# ❤️ Kalp Hastalığı Risk Tahmin Aracı\n"
                "Aşağıdaki klinik değerleri girin, model risk tahminini ve **neden** böyle "
                "tahmin ettiğini açıklasın. *(Eğitim amaçlı çalışma — tıbbi tanı değildir.)*")
    with gr.Row():
        with gr.Column():
            age = gr.Slider(20, 100, value=54, step=1, label="Yaş")
            sex = gr.Radio([("Kadın", 0), ("Erkek", 1)], value=1, label="Cinsiyet")
            cp = gr.Dropdown([("0 - Tipik anjina", 0), ("1 - Atipik anjina", 1),
                              ("2 - Anjina dışı ağrı", 2), ("3 - Asemptomatik", 3)],
                             value=0, label="Göğüs ağrısı tipi")
            trestbps = gr.Slider(80, 200, value=130, step=1, label="Dinlenme kan basıncı (mm Hg)")
            chol = gr.Slider(100, 600, value=245, step=1, label="Kolesterol (mg/dl)")
            fbs = gr.Radio([("Hayır", 0), ("Evet", 1)], value=0, label="Açlık kan şekeri > 120 mg/dl")
            restecg = gr.Dropdown([("0 - Normal", 0), ("1 - ST-T anormalliği", 1),
                                   ("2 - Sol ventrikül hipertrofisi", 2)], value=1, label="Dinlenme EKG")
        with gr.Column():
            thalach = gr.Slider(60, 220, value=150, step=1, label="Ulaşılan maksimum kalp hızı")
            exang = gr.Radio([("Hayır", 0), ("Evet", 1)], value=0, label="Egzersize bağlı anjina")
            oldpeak = gr.Slider(0, 7, value=1.0, step=0.1, label="ST depresyonu (oldpeak)")
            slope = gr.Dropdown([("0 - Yukarı eğimli", 0), ("1 - Düz", 1),
                                 ("2 - Aşağı eğimli", 2)], value=1, label="ST segmenti eğimi")
            ca = gr.Dropdown([(str(i), i) for i in range(5)], value=0, label="Boyanan damar sayısı (ca)")
            thal = gr.Dropdown([("0", 0), ("1 - Normal", 1), ("2 - Sabit defekt", 2),
                                ("3 - Geri dönüşlü defekt", 3)], value=2, label="Talasemi (thal)")
    btn = gr.Button("Riski Hesapla", variant="primary")
    with gr.Row():
        cikti_metin = gr.Markdown()
    cikti_grafik = gr.Plot(label="Açıklama")

    btn.click(tahmin_et,
              inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal],
              outputs=[cikti_metin, cikti_grafik])

# Colab'da paylaşılabilir link için share=True
demo.launch(share=True)
