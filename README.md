# Kalp Damar Rahatsızlıkları Riskine Göre Kalp Hastalığı Tahmini

**Açıklanabilir Yapay Zeka (XAI) Yaklaşımıyla Risk Faktörü Analizi**

Bu proje, hastalara ait klinik risk faktörlerinden yola çıkarak kalp hastalığı varlığını sınıflandıran ve tahminlerini **açıklanabilir** kılan bir makine öğrenmesi çalışmasıdır. Microsoft AI internship programı kapsamında geliştirilmiştir.

---

## ⚠️ Önemli Not (Kapsam)

Kullanılan veri seti (UCI Cleveland) bir **teşhis** verisidir — yani bir kişinin ölçüm anında kalp hastalığı olup olmadığını içerir. Bu nedenle model *"gelecekte kalp krizi geçirir mi?"* sorusunu değil, *"mevcut risk faktörleri kalp hastalığına işaret ediyor mu?"* sorusunu yanıtlar.

**Bu bir tanı aracı değildir; yalnızca eğitimsel/analitik bir çalışmadır.**

---

## 📊 Veri Seti

- **Kaynak:** UCI Heart Disease (Cleveland)
- **Boyut:** 303 hasta, 14 değişken
- **Hedef:** `target` (1 = kalp hastalığı var, 0 = yok)
- **Risk faktörleri:** yaş, cinsiyet, göğüs ağrısı tipi, kan basıncı, kolesterol, maksimum kalp hızı, ST depresyonu, boyanan damar sayısı vb.

---

## 🔍 Yöntem

1. **Keşifsel Veri Analizi (EDA)** — dağılımlar, korelasyon analizi, hedef kırılımları
2. **Sızıntısız (leakage-free) ön işleme** — ölçekleyici yalnızca eğitim verisine fit edildi
3. **Üç model:**
   - Lojistik Regresyon (yorumlanabilir baseline)
   - Random Forest
   - XGBoost
4. **Tıbbi bağlama uygun değerlendirme** — accuracy'nin ötesinde AUC, duyarlılık, özgüllük, kalibrasyon
5. **SHAP ile açıklanabilirlik** — global ve lokal analiz
6. **Çapraz doğrulama** — SHAP ve lojistik regresyon katsayılarının karşılaştırılması

---

## 📈 Sonuçlar

| Model | 5-Katlı CV AUC |
|-------|:-------------:|
| Lojistik Regresyon | 0.893 |
| **Random Forest** | **0.909** |
| XGBoost | 0.887 |

**En etkili risk faktörleri:** göğüs ağrısı tipi (`cp`), boyanan damar sayısı (`ca`), maksimum kalp hızı (`thalach`), ST depresyonu (`oldpeak`) ve talasemi (`thal`).

SHAP analizi ile lojistik regresyon katsayıları büyük ölçüde örtüşmektedir; bu da bulguların tek bir modelin kaprisine bağlı olmadığını gösterir. Tüm faktörler kardiyoloji literatüründeki bilinen iskemik kalp hastalığı belirteçleriyle uyumludur.

---

## 🚀 Nasıl Çalıştırılır

1. Bu repoyu indirin veya klonlayın.
2. Gerekli paketleri kurun:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap
   ```
3. `kalp_hastaligi_tahmini.ipynb` dosyasını Jupyter Notebook veya Google Colab'da açın.
4. `heart.csv` dosyasının notebook ile aynı klasörde olduğundan emin olun.
5. Tüm hücreleri çalıştırın.

> Google Colab kullanıyorsanız: notebook'u yükleyin, `heart.csv`'yi sol paneldeki dosya alanına ekleyin ve "Tümünü çalıştır" deyin.

---

## 🛠️ Kullanılan Teknolojiler

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn

---

## 📌 Sınırlamalar

- Veri seti küçük (~300 hasta) ve tek merkezlidir; genellenebilirliği sınırlıdır.
- Teşhis verisidir, gelecekteki kalp krizini öngörmez.
- Klinik kullanıma uygun değildir; eğitimsel bir çalışmadır.
- Gerçek bir risk skorlama aracı için uzunlamasına (prospektif) veri, dış doğrulama ve klinik onay gerekir.

---

## 🔮 Geliştirme Fikirleri

- Daha büyük veri seti (ör. 70.000 satırlık Cardiovascular Disease)
- Hiperparametre optimizasyonu
- Eşik (threshold) ayarı ile duyarlılık odaklı optimizasyon
