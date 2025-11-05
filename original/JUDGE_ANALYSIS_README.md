# Hakim Analizi - Kullanım Talimatları

## 🎯 Amaç

Bu analiz, "**AİHM'deki ülke farkları hakim etkisinden mi kaynaklanıyor?**" sorusuna yanıt verir.

Eğer ülke farkları sadece bazı hakimlerin daha sert olmasından kaynaklanıyorsa, hakim kontrolü eklediğimizde ülke etkisi kaybolmalıdır. Eğer kaybolmazsa → **Sistematik ülke farkları** vardır.

---

## 📋 Önkoşullar

1. **JSON dosyası hazır olmalı:**
   ```bash
   # Git LFS ile:
   git lfs pull

   # Veya manual olarak cases-2000.json dosyasını koyun
   ```

2. **Python paketleri yüklü olmalı:**
   ```bash
   pip3 install pandas numpy matplotlib seaborn statsmodels scipy
   ```

---

## 🚀 Çalıştırma Adımları

### 1. CSV'yi Hakim Bilgileriyle Yeniden Oluştur

```bash
python3 assignment2.py
```

**Çıktı:**
- `extracted_data.csv` - Artık hakim bilgileri de içerir
- Yeni kolonlar: `judge_president`, `judge_count`, `judges_all`, `judge_names_list`

**Kontrol:**
```bash
head extracted_data.csv
```

Şunları görmelisiniz:
- `judge_president`: Başkan hakim ismi
- `judge_count`: Paneldeki hakim sayısı
- `judges_all`: Tüm hakimler (noktalı virgülle ayrılmış)

---

### 2. Hakim Analizini Çalıştır

```bash
python3 judge_analysis.py
```

**Çıktı:**
- Terminal çıktısı: Detaylı analiz sonuçları
- `judge_analysis_visualizations.png` - 6 görselleştirme

---

## 📊 Analiz İçeriği

### Tanımlayıcı İstatistikler
- **En aktif hakimler** (en çok dava gören)
- **Hakim başına ihlal oranları**
- **Hakim varyasyonu** (en yüksek vs en düşük ihlal oranları)
- **Başkan hakim istatistikleri**

### Hakim × Ülke Etkileşimi
- **Bölgesel bias:** Her hakim Doğu vs Batı Avrupa'ya nasıl davranıyor?
- **Hakimler arası tutarlılık:** Hepsi aynı pattern'i mi gösteriyor?
- **Bias dağılımı:** Bazı hakimler daha mı "biased"?

### Model Karşılaştırması (EN ÖNEMLİ!)
```
Model 1: violation ~ country + article + year
Model 2: violation ~ country + article + year + judge_president
```

**Kritik Soru:**
- Model 2'de ülke etkisi kayboldu mu?
- **Kaybolmadıysa** → Ülke farkları sistematik
- **Kayboldu ise** → Hakim ataması meseleydi

---

## 📈 Görselleştirmeler

`judge_analysis_visualizations.png` içerir:

1. **Hakim İhlal Oranı Dağılımı** - Hakimler arası varyasyon
2. **En Aktif Hakimler** - Dava sayısı
3. **Hakim Bölgesel Bias** - Doğu-Batı farkı
4. **Başkan vs Üye** - Başkan hakim fark eder mi?
5. **Deneyim vs İhlal** - Tecrübeli hakimler farklı mı?
6. **Ülke İhlal Oranları** - Top 10 ülke

---

## 🎯 Sonuçları Yorumlama

### Senaryo 1: Ülke Etkisi KALICI (Beklenen)
```
Without judge control: 9/16 countries significant
With judge control: 8/16 countries significant
→ Country effect PERSISTS
```

**Anlam:**
- Hakim kontrolü eklenmesine rağmen ülke etkisi kaybolmadı
- **Sistematik ülke farkları** var
- Sadece "hangi hakim dava aldı" sorunu değil

**Sonuç:** Research question'ınıza güçlü yanıt!

---

### Senaryo 2: Ülke Etkisi AZALDI (Alternatif)
```
Without judge control: 9/16 countries significant
With judge control: 3/16 countries significant
→ Judge effects EXPLAIN country differences
```

**Anlam:**
- Hakim kontrolü ülke etkisini büyük ölçüde açıkladı
- Belki belirli hakimler belirli ülkelere atandı?
- Daha karmaşık hikaye

---

### Senaryo 3: Karma Sonuç
```
Without judge control: 9/16 countries significant
With judge control: 6/16 countries significant
→ BOTH judges AND countries matter
```

**Anlam:**
- Hem hakim hem ülke etkisi var
- İkisi de önemli
- Interaction effect olabilir

---

## 🔬 Akademik Katkı

Bu analiz şunları gösterir:

1. **Robustness Check:** Ülke etkisi hakim atamasından kaynaklanmıyor
2. **Mechanism:** Sistematik vs idiosyncratic ayrımı
3. **Contribution:** Literatürdeki çoğu çalışma hakim kontrolü yapmıyor

**Makalenizde:**
> "To rule out the possibility that country effects are driven by judge
> assignment, we control for judge fixed effects. Country effects persist
> even after controlling for the identity of judges on the panel,
> suggesting systematic rather than idiosyncratic treatment differences."

---

## ⚠️ Olası Sorunlar ve Çözümler

### Problem 1: "Missing judge columns" hatası
```
❌ ERROR: Missing judge columns: ['judge_president', ...]
```

**Çözüm:**
```bash
# assignment2.py'yi çalıştırıp CSV'yi yeniden oluşturun
python3 assignment2.py
```

---

### Problem 2: "No cases with judge information"
```
❌ ERROR: No cases with judge information!
```

**Neden:** JSON'da decision_body boş veya eksik

**Çözüm:**
- JSON dosyasını kontrol edin
- Gerçek JSON dosyası mı yoksa LFS pointer mı?
```bash
head cases-2000.json
# Eğer "version https://git-lfs..." görüyorsanız:
git lfs pull
```

---

### Problem 3: "Singular matrix" hatası

**Neden:** Perfect separation (bazı hakimler sadece 1 ülkede çalışmış)

**Çözüm:** Script'te min_cases parametrelerini artırın:
```python
result1, result2 = simple_country_model_with_judges(
    df,
    min_country_cases=50,  # 30'dan 50'ye çıkar
    min_judge_cases=30     # 20'den 30'a çıkar
)
```

---

## 📝 Sonuçları Raporlama

### Özet Tablo (Örnek):

| Model | Pseudo R² | Sig Countries | AIC |
|-------|-----------|---------------|-----|
| Without Judge | 0.226 | 9/16 (56%) | 800 |
| With Judge | 0.235 | 8/16 (50%) | 795 |
| **Difference** | +0.009 | -1 | -5 |

**Yorum:**
- Hakim kontrolü model fit'i hafifçe iyileştirdi (R² +0.009)
- Ama ülke etkisi hala güçlü (%50 anlamlı)
- → Sistematik ülke farkları

---

## 🎓 İleri Seviye (Opsiyonel)

### Mixed Effects Model (Statsmodels)
```python
# Daha sofistike: her hakim için random intercept
import statsmodels.formula.api as smf

model = smf.mixedlm(
    "has_violation ~ country_name + primary_article + year",
    df,
    groups=df["judge_president"]
)
result = model.fit()
```

**Avantaj:** İstatistiksel olarak daha doğru

**Dezavantaj:** Konvergans sorunları olabilir

---

## 📞 Yardım

Sorun yaşarsanız:
1. CSV'de hakim kolonları var mı kontrol edin
2. Hakim bilgisi olan vaka sayısını kontrol edin
3. Min_cases parametrelerini ayarlayın

**Debug:**
```python
import pandas as pd
df = pd.read_csv('extracted_data.csv')
print(df.columns)  # Hakim kolonları var mı?
print(df['judge_count'].value_counts())  # Kaç vakada hakim bilgisi var?
print(df['judge_president'].value_counts().head())  # En sık başkanlar
```

---

## ✅ Başarı Kriterleri

Analiz başarılı sayılır eğer:
- [x] CSV'de hakim kolonları var
- [x] 500+ vaka hakim bilgisi içeriyor
- [x] Model 1 ve Model 2 başarıyla çalıştı
- [x] Görselleştirme PNG oluştu
- [x] Ülke etkisi karşılaştırması yapıldı

---

**Sorularınız için:** Bu dosyayı güncelleyebilir veya benimle konuşabilirsiniz!
