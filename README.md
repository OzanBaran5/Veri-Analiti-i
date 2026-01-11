# 📊 Veri Analitiği ve İstatistiksel Modelleme Portföyü

Bu depo, İstatistiksel Veri Analizi yöntemlerinin (Regresyon, Korelasyon, Hipotez Testleri) teorik temellerini ve Python ile gerçek hayat senaryoları üzerindeki pratik uygulamalarını içermektedir.

**R. [cite_start]Tanju Sirmen**'in ders notlarından referans alınarak hazırlanan bu projeler, teorik bilginin kod ile nasıl hayata geçirildiğini gösterir[cite: 3, 216, 466, 700].

---

## 📂 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Teorik Altyapı (Ders Notları)](#-teorik-altyapı-ders-notları)
- [Analiz Scriptleri](#-analiz-scriptleri)
  - [1. Sosyo-Ekonomik Analizler](#1-sosyo-ekonomik-analizler)
  - [2. Çevresel ve Demografik Analizler](#2-çevresel-ve-demografik-analizler)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Kurulum](#-kurulum)

---

## 💡 Proje Hakkında

Bu çalışmanın temel amacı, ham veriyi anlamlı içgörülere dönüştürürken bilimsel yöntemlere sadık kalmaktır. Depoda yer alan scriptler sadece veriyi görselleştirmekle kalmaz, **Scipy** ve **Scikit-learn** kütüphanelerini kullanarak verinin arkasındaki matematiksel ilişkileri kanıtlar.

**Analizlerde şu sorulara yanıt aranmıştır:**
* *Ekonomik rahatlık suç oranlarını nasıl etkiler?*
* *Eğitim seviyesi ile gelir arasında istatistiksel olarak anlamlı bir fark var mı?*
* *Nüfus artışı ve emisyon arasındaki ilişki nedir?*

---

## 📚 Teorik Altyapı (Ders Notları)

Kodların temel aldığı istatistiksel kavramlar aşağıdaki kaynaklara dayanmaktadır:

| Konu | Açıklama | Kaynak Dosya |
|------|----------|--------------|
| **Regresyon** | [cite_start]Bağımlı ve bağımsız değişkenler arasındaki ilişkinin modellenmesi ($y=mx+b$)[cite: 218, 230]. | `Introduction to Regression.pdf` |
| **Korelasyon** | [cite_start]Değişkenler arasındaki ilişkinin yönü ve gücünün (Pearson r) ölçülmesi[cite: 468]. | `Correlation Analysis.pdf` |
| **Hipotez Testi** | [cite_start]Örneklem verisine dayanarak popülasyon hakkında karar verme ($H_0$ reddi)[cite: 701]. | `Hypothesis Testing.pdf` |
| **Rastgelelik** | [cite_start]Stokastik süreçler ve deterministik olmayan sistemlerin analizi[cite: 76, 142]. | `Randomness.pdf` |

---

## 🛠 Analiz Scriptleri

### 1. Sosyo-Ekonomik Analizler

#### 🏙️ `4 Bölgeli Karar Matrisi.py`
Şehirleri **Tasarruf Potansiyeli** ve **Suç Oranına** göre 4 stratejik bölgeye ayırır.
* **Kullanılan Yöntem:** Mantıksal Segmentasyon, Regresyon Analizi.
* **Çıktı:** "İdeal Bölge", "Riskli Cazibe", "Mütevazı Liman", "Alarm Veren" kategorizasyonu.

#### 📈 `yaşam_kalite_endeksi.py`
Maaş, Kira, İşsizlik ve Suç verilerini normalize ederek şehirler için bir **"Yaşam Kalitesi Puanı"** hesaplar.
* **Kullanılan Yöntem:** Min-Max Normalizasyonu, T-Testi (Eğitim seviyesinin puana etkisi).
* **Özellik:** Verileri 0-1 arasına çekerek adil karşılaştırma yapar.

#### 💰 `Geliri_En_Çok_Etkileyen.py`
Şehrin gelir seviyesini en çok neyin etkilediğini (Eğitim mi? Altyapı mı?) bulur.
* **Kullanılan Yöntem:** Çoklu Doğrusal Regresyon (Multiple Linear Regression).
* **Analiz:** Katsayıların (Coefficients) yüzdesel etkiye dönüştürülmesi.

#### 🚨 `işsizlik_eğitim_suc.py`
Ekonomik rahatlık skoru ile suç oranları arasındaki ilişkiyi inceler.
* **Kullanılan Yöntem:** Pearson Korelasyonu, Hipotez Testi ($p < 0.05$ kontrolü).

#### 📉 `baski_endeksi_analizi.py`
Şehirlerdeki sosyal baskıyı (Suç + İşsizlik) ölçer ve bunun **kiralar üzerindeki etkisini** analiz eder.
* **Kullanılan Yöntem:** Aykırı Değer Tespiti (Outlier Detection) ve Regresyon Eğilimi.

#### 🎓 `Egitim_Fırsatı.py`
Ekonomik Fırsat Doğurganlık Endeksi (EFDE) ile eğitim seviyesi arasındaki ilişkiyi haritalandırır.

---

### 2. Çevresel ve Demografik Analizler

#### 👥 `Nüfus_Tahmin.py`
Türkiye'nin doğurganlık hızı verilerini kullanarak gelecekteki nüfusunu simüle eder.
* **Kullanılan Yöntem:** Zaman Serisi Regresyonu ve Dinamik Simülasyon döngüsü.
* **Özellik:** P-değerine göre *Regresyon* veya *Ortalama* yöntemini seçen **Akıllı Karar Mekanizması**.

#### 🏭 `emisyon_gsyh_analizi.py`
Ekonomik büyüme (GSYH) ile Karbon Emisyonu arasındaki ilişkiyi test eder (Çevresel Kuznets Eğrisi hipotezi).
* **Kullanılan Yöntem:** Korelasyon Analizi, Çift Eksenli (Dual Axis) Görselleştirme.

#### ⚡ `sicaklık_enerji.py`
Sıcaklık değişimlerinin kişi başı enerji tüketimine etkisini analiz eder.
* **Kullanılan Yöntem:** Z-Score ile Aykırı Değer (Outlier) Analizi, %95 Güven Aralığı (Confidence Interval).

#### 💧 `su_tuketim_analizi.py`
Geçmiş su tüketim verilerini analiz ederek tüketim trendinin yönünü ve gücünü belirler.

---

## 💻 Kullanılan Teknolojiler

Proje **Python 3.x** ile geliştirilmiş olup aşağıdaki kütüphaneleri kullanır:

* **`pandas`**: Veri manipülasyonu ve temizleme.
* **`numpy`**: Sayısal hesaplamalar.
* **`matplotlib` & `seaborn`**: Veri görselleştirme (Regresyon doğruları, Scatter plotlar).
* **`scipy`**: İstatistiksel testler (T-Test, Shapiro-Wilk, Pearson r).
* [cite_start]**`scikit-learn`**: Makine öğrenimi modelleri (LinearRegression), MSE ve R² hesaplamaları[cite: 355, 363].

---

## 🚀 Kurulum

1.  Bu depoyu klonlayın:
    ```bash
    git clone https://github.com/OzanBaran5/Veri-Analiti-i.git
    ```

2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    ```

3.  Analiz scriptlerini çalıştırın (Örnek):
    ```bash
    python "4 Bölgeli Karar Matrisi.py"
    ```
    *(Not: Scriptlerin çalışması için `message.txt` veya ilgili `.csv` veri dosyalarının aynı dizinde olduğundan emin olun.)*

---
> **Not:** Bu çalışma, teorik istatistik bilgilerinin pratik veri bilimi problemlerine nasıl uygulanacağını göstermek amacıyla hazırlanmıştır.
