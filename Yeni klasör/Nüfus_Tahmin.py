import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. VERİ HAZIRLIĞI
veriler = {
    2001: 2.38, 2014: 2.19, 2015: 2.16, 2016: 2.11, 2017: 2.08,
    2018: 2.00, 2019: 1.89, 2020: 1.77, 2021: 1.71, 2022: 1.63,
    2023: 1.51, 2024: 1.48
}

# Veriyi analize uygun formata (DataFrame) çeviriyoruz.
df = pd.DataFrame(list(veriler.items()), columns=['Yil', 'Dogurganlik'])

def nufus_tahmini_yap_ve_ciz(baslangic_nufusu, yil_sayisi=5):
    print(f"--- BİLİMSEL ANALİZ RAPORU ---\n")

    # ---------------------------------------------------------
    # A) KORELASYON ANALİZİ (Dosya: dersnot...8705.pdf)
    # ---------------------------------------------------------
    # Yıllar ilerledikçe doğurganlık hızı ne yönde değişiyor?
    # PDF [cite: 468, 475]'e göre Pearson korelasyon katsayısını hesaplıyoruz.
    korelasyon, _ = stats.pearsonr(df['Yil'], df['Dogurganlik'])
    
    print(f"[1] KORELASYON ANALİZİ:")
    print(f"Yıl ve Doğurganlık Arasındaki İlişki (r): {korelasyon:.3f}")
    if korelasyon < -0.7:
        print("Yorum: Güçlü NEGATİF ilişki (Yıllar geçtikçe doğurganlık ciddi şekilde düşüyor).")
    else:
        print("Yorum: İlişki zayıf veya pozitif.")
    print("-" * 30)

    # ---------------------------------------------------------
    # B) HİPOTEZ TESTİ (Dosya: dersnot...8749.pdf)
    # ---------------------------------------------------------
    # H0: Doğurganlık hızı ortalaması yenilenme düzeyi olan 2.10'a eşittir. [cite: 765]
    # H1: Ortalamamız 2.10'dan farklıdır. [cite: 774]
    # Tek Örneklem T-Testi (One-sample t-test) kullanıyoruz. [cite: 729]
    t_stat, p_value = stats.ttest_1samp(df['Dogurganlik'], 2.10)
    
    print(f"[2] HİPOTEZ TESTİ (Referans Değer: 2.10):")
    print(f"p-değeri: {p_value:.5f}")
    if p_value < 0.05: # PDF [cite: 779]'daki alpha=0.05 anlamlılık düzeyi
        print("Sonuç: H0 Reddedildi. Mevcut ortalama 2.10'dan istatistiksel olarak farklı.")
    else:
        print("Sonuç: H0 Reddedilemedi. Fark tesadüfi olabilir.")
    print("-" * 30)

    # ---------------------------------------------------------
    # C) REGRESYON ANALİZİ (Dosya: dersnot...8721.pdf)
    # ---------------------------------------------------------
    # Gelecek yılların doğurganlık hızını tahmin etmek için model kuruyoruz.
    # Model: Doğurganlık = b0 + b1 * Yıl [cite: 230]
    X = df[['Yil']] # Bağımsız değişken (Girdi)
    y = df['Dogurganlik'] # Bağımlı değişken (Çıktı)
    
    model = LinearRegression()
    model.fit(X, y) # Modeli eğitiyoruz
    
    r2 = r2_score(y, model.predict(X)) # Modelin başarısı (R-Kare) [cite: 341]
    
    print(f"[3] REGRESYON ANALİZİ:")
    print(f"Model Denklemi: Doğurganlık = {model.intercept_:.2f} + ({model.coef_[0]:.4f} * Yıl)")
    print(f"Model Güvenilirliği (R2): {r2:.3f}")
    print("-" * 30)

    # ---------------------------------------------------------
    # NÜFUS TAHMİNİ (DİNAMİK HESAPLAMA)
    # ---------------------------------------------------------
    current_population = baslangic_nufusu
    yenilenme_duzeyi = 2.10
    
    yillar_plot = [2024]
    nufuslar_plot = [current_population]
    tahmin_edilen_hizlar = [] # Grafik için saklayalım

    print(f"\n--- Yıllık Simülasyon (Regresyon Destekli) ---")
    
    for i in range(1, yil_sayisi + 1):
        gelecek_yil = 2024 + i
        
        # ADIM 1: O yılın doğurganlık hızını REGRESYON ile tahmin et
        # Eski kodda sabit ortalama kullanılıyordu, şimdi model kullanıyoruz.
        tahmini_hiz = model.predict([[gelecek_yil]])[0]
        tahmin_edilen_hizlar.append(tahmini_hiz)
        
        # ADIM 2: Senin formülünle değişim yüzdesini hesapla
        fark = tahmini_hiz - yenilenme_duzeyi
        # Not: Formülünüzü korudum ancak matematiksel olarak nüfus artış hızı
        # genellikle (Doğum - Ölüm + Göç) ile hesaplanır.
        # Bu formül doğurganlık farkına dayalı bir simülasyondur.
        yillik_degisim_yuzdesi = (fark / yenilenme_duzeyi) * 100 
        
        # ADIM 3: Nüfusu güncelle
        degisim_miktari = current_population * (yillik_degisim_yuzdesi / 1000)
        current_population += degisim_miktari
        
        yillar_plot.append(gelecek_yil)
        nufuslar_plot.append(int(current_population))
        
        print(f"{gelecek_yil} Tahmini Hız: {tahmini_hiz:.2f} -> Nüfus: {int(current_population):,}")

    # ---------------------------------------------------------
    # GÖRSELLEŞTİRME
    # ---------------------------------------------------------
    # Grafik 



    plt.figure(figsize=(12, 6))
    
    # Ana Nüfus Grafiği
    plt.plot(yillar_plot, nufuslar_plot, marker='o', linestyle='-', color='#1f77b4', label='Tahmini Nüfus', linewidth=2)
    
    # Başlıklar
    plt.title(f'Bilimsel Nüfus Tahmini (Regresyon Modelli)\n(R2 Başarısı: {r2:.2f})', fontsize=14)
    plt.xlabel('Yıl', fontsize=12)
    plt.ylabel('Nüfus', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(yillar_plot)
    
    # Değerleri yazdırma
    for x, y in zip(yillar_plot, nufuslar_plot):
        plt.text(x, y + (y*0.001), f'{y:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

# --- KULLANIM ---
turkiye_nufus_2024 = 85372377
nufus_tahmini_yap_ve_ciz(turkiye_nufus_2024, 10)