import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================================================
# 1. VERİ YÜKLEME VE BİLİMSEL HAZIRLIK
# =========================================================
try:
    df = pd.read_csv('message.txt')
    # Sütun isimlerini temizle
    df.columns = df.columns.str.strip()
    
    # İsimleri standartlaştır
    df = df.rename(columns={
        'yıllık_ortalama_sicaklik': 'Sicaklik',
        'Elektrik': 'Toplam_Enerji'
    })
    
    # KRİTİK ADIM: Kişi Başına Enerji Tüketimi (Normalizasyon)
    df['Kişi_Basi_Enerji'] = df['Toplam_Enerji'] / df['Nufus']

    # Eksik verileri at
    df = df.dropna(subset=['Sicaklik', 'Kişi_Basi_Enerji'])

except FileNotFoundError:
    print("HATA: message.txt dosyası bulunamadı.")
    exit()

# =========================================================
# 2. AYKIRI DEĞER ANALİZİ (OUTLIER DETECTION - Z-SCORE)
# =========================================================
# Ortalamadan 2 standart sapma sapanları yakala
z_scores = stats.zscore(df['Kişi_Basi_Enerji'])
threshold = 2
df['Aykiri_Mi'] = np.abs(z_scores) > threshold
aykiri_sehirler = df[df['Aykiri_Mi'] == True]

# =========================================================
# 3. BİLİMSEL ANALİZ MOTORU (RAPORLAMA)
# =========================================================
print("\n" + "="*65)
print("             BİLİMSEL ANALİZ RAPORU (SICAKLIK vs ENERJİ)")
print("="*65)

# A) KORELASYON
x_col = 'Sicaklik'
y_col = 'Kişi_Basi_Enerji'
r_val, p_val = stats.pearsonr(df[x_col], df[y_col])

print(f"[1] KORELASYON ANALİZİ:")
print(f"    Katsayı (r): {r_val:.4f}")
yorum = "Pozitif (Doğru Orantı)" if r_val > 0 else "Negatif (Ters Orantı)"
print(f"    Yorum: {yorum}. İlişki gücü %{abs(r_val)*100:.1f}.")

# B) HİPOTEZ TESTİ
alpha = 0.05
durum = "ANLAMLI (GEÇERLİ)" if p_val < alpha else "TESADÜFİ (GEÇERSİZ)"
print(f"\n[2] HİPOTEZ TESTİ (P-Value: {p_val:.5f}):")
print(f"    Sonuç: İlişki istatistiksel olarak {durum}DİR.")

# C) REGRESYON BAŞARISI (R2)
X = df[[x_col]].values
y = df[y_col].values
model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"\n[3] REGRESYON BAŞARISI (R2):")
print(f"    Değer: {r2:.4f}")
print(f"    Açıklama: Enerji tüketiminin %{r2*100:.1f}'i sıcaklık ile açıklanabilir.")
print("="*65 + "\n")

# =========================================================
# 4. GÖRSELLEŞTİRME (KORİDORLU & AYKIRI DEĞERLİ)
# =========================================================
plt.figure(figsize=(12, 7))

# A) ANA ANALİZ + KORİDOR (Güven Aralığı)
# sns.regplot buradaki sihirli değnek.
# ci=95 -> %95 Güven Aralığı (O gölgeli koridor)
# scatter_kws -> Mavi noktaların stili
# line_kws -> Kırmızı çizginin stili
sns.regplot(x='Sicaklik', y='Kişi_Basi_Enerji', data=df,
            ci=95,  
            scatter_kws={'s': 80, 'alpha': 0.6, 'color': 'tab:blue', 'label': 'Normal Şehirler'}, 
            line_kws={'color': 'red', 'linewidth': 2, 'label': f'Trend ve %95 Güven Koridoru'})

# B) AYKIRI DEĞERLERİ ÜZERİNE ÇAK (Kırmızı X)
# Koridorun ve mavi noktaların üstüne basması için zorder=5 verdim
sns.scatterplot(x=aykiri_sehirler['Sicaklik'], y=aykiri_sehirler['Kişi_Basi_Enerji'], 
                s=200, color='red', marker='X', label='Aykırı Değerler (Outliers)', zorder=5)

# C) AYKIRI ŞEHİRLERİN İSİMLERİ
for i, row in aykiri_sehirler.iterrows():
    # Yazıların noktaların üzerine binmemesi için hafif yukarı kaydırdım (+ değer ekleyerek)
    plt.text(row['Sicaklik'], row['Kişi_Basi_Enerji'], f"  {row['Il']}", 
             fontsize=11, fontweight='bold', color='darkred', va='center')

# EKSEN VE BAŞLIK AYARLARI
plt.title(f"Sıcaklık ve Enerji Analizi\n(Güven Aralığı ve Aykırı Değer Tespiti)", fontsize=14)
plt.xlabel("Yıllık Ortalama Sıcaklık (°C)", fontsize=12)
plt.ylabel("Kişi Başına Düşen Enerji (Birim)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()