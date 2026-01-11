import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
# Ders notlarında (dersnot_5744_1765368721.pdf) regresyon için scikit-learn kullanıldığı için eklenmiştir[cite: 355, 357].
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. VERİ SETİ OLUŞTURMA
# ---------------------------------------------------------
# message.txt dosyası olmadığı için örnek veri seti oluşturuluyor.
data = {
    'Il': ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Adana', 'Konya', 'Gaziantep', 'Mersin', 'Kayseri', 'Eskişehir', 'Trabzon', 'Samsun', 'Denizli', 'Şanlıurfa', 'Malatya', 'Erzurum', 'Diyarbakır', 'Kocaeli', 'Manisa'],
    'Maas': [35000, 31000, 30000, 28000, 27000, 24000, 23000, 22000, 24000, 25000, 26000, 24000, 23000, 25000, 20000, 21000, 20500, 21000, 32000, 25000],
    'Kira': [18000, 14000, 15000, 12000, 13000, 10000, 9000, 8500, 10000, 9500, 10000, 9000, 8500, 9500, 7000, 7500, 7000, 7500, 14000, 10000],
    'Suc_Orani': [4.5, 3.8, 4.0, 3.2, 3.5, 4.8, 2.5, 3.9, 3.6, 2.8, 2.2, 2.4, 2.9, 2.7, 4.1, 3.0, 2.6, 4.3, 3.4, 2.8]
}
df = pd.DataFrame(data)
df = df.rename(columns={'Il': 'Sehir'})

# ---------------------------------------------------------
# 2. HESAPLAMALAR VE KATEGORİZASYON
# ---------------------------------------------------------
df['Tasarruf'] = df['Maas'] - df['Kira']

avg_tasarruf = df['Tasarruf'].mean()
avg_suc = df['Suc_Orani'].mean()

def kategori_belirle(row):
    if row['Tasarruf'] >= avg_tasarruf and row['Suc_Orani'] <= avg_suc:
        return 'İdeal (Zengin & Güvenli)'
    elif row['Tasarruf'] >= avg_tasarruf and row['Suc_Orani'] > avg_suc:
        return 'Riskli Cazibe (Zengin ama Tehlikeli)'
    elif row['Tasarruf'] < avg_tasarruf and row['Suc_Orani'] <= avg_suc:
        return 'Mütevazı Liman (Fakir ama Güvenli)'
    else:
        return 'Alarm Veren (Fakir & Tehlikeli)'

df['Kategori'] = df.apply(kategori_belirle, axis=1)

# ---------------------------------------------------------
# 3. İSTATİSTİKSEL ANALİZLER (DERS NOTLARINA UYGUN)
# ---------------------------------------------------------
print("--- İSTATİSTİKSEL ANALİZ RAPORU ---\n")

# A) KORELASYON ANALİZİ (Dosya: dersnot...8705.pdf)
# Değişkenler arasındaki ilişkinin yönünü ve gücünü ölçer (Pearson r).
corr_matrix = df[['Maas', 'Kira', 'Tasarruf', 'Suc_Orani']].corr()
print("1. KORELASYON MATRİSİ:")
print(corr_matrix)
print("\nNot: Korelasyon nedensellik belirtmez, sadece ilişkiyi gösterir[cite: 473].\n")

# B) HİPOTEZ TESTİ (Dosya: dersnot...8749.pdf)
# Soru: "Suç oranı yüksek ve düşük şehirlerin tasarruf ortalamaları farklı mı?"
# Test: Bağımsız İki Örneklem T-Testi (Independent Samples t-test)[cite: 896].

yuksek_suc = df[df['Suc_Orani'] > avg_suc]['Tasarruf']
dusuk_suc = df[df['Suc_Orani'] <= avg_suc]['Tasarruf']

# Önce Normallik Testi (Shapiro-Wilk) [cite: 1017]
stat_y, p_y = stats.shapiro(yuksek_suc)
stat_d, p_d = stats.shapiro(dusuk_suc)

print("2. HİPOTEZ TESTİ (T-Testi):")
if p_y > 0.05 and p_d > 0.05:
    # Veriler normal dağılıyorsa parametrik test (t-test) uygulanır[cite: 891].
    t_stat, p_value = stats.ttest_ind(yuksek_suc, dusuk_suc)
    print(f"T-Testi Sonucu: p-değeri = {p_value:.4f}")
    if p_value < 0.05:
        print("Karar: H0 Reddedildi. Gruplar arasında anlamlı fark var[cite: 859].")
    else:
        print("Karar: H0 Reddedilemedi. Anlamlı bir fark yok.")
else:
    print("Veriler normal dağılım göstermediği için parametrik olmayan Mann-Whitney U testi önerilir[cite: 922].")

# C) REGRESYON ANALİZİ (Dosya: dersnot...8721.pdf)
# Tasarruf miktarını kullanarak Suç Oranını tahmin eden model.
# Burada sklearn kullanımı PDF'teki örnekle birebir uyumludur[cite: 357].

X = df[['Tasarruf']] # Bağımsız değişken
y = df['Suc_Orani']  # Bağımlı değişken

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred) # Belirlilik Katsayısı [cite: 341]

print(f"\n3. REGRESYON ANALİZİ (sklearn ile):")
print(f"Denklem: Suç Oranı = {model.intercept_:.2f} + ({model.coef_[0]:.5f} * Tasarruf)")
print(f"R-Kare (R2): {r2:.4f}")
print("-" * 40)

# ---------------------------------------------------------
# 4. GÖRSELLEŞTİRME
# ---------------------------------------------------------
plt.figure(figsize=(14, 10))

# Veri Noktaları
sns.scatterplot(data=df, x='Tasarruf', y='Suc_Orani', hue='Kategori', style='Kategori', s=150, palette='deep', zorder=2)

# Regresyon Doğrusu (Ders notlarında "Regression Line" olarak geçer [cite: 238])
plt.plot(df['Tasarruf'], y_pred, color='red', linewidth=2, label='Regresyon Doğrusu', zorder=1)

# Ortalama Çizgileri
plt.axvline(x=avg_tasarruf, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=avg_suc, color='gray', linestyle='--', linewidth=1)

# Etiketler ve Açıklamalar
for k in df['Kategori'].unique():
    subset = df[df['Kategori'] == k]
    head_tail = pd.concat([subset.head(2), subset.tail(2)])
    for i, row in head_tail.iterrows():
        plt.text(row['Tasarruf'], row['Suc_Orani']+0.01, row['Sehir'], fontsize=9, fontweight='bold', color='black')

# Bölge İsimleri
plt.text(df['Tasarruf'].max(), df['Suc_Orani'].min(), 'İDEAL BÖLGE', ha='right', va='bottom', fontsize=12, color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
plt.text(df['Tasarruf'].max(), df['Suc_Orani'].max(), 'RİSKLİ CAZİBE', ha='right', va='top', fontsize=12, color='orange', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
plt.text(df['Tasarruf'].min(), df['Suc_Orani'].min(), 'MÜTEVAZI LİMAN', ha='left', va='bottom', fontsize=12, color='blue', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
plt.text(df['Tasarruf'].min(), df['Suc_Orani'].max(), 'ALARM BÖLGESİ', ha='left', va='top', fontsize=12, color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

plt.title(f'Finansal Huzur ve Güvenlik Matrisi\n(Regresyon R2: {r2:.2f})', fontsize=16, fontweight='bold')
plt.xlabel('Aylık Net Tasarruf Potansiyeli (TL)', fontsize=12)
plt.ylabel('Suç Oranı (%)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()