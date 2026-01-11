import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 1. VERİ YÜKLEME
# ---------------------------------------------------------
# Kendi dosyanı kullanmak için alttaki satırın başındaki # işaretini kaldır:
# df = pd.read_csv('message.txt')

# Şimdilik kod çalışsın diye örnek veri (Sen dosyayı bağlayınca burası devre dışı kalır)
if 'df' not in locals():
    data = {
        'Sehir': ['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Bursa', 'Adana', 'Konya', 'Gaziantep', 'Mersin', 'Kayseri', 
                  'Eskişehir', 'Trabzon', 'Samsun', 'Denizli', 'Şanlıurfa', 'Malatya', 'Erzurum', 'Diyarbakır', 'Kocaeli', 'Muğla'],
        'Maas': [35000, 31000, 30000, 27000, 28000, 24000, 23000, 22000, 24000, 25000, 
                 26000, 24000, 23000, 25000, 20000, 21000, 20500, 21000, 32000, 26000],
        'Kira': [25000, 15000, 18000, 19000, 13000, 10000, 8000, 8500, 11000, 9000, 
                 10000, 9500, 8500, 9500, 7000, 7500, 6500, 7500, 14000, 20000]
    }
    df = pd.DataFrame(data)

# Sütun isimlerini düzeltme (Orijinal koduna sadık kalarak)
if 'Il' in df.columns:
    df = df.rename(columns={'Il': 'Sehir', 'Ortalama_Maas': 'Maas'})
if 'Ortalama_Maas' in df.columns:
    df = df.rename(columns={'Ortalama_Maas': 'Maas'})

# ---------------------------------------------------------
# 2. HESAPLAMALAR VE ANALİZLER (PDF ENTEGRASYONU)
# ---------------------------------------------------------
df['Barinma_Yuku'] = (df['Kira'] / df['Maas']) * 100

print("\n--- BİLİMSEL ANALİZ RAPORU ---")

# A) KORELASYON
korelasyon = df['Maas'].corr(df['Kira'])
print(f"[1] Korelasyon (r): {korelasyon:.2f}")

# B) REGRESYON
X = df[['Maas']]
y = df['Kira']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"[2] Regresyon R2: {r2:.2f} (Kira tahmini başarısı)")

# C) HİPOTEZ TESTİ
avg_maas = df['Maas'].mean()
grup_zengin = df[df['Maas'] >= avg_maas]['Barinma_Yuku']
grup_normal = df[df['Maas'] < avg_maas]['Barinma_Yuku']
t_stat, p_val = stats.ttest_ind(grup_zengin, grup_normal)
print(f"[3] Hipotez Testi p-değeri: {p_val:.4f}")
print("-" * 30)

# ---------------------------------------------------------
# 3. GÖRSELLEŞTİRME (YAN YANA İKİ GRAFİK)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- SOL GRAFİK: LOLLIPOP (SENİN ORİJİNAL AYARLARINLA) ---
top_burden = df.sort_values('Barinma_Yuku', ascending=False).head(15)

axes[0].hlines(y=top_burden['Sehir'], xmin=0, xmax=top_burden['Barinma_Yuku'], color='grey', alpha=0.6)

# Orijinal Renk Mantığın: >70 Kırmızı, >50 Turuncu, Diğerleri Sarı(Gold)
colors = ['red' if x > 70 else 'orange' if x > 50 else 'gold' for x in top_burden['Barinma_Yuku']]

axes[0].scatter(top_burden['Barinma_Yuku'], top_burden['Sehir'], color=colors, s=150, alpha=1)

# Orijinal Kritik Eşik Çizgileri
axes[0].axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='Kritik Eşik (%50)')
axes[0].axvline(x=30, color='green', linestyle='--', linewidth=1.5, label='Sağlıklı Sınır (%30)')

for i, (value, name) in enumerate(zip(top_burden['Barinma_Yuku'], top_burden['Sehir'])):
    axes[0].text(value + 1, i, f"%{value:.1f}", va='center', fontsize=9, fontweight='bold', color='black')

axes[0].set_title('Barınma Yükü Endeksi (En Yüksek 15 Şehir)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Kira / Maaş Oranı (%)')
axes[0].legend(loc='lower right')
axes[0].grid(True, axis='x', linestyle='--', alpha=0.3)

# --- SAĞ GRAFİK: REGRESYON ANALİZİ (MAAŞ vs KİRA) ---
# Burada maaşın kirayı nasıl etkilediğini gösteriyoruz.
sns.scatterplot(x=df['Maas'], y=df['Kira'], s=100, color='blue', alpha=0.6, ax=axes[1], label='Şehir Verileri')
axes[1].plot(df['Maas'], y_pred, color='red', linewidth=2, label=f'Regresyon Doğrusu (R2={r2:.2f})')

# Şehir isimlerini noktalara ekleyelim (Karışıklık olmasın diye sadece bazıları veya hepsi)
for i in range(len(df)):
    axes[1].text(df['Maas'].iloc[i]+100, df['Kira'].iloc[i], df['Sehir'].iloc[i], fontsize=8, alpha=0.7)

axes[1].set_title('Regresyon Analizi: Maaş Arttıkça Kira Ne Oluyor?', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Ortalama Maaş (TL)')
axes[1].set_ylabel('Ortalama Kira (TL)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()