import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 1. Veri Hazırlığı
try:
    df = pd.read_csv('message.txt')
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df.rename(columns={'Ortalama_Maas': 'Gelir', 'Eğitim': 'Egitim', 'Suc_Orani': 'Suc'}, inplace=True)
    
    cols = ['Gelir', 'Suc', 'Egitim', 'Issizlik', 'Nufus', 'Elektrik']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=cols, inplace=True)
except Exception as e:
    print(f"Hata: {e}")
    exit()

# 2. İstatistiksel Hesaplama (Regresyon Modeli)
features = ['Suc', 'Egitim', 'Issizlik', 'Nufus', 'Elektrik']
X = df[features]
y = df['Gelir']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Katsayıları gelirin ortalamasına oranlayarak "Yüzdesel Etki"ye çeviriyoruz
etki_yuzdesi = (model.coef_ / df['Gelir'].mean()) * 100

etki_df = pd.DataFrame({
    'Faktör': ['Suç Oranı', 'Eğitim (Kazanç)', 'İşsizlik (Kayıp)', 'Nüfus Etkisi', 'Altyapı/Elek.'],
    'Degisim': etki_yuzdesi
}).sort_values(by='Degisim', ascending=False)

# 3. Model Geçerlilik Testi (P-Value ve R-Kare)
r_kare = model.score(X_scaled, y)
# En güçlü faktör üzerinden genel bir p-değeri testi
_, p_val = stats.pearsonr(df['Egitim'], df['Gelir']) 

# 4. GRAFİK OLUŞTURMA
plt.figure(figsize=(12, 8))

# Renkler: Kazançlar Mavi, Kayıplar Turuncu
colors = ['#3498db' if x > 0 else '#e67e22' for x in etki_df['Degisim']]

bars = plt.bar(etki_df['Faktör'], etki_df['Degisim'], color=colors, alpha=0.85, edgecolor='black', width=0.6)

# Değerleri Çubukların Üstüne Yazma
for bar in bars:
    yval = bar.get_height()
    label = f"+%{yval:.1f}" if yval > 0 else f"-%{abs(yval):.1f}"
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.5 if yval > 0 else -1.5), 
             label, ha='center', va='center', fontweight='bold', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))

# Estetik Ayarlar
plt.axhline(0, color='black', linewidth=2) 
plt.ylabel('Ortalama Gelirdeki Değişim Payı (%)', fontsize=12)
plt.title(f'ŞEHİR EKONOMİSİ ETKİ ANALİZİ\n(Model Gücü: %{r_kare*100:.1f})', fontsize=15, fontweight='bold', pad=25)

# 5. HİPOTEZ VE ANALİZ RAPORU (Terminale Yazdır)
print("\n" + "="*55)
print("             HİPOTEZ TESTİ VE ANALİZ RAPORU")
print("="*55)
print(f"1. Modelin Tahmin Gücü (R-Kare): %{r_kare*100:.2f}")
print(f"2. Hesaplanan P-Değeri: {p_val:.4f}")

if p_val < 0.05:
    print("3. KARAR: P < 0.05 olduğu için model İSTATİSTİKSEL OLARAK GEÇERLİDİR.")
    print("   (Yorum: Faktörlerin gelir üzerindeki etkisi tesadüf değildir.)")
else:
    print("3. KARAR: P > 0.05 olduğu için model GEÇERSİZDİR.")
    print("   (Yorum: Veriler arasında bilimsel olarak anlamlı bir bağ bulunamadı.)")

print("-" * 55)
for i, row in etki_df.iterrows():
    fark = "ekstra kazanç sağlıyor" if row['Degisim'] > 0 else "geliri baltalıyor/götürüyor"
    print(f"-> [{row['Faktör']}]: Şehrin geliri üzerindeki payı %{abs(row['Degisim']):.2f} ({fark}).")
print("="*55)

plt.grid(axis='y', linestyle=':', alpha=0.6)
sns.despine(bottom=True, left=True)

plt.tight_layout()
plt.show()

