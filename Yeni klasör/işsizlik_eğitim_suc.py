import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats 

try:
    df = pd.read_csv('message.txt')
except FileNotFoundError:
    print("UYARI: message.txt bulunamadı.")
    exit()


df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('yıllık_ortalama_sicaklik', 'Sicaklik')
df.rename(columns={'Ortalama_Maas': 'Gelir', 'Eğitim': 'Egitim', 'Suc_Orani': 'Suc'}, inplace=True)

df['Kira'] = pd.to_numeric(df['Kira'], errors='coerce')
df.dropna(subset=['Gelir', 'Kira', 'Suc', 'Issizlik', 'Nufus'], inplace=True)

# --- 1. NORMALİZASYON ---
min_gelir = df['Gelir'].min()
max_gelir = df['Gelir'].max()
df['Gelir_Norm'] = (df['Gelir'] - min_gelir) / (max_gelir - min_gelir + 1e-6)

# --- 2. EKONOMİK RAHATLIK SKORU HESAPLAMA ---
df['Ekonomik_Rahatlik'] = df['Gelir_Norm'] / (df['Kira'] * df['Issizlik'] + 1e-6)

# --- 3. BÖLGE TANIMI ---
ort_skor = df['Ekonomik_Rahatlik'].mean()
ort_suc = df['Suc'].mean()

def rahatlik_bolge_belirle(row):
    if row['Ekonomik_Rahatlik'] >= ort_skor and row['Suc'] < ort_suc:
        return "Yüksek Rahatlık & Düşük Suç (İDEAL)" 
    elif row['Ekonomik_Rahatlik'] < ort_skor and row['Suc'] >= ort_suc:
        return "Düşük Rahatlık & Yüksek Suç (RİSKLİ)" 
    elif row['Ekonomik_Rahatlik'] >= ort_skor and row['Suc'] >= ort_suc:
        return "Pahalı ve Sorunlu Rahatlık (ÇELİŞKİLİ)"
    else:
        return "Gelişime Açık (PASİF)"

df['Bolge'] = df.apply(rahatlik_bolge_belirle, axis=1)

# --- 4. HİPOTEZ TESTİ VE KORELASYON HESABI ---
# r_val: Korelasyon gücü, p_val: Hipotez testi anlamlılık değeri
r_val, p_val = stats.pearsonr(df['Ekonomik_Rahatlik'], df['Suc'])
durum = "Anlamlı" if p_val < 0.05 else "Anlamsız"

# --- 5. GRAFİK OLUŞTURMA ---
plt.figure(figsize=(12, 8))

# REGRESYON ÇİZGİSİ (Trend Analizi)
sns.regplot(
    data=df, x='Ekonomik_Rahatlik', y='Suc', 
    scatter=False, color='red', line_kws={"linestyle": "--", "label": "Regresyon Eğilimi"}
)

# SCATTER PLOT (Nokta Dağılımı)
sns.scatterplot(
    data=df, x='Ekonomik_Rahatlik', y='Suc', 
    hue='Bolge', size='Nufus', sizes=(100, 1000), 
    alpha=0.7, palette='deep' 
)

# Ortalama Eksen Çizgileri
plt.axvline(x=ort_skor, color='black', linestyle=':', linewidth=1, alpha=0.5) 
plt.axhline(y=ort_suc, color='black', linestyle=':', linewidth=1, alpha=0.5) 

# İstatistiksel Bilgi Kutusu (Korelasyon + Hipotez Testi)
istatistik_notu = f'Korelasyon (r): {r_val:.2f}\nP-Değeri: {p_val:.4f}\nSonuç: {durum}'
plt.text(0.05, 0.05, istatistik_notu, transform=plt.gca().transAxes, 
         fontsize=11, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# Kritik İlleri Etiketleme
critical_cities = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Hakkari', 'Van', 'Antalya'] 
renk_paleti = {
    'Yüksek Rahatlık & Düşük Suç (İDEAL)': 'green', 
    'Düşük Rahatlık & Yüksek Suç (RİSKLİ)': 'red', 
    'Pahalı ve Sorunlu Rahatlık (ÇELİŞKİLİ)': 'orange', 
    'Gelişime Açık (PASİF)': 'navy'
}

for il in critical_cities:
    if il in df['Il'].values:
        row = df[df['Il'] == il].iloc[0]
        plt.annotate(il, (row['Ekonomik_Rahatlik'], row['Suc']), 
                     textcoords="offset points", xytext=(5, 5), ha='left', 
                     fontsize=10, color=renk_paleti.get(row['Bolge'], 'black'), weight='bold')

plt.title('Ekonomik Rahatlık ve Suç İlişkisi: Regresyon ve Hipotez Analizi', fontsize=16)
plt.xlabel('Ekonomik Rahatlık Skoru', fontsize=12)
plt.ylabel('Suç Oranı', fontsize=12)
plt.legend(title='Bölge Analizi', loc='upper right', bbox_to_anchor=(1.15, 1)) 
plt.grid(True, alpha=0.3)
plt.tight_layout()


print("-" * 40)
print(f"Korelasyon Katsayısı (r): {r_val:.4f}")
print(f"Hipotez Testi P-Değeri: {p_val:.4f}")
print(f"Karar: İlişki istatistiksel olarak {durum}.")
print("-" * 40)

plt.show()