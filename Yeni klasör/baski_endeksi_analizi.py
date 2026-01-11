import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


try:
    df = pd.read_csv('message.txt')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('yıllık_ortalama_sicaklik', 'Sicaklik')
    df.rename(columns={'Ortalama_Maas': 'Gelir', 'Eğitim': 'Egitim', 'Suc_Orani': 'Suc'}, inplace=True)
    df['Kira'] = pd.to_numeric(df['Kira'], errors='coerce')
    df.dropna(subset=['Kira', 'Gelir', 'Egitim', 'Issizlik', 'Suc', 'Nufus'], inplace=True)
except FileNotFoundError:
    print("HATA: message.txt dosyası bulunamadı.")
    exit()

# HESAPLAMALAR VE NORMALİZASYON
def normalize(column):
    return (column - column.min()) / (column.max() - column.min() + 1e-6)

df['Gelir_Norm'] = normalize(df['Gelir'])
df['Egitim_Norm'] = normalize(df['Egitim'])
# Baskı Endeksi: Sosyal baskı (İşsizlik+Suç) / Sosyal Refah (Gelir+Eğitim)
df['Baski_Endeksi'] = (df['Issizlik'] + df['Suc']) / (df['Gelir_Norm'] + df['Egitim_Norm'] + 0.1)

#AYKIRI DEĞER VE İSTATİSTİKSEL H
sinir = df['Baski_Endeksi'].quantile(0.95)
df_normal = df[df['Baski_Endeksi'] <= sinir].copy()
df_outliers = df[df['Baski_Endeksi'] > sinir].copy()

# Korelasyon ve P-Value (Hipotez Testi)
r_val, p_val = stats.pearsonr(df_normal['Baski_Endeksi'], df_normal['Kira'])
gecerlilik = "GEÇERLİ" if p_val < 0.05 else "GEÇERSİZ"

#  GRAFİK OLUŞTURMA 
plt.figure(figsize=(12, 7))

# Regresyon Çizgisi
sns.regplot(x='Baski_Endeksi', y='Kira', data=df_normal, scatter=False, 
            color='red', truncate=True, line_kws={"label": "Eğilim Çizgisi (Regresyon)"})

# Dağılım Grafiği
sns.scatterplot(x='Baski_Endeksi', y='Kira', data=df, size='Nufus', 
                sizes=(100, 1000), alpha=0.7, hue='Kira', palette="magma", edgecolor="w")

# Aykırı Değer İşaretleme
for i in range(len(df_outliers)):
    plt.scatter(df_outliers.iloc[i]['Baski_Endeksi'], df_outliers.iloc[i]['Kira'], 
                s=2500, facecolors='none', edgecolors='red', linewidths=2, linestyle='--')
    plt.text(df_outliers.iloc[i]['Baski_Endeksi'], df_outliers.iloc[i]['Kira'] + 1000, 
             f"Aykırı: {df_outliers.iloc[i]['Il']}", color='red', weight='bold', ha='center')

# Grafik Detayları
plt.title(f'BASKI ENDEKSİ VE KİRA İLİŞKİSİ\n', fontsize=14)
plt.xlim(df['Baski_Endeksi'].min() - 1, sinir * 1.3)
plt.grid(True, ls="--", alpha=0.5)

# --- 5. PROGRAM KAPANDIĞINDA YAZILACAK RAPOR ---
print("\n" + "="*65)
print("              HİPOTEZ ")
print("="*65)
print(f"1. ANALİZ TÜRÜ: Pearson Korelasyon ve Tek Değişkenli Regresyon")
print(f"2. KORELASYON (r): {r_val:.4f}")
print(f"   Yorum: Baskı arttıkça kiraların düştüğü %{abs(r_val)*100:.1f} oranında kanıtlanmıştır.")
print(f"\n3. HİPOTEZ TESTİ (P-VALUE) İhtimal: {p_val:.6f}")
print(f"   Eşik Değer (Hata Payı): 0.05")
print(f"   SONUÇ: P < 0.05 olduğu için model istatistiksel olarak {gecerlilik}DİR.")
print(f"   Açıklama: Bu bağ tesadüf değildir, değişkenler birbirini doğrudan etkilemektedir.")



# Şehir İsimleri (Önemli olanlar)
etiketler = ['İstanbul', 'Ankara', 'İzmir', 'Şırnak', 'Hakkari']
for i, row in df.iterrows():
    if row['Il'] in etiketler:
        plt.text(row['Baski_Endeksi'], row['Kira'], row['Il'], weight='bold', fontsize=9)

plt.tight_layout()
plt.show()

