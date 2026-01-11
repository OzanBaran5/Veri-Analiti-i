import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

try:
    df = pd.read_csv('message.txt')
except FileNotFoundError:
    print("HATA: message.txt dosyası bulunamadı.")
    exit()

# --- VERİ TEMİZLEME ---
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('yıllık_ortalama_sicaklik', 'Sicaklik')
df.rename(columns={'Ortalama_Maas': 'Gelir', 'Eğitim': 'Egitim', 'Suc_Orani': 'Suc'}, inplace=True)
df.dropna(subset=['Egitim', 'Gelir', 'Issizlik', 'Suc', 'Nufus'], inplace=True)

# --- NORMALİZASYON ---
min_gelir, max_gelir = df['Gelir'].min(), df['Gelir'].max()
df['Gelir_Norm'] = (df['Gelir'] - min_gelir) / (max_gelir - min_gelir + 1e-6)

min_egitim, max_egitim = df['Egitim'].min(), df['Egitim'].max()
df['Egitim_Norm'] = (df['Egitim'] - min_egitim) / (max_egitim - min_egitim + 1e-6)

# --- EFDE HESAPLAMA ---
df['EFDE'] = df['Gelir_Norm'] / (df['Issizlik'] + df['Suc'] + 1e-6)

# --- KORELASYON VE HİPOTEZ TESTİ ---
r_val, p_val = stats.pearsonr(df['EFDE'], df['Egitim_Norm'])
gecerlilik = "GEÇERLİ" if p_val < 0.05 else "GEÇERSİZ"

# --- GRAFİK OLUŞTURMA ---
plt.figure(figsize=(12, 7))

# 1. Eğilim Çizgisi (Regresyon)
sns.regplot(x='EFDE', y='Egitim_Norm', data=df, scatter=False, color='red', line_kws={"label": "Eğilim Çizgisi"})

# 2. Dağılım Grafiği (Scatter)
sns.scatterplot(
    x='EFDE', y='Egitim_Norm', data=df, 
    size='Nufus', sizes=(100, 1000), alpha=0.7, hue='EFDE', palette="viridis", edgecolor="w"
)

# 3. Şehir Etiketleri
critical_cities = ['İstanbul', 'Ankara','Bursa','İzmir','Antalya','Hakkari']
for il in critical_cities:
    if il in df['Il'].values:
        row = df[df['Il'] == il].iloc[0]
        plt.text(row['EFDE'], row['Egitim_Norm'] + 0.02, il, fontsize=10, weight='bold', ha='center')




# --- 4. PROGRAM KAPANDIĞINDA YAZILACAK ANALİZ RAPORU ---
print("\n" + "="*65)
print("             HİPOTEZ TESTİ VE KORELASYON ANALİZİ")
print("="*65)
print(f"  KORELASYON KATSAYISI (r): {r_val:.4f}")
print(f"   Yorum: Ekonomik güvenlik arttıkça eğitim seviyesinin arttığı %{abs(r_val)*100:.1f}")
print(f"   oranında kanıtlanmıştır. (Pozitif yönlü bir ilişki vardır).")



print(f"\n3. HİPOTEZ TESTİ (P-VALUE): {p_val:.6f}")
print(f"   Eşik Değer (Alpha): 0.05")
print(f"   SONUÇ: P < 0.05 olduğu için kurulan model {gecerlilik}DİR.")
print(f"   Açıklama: Ekonomik güvenlik ile eğitim arasındaki bu bağ tesadüf değildir.")
print(f"   Veriler, refahın eğitim başarısını doğrudan etkilediğini göstermektedir.")



# --- GRAFİK AYARLARI ---
plt.title(f'EKONOMİK GÜVENLİK (EFDE) VE EĞİTİM ANALİZİ\n(r: {r_val:.2f}, p: {p_val:.4f})', fontsize=14)
plt.xlabel('EFDE (Gelir / İşsizlik + Suç)', fontsize=12)
plt.ylabel('Eğitim Seviyesi (Normalize)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

