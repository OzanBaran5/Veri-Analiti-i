import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================
# 1. VERİ YÜKLEME VE HAZIRLIK
# ==============================
try:
    df = pd.read_csv("emisyon_gsyh.csv")
    df = df.sort_values("Yil")
    
    # Veri setinde boşluk varsa temizleyelim
    df = df.dropna(subset=["Toplam_Emisyon", "GSYH_Milyar_USD"])
    
except FileNotFoundError:
    print("HATA: emisyon_gsyh.csv dosyası bulunamadı.")
    # Test için dummy veri (Dosya yoksa çalışması için)
    # df = pd.DataFrame({
    #     'Yil': range(2001, 2024),
    #     'Toplam_Emisyon': np.linspace(300, 550, 23) + np.random.normal(0, 10, 23),
    #     'GSYH_Milyar_USD': np.linspace(200, 900, 23) + np.random.normal(0, 50, 23)
    # })
    exit()

# ==============================
# 2. BİLİMSEL ANALİZ MOTORU
# ==============================
print("\n" + "="*65)
print("       EKONOMİ VE ÇEVRE ETKİLEŞİM RAPORU (Kuznets Analizi)")
print("="*65)

# A) KORELASYON ANALİZİ (Para kirlilik yaratıyor mu?)
x_col = 'GSYH_Milyar_USD'
y_col = 'Toplam_Emisyon'

r_val, p_val = stats.pearsonr(df[x_col], df[y_col])

print(f"[1] İLİŞKİ GÜCÜ (KORELASYON):")
print(f"    Katsayı (r): {r_val:.4f}")
yorum = "Güçlü Pozitif (Büyüdükçe Kirletiyoruz)" if r_val > 0.7 else "Zayıflayan İlişki (Ayrışma Başlamış Olabilir)"
print(f"    Yorum: Ekonomi ile Emisyon arasında {yorum} bir ilişki var.")

# B) HİPOTEZ TESTİ
alpha = 0.05
durum = "İSTATİSTİKSEL OLARAK ANLAMLI" if p_val < alpha else "TESADÜFİ (İLİŞKİ YOK)"
print(f"\n[2] GÜVENİLİRLİK (P-Value: {p_val:.6f}):")
print(f"    Sonuç: Bu ilişki {durum}DIR.")

# C) REGRESYON MODELİ
X = df[[x_col]].values
y = df[y_col].values
model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"\n[3] MODEL BAŞARISI (R2):")
print(f"    Değer: {r2:.4f}")
print(f"    Anlamı: Emisyon artışının %{r2*100:.1f}'i ekonomik büyüme ile açıklanmaktadır.")
print("="*65 + "\n")

# ==============================
# 3. GÖRSELLEŞTİRME (ÇİFT PANEL)
# ==============================
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12))

# --- ÜST GRAFİK: ZAMAN SERİSİ (TARİHÇE) ---
color_emisyon = 'tab:blue'
ax1.set_xlabel('Yıl', fontsize=12, fontweight='bold')
ax1.set_ylabel('Toplam Emisyon (Mt CO₂e)', color=color_emisyon, fontsize=12, fontweight='bold')
ax1.plot(df["Yil"], df["Toplam_Emisyon"], color=color_emisyon, marker="o", linewidth=2, label="Emisyon")
ax1.tick_params(axis='y', labelcolor=color_emisyon)
ax1.grid(True, linestyle="--", alpha=0.5)

# Çift Eksen (GSYH)
ax2 = ax1.twinx() 
color_gsyh = 'tab:red'
ax2.set_ylabel('GSYH (Milyar USD)', color=color_gsyh, fontsize=12, fontweight='bold')
ax2.plot(df["Yil"], df["GSYH_Milyar_USD"], color=color_gsyh, marker="s", linestyle='--', linewidth=2, label="GSYH")
ax2.tick_params(axis='y', labelcolor=color_gsyh)

ax1.set_title("Bölüm 1: Yıllara Göre Gelişim (Tarihsel Süreç)", fontsize=14)

# --- ALT GRAFİK: REGRESYON (BİLİMSEL KANIT) ---
# İşte o meşhur "Kırmızı Koridor" burada devreye giriyor
sns.regplot(x=x_col, y=y_col, data=df, ax=ax3,
            ci=95,  # %95 Güven Aralığı Koridoru
            scatter_kws={'s': 100, 'color': 'purple', 'alpha': 0.6, 'label': 'Yıllık Veriler'},
            line_kws={'color': 'orange', 'linewidth': 3, 'label': f'Etki Trendi (R2={r2:.2f})'})

# Yılları noktaların üzerine yazalım (Hangi yıl nerede?)
for i, row in df.iterrows():
    # Sadece belli yılları yazalım (karmaşa olmasın diye 3 yılda bir veya uç değerler)
    if i % 3 == 0 or row['Yil'] == df['Yil'].max() or row['Yil'] == df['Yil'].min():
        ax3.text(row[x_col], row[y_col], str(int(row['Yil'])), fontsize=9, fontweight='bold')

ax3.set_title("Bölüm 2: Ekonomik Büyümenin Çevreye Etkisi (Korelasyon Analizi)", fontsize=14)
ax3.set_xlabel("GSYH (Milyar USD) - Zenginleşme", fontsize=12)
ax3.set_ylabel("Toplam Emisyon (Mt CO₂e) - Kirlilik", fontsize=12)
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc='upper left')

plt.tight_layout()
plt.show()