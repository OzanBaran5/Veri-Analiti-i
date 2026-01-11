import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================
# 1. VERİ YÜKLEME VE HAZIRLIK
# ==============================
try:
    # CSV dosyasının çalıştığın klasörde olduğundan emin ol
    df = pd.read_csv("su_verisi.csv")
    df = df.sort_values("Yıl")
    # 2008 öncesi veriler eksik/tutarsız olabilir diye filtreledim
    df = df[df["Yıl"] >= 2008].copy()
except FileNotFoundError:
    print("HATA: su_verisi.csv dosyası bulunamadı. Lütfen dosyayı aynı klasöre koyun.")
    exit()

# Kişi başı günlük su (litre) hesaplama
# (Toplam m3 -> Litre) / (Nüfus * 365 gün)
df["Kisi_Basi_Gunluk_Su_Litre"] = (
    df["Toplam_Su_Miktari_Bin_m3"] * 1_000_000
) / (df["Nüfus"] * 365)

# ==============================
# 2. BİLİMSEL ANALİZ MOTORU (GEÇMİŞ VERİ İÇİN)
# ==============================
print("\n" + "="*65)
print("             BİLİMSEL ANALİZ RAPORU (GEÇMİŞ DÖNEM)")
print("="*65)

# A) KORELASYON ANALİZİ (Pearson)
x_col = 'Yıl'
y_col = 'Kisi_Basi_Gunluk_Su_Litre'

r_val, p_korelasyon = stats.pearsonr(df[x_col], df[y_col])

print(f"[1] KORELASYON ANALİZİ:")
print(f"    Katsayı (r): {r_val:.4f}")
yorum = "Pozitif (Artış Trendi)" if r_val > 0 else "Negatif (Azalış Trendi)"
print(f"    Yorum: Yıllar ilerledikçe su tüketiminde %{abs(r_val)*100:.1f} düzeyde {yorum} var.")

# B) HİPOTEZ TESTİ (P-Value)
alpha = 0.05
durum = "GEÇERLİ (İSTATİSTİKSEL OLARAK ANLAMLI)" if p_korelasyon < alpha else "GEÇERSİZ (TESADÜF OLABİLİR)"

print(f"\n[2] HİPOTEZ TESTİ (Güven Düzeyi %95):")
print(f"    P-Value Değeri: {p_korelasyon:.6f}")
print(f"    SONUÇ: Tespit edilen trend {durum}DIR.")

# C) REGRESYON ANALİZİ (R-Kare - Mevcut Trend Eğilimi)
X = df[[x_col]].values
y = df[y_col].values
model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"\n[3] REGRESYON PERFORMANSI:")
print(f"    R-Kare (R2): {r2:.4f}")
print(f"    Açıklama: Su tüketimindeki değişimin %{r2*100:.1f}'i zaman faktörü ile açıklanabilir.")
print(f"    Model Denklemi: Tüketim = {model.intercept_:.2f} + ({model.coef_[0]:.4f} * Yıl)")
print("="*65 + "\n")

# ==============================
# 4. GÖRSELLEŞTİRME (SADELEŞTİRİLMİŞ)
# ==============================
fig, ax1 = plt.subplots(figsize=(12, 7))

# A) Gerçek Veri (Mavi Noktalar ve Çizgi)
ax1.plot(df["Yıl"], df["Kisi_Basi_Gunluk_Su_Litre"], marker="o", color="tab:blue", label="Gerçek Veri (2008-2024)", linewidth=2, markersize=8)

# B) Regresyon Doğrusu (Kırmızı Tarihsel Trend)
# Bu bir tahmin değil, mevcut verinin ortalamasını gösteren matematiksel çizgidir.
regresyon_y = model.predict(X)
ax1.plot(df["Yıl"], regresyon_y, color="red", linestyle="-", linewidth=3, label=f"Genel Eğilim (Trend R2={r2:.2f})")

# --- İPTAL EDİLEN YEŞİL ÇİZGİ ---
# ax1.plot(df_su_tahmin["Yıl"], ... color="green", label="Gelecek Tahmini")
# --------------------------------

# Eksen ve Etiketler (Sol Taraf - Su)
ax1.set_xlabel("Yıl", fontsize=12, fontweight='bold')
ax1.set_ylabel("Kişi Başı Günlük Su (Litre)", fontsize=12, color="tab:blue", fontweight='bold')
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.6)

# Nüfus Eklentisi (İkincil Sağ Eksen - Turuncu)
# BURASI DEĞİŞTİ: Artık sadece mevcut 'df' verisini kullanıyor, geleceği değil.
ax2 = ax1.twinx()
ax2.plot(df["Yıl"], df["Nüfus"], color="tab:orange", alpha=0.6, linestyle="-.", linewidth=2, marker="x", label="Nüfus (Sağ Eksen)")
ax2.set_ylabel("Nüfus", color="tab:orange", fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor="tab:orange")

# Başlık ve Legend
# Başlıktan "ve Tahmini" ibaresini kaldırdım.
plt.title(f"Su Tüketimi Bilimsel Analizi (2008-2024)\n(Korelasyon: {r_val:.2f} | P-Value: {p_korelasyon:.5f})", fontsize=14)

# Legend Birleştirme (Sol ve Sağ eksen etiketlerini birleştirir)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, shadow=True)

plt.tight_layout()
# Grafiği kaydetmek istersen aşağıdaki satırı açabilirsin:
# plt.savefig("su_tuketim_analizi_sade.png", dpi=300)
plt.show()