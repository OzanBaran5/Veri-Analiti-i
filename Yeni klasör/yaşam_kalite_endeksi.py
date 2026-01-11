import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 1. VERİYİ OKUMA (Mock Data Oluşturuyoruz)
# ---------------------------------------------------------
# 'message.txt' olmadığı için kodun çalışması adına örnek veri seti.
# Sen kendi dosyanı okutmak için alttaki satırın yorumunu kaldırabilirsin:
# df = pd.read_csv('message.txt')

if 'df' not in locals():
    data = {
        'Il': ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Eskişehir', 'Konya', 'Gaziantep', 'Trabzon', 'Kayseri', 
               'Muğla', 'Çanakkale', 'Tekirdağ', 'Adana', 'Mersin', 'Diyarbakır', 'Şanlıurfa', 'Van', 'Erzurum', 'Samsun'],
        'Ortalama_Maas': [35000, 32000, 30000, 28000, 27000, 26000, 24000, 23000, 24000, 25000, 
                          25000, 24500, 27500, 22000, 22500, 20000, 19000, 18500, 21000, 23000],
        'Eğitim': [8.5, 9.2, 8.8, 7.5, 7.8, 9.5, 7.0, 6.5, 7.2, 7.4, 
                   8.0, 8.3, 7.6, 6.8, 7.0, 5.5, 4.8, 5.2, 6.5, 7.1], # 10 üzerinden puan
        'Kira': [20000, 15000, 16000, 13000, 17000, 9000, 8000, 7500, 8500, 8000, 
                 18000, 11000, 12000, 9500, 10000, 6000, 5500, 5000, 6000, 8500],
        'Issizlik': [12, 10, 11, 9, 8, 8, 7, 13, 9, 8, 
                     6, 7, 8, 14, 13, 18, 19, 20, 10, 9], # Yüzde
        'Suc_Orani': [5, 4, 4, 3, 4, 2, 2, 4, 2, 3, 
                      2, 2, 3, 5, 4, 4, 3, 3, 2, 3] # 1-5 arası endeks
    }
    df = pd.DataFrame(data)

# Sütun isimlerini düzeltme (Orijinal Kod)
df = df.rename(columns={'Il': 'Sehir', 'Ortalama_Maas': 'Maas'})

# ---------------------------------------------------------
# 2. YAŞAM KALİTESİ ENDEKSİ HESABI (Orijinal Kod)
# ---------------------------------------------------------
cols = ['Maas', 'Eğitim', 'Kira', 'Issizlik', 'Suc_Orani']
df_norm = df.copy()

for col in cols:
    # Normalizasyon
    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Endeks Formülü
df['Yasam_Puan'] = (df_norm['Maas'] + df_norm['Eğitim']) - (df_norm['Kira'] + df_norm['Issizlik'] + df_norm['Suc_Orani'])

# ---------------------------------------------------------
# 2.5 BİLİMSEL ANALİZ ENTEGRASYONU (PDF EKLENTİSİ)
# ---------------------------------------------------------
print("\n--- BİLİMSEL ANALİZ RAPORU ---")

# A) KORELASYON ANALİZİ (Correlation Analysis)
# Soru: Maaş arttıkça Yaşam Puanı da artıyor mu?
korelasyon = df['Maas'].corr(df['Yasam_Puan'])
print(f"[1] Korelasyon Analizi:")
print(f"Maaş ile Yaşam Puanı İlişkisi (r): {korelasyon:.2f}")
print("Yorum: Güçlü pozitif ilişki." if korelasyon > 0.7 else "Orta/Zayıf ilişki.")

# B) HİPOTEZ TESTİ (Hypothesis Testing)
# Soru: Eğitimi yüksek olan şehirlerin yaşam puanı, düşük olanlardan farklı mı?
egitim_ort = df['Eğitim'].mean()
grup_yuksek = df[df['Eğitim'] >= egitim_ort]['Yasam_Puan']
grup_dusuk = df[df['Eğitim'] < egitim_ort]['Yasam_Puan']

# Bağımsız Örneklem T-Testi
t_stat, p_val = stats.ttest_ind(grup_yuksek, grup_dusuk)
print(f"\n[2] Hipotez Testi (Eğitim Seviyesine Göre):")
print(f"p-değeri: {p_val:.4f}")
if p_val < 0.05:
    print("Sonuç: H0 Reddedildi. Eğitim seviyesi yaşam puanını istatistiksel olarak etkiliyor.")
else:
    print("Sonuç: H0 Reddedilemedi. Fark tesadüfi olabilir.")

# C) REGRESYON ANALİZİ (Regression Analysis)
# Soru: İşsizlik oranını bilirsek Yaşam Puanını tahmin edebilir miyiz?
X = df[['Issizlik']]
y = df['Yasam_Puan']
model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))
y_pred = model.predict(X)

print(f"\n[3] Regresyon Analizi:")
print(f"Model: Puan = {model.intercept_:.2f} + ({model.coef_[0]:.2f} x İşsizlik)")
print(f"Başarı (R2): {r2:.2f}")
print("-" * 40)

# ---------------------------------------------------------
# 3. GÖRSELLEŞTİRME (Güncellenmiş)
# ---------------------------------------------------------
# Orijinal grafiği koruyoruz, yanına bilimsel analizi ekliyoruz.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- SOL GRAFİK: ORİJİNAL BAR PLOT ---
top_quality = df.sort_values('Yasam_Puan', ascending=False).head(20)
sns.barplot(data=top_quality, x='Yasam_Puan', y='Sehir', palette='magma', ax=axes[0])

axes[0].set_title('Yaşam Kalitesi Endeksi (Sıralama)', fontsize=14)
axes[0].set_xlabel('Hesaplanan Puan')
axes[0].set_ylabel('Şehir')
axes[0].grid(True, axis='x', linestyle='--', alpha=0.5)

for index, value in enumerate(top_quality['Yasam_Puan']):
    axes[0].text(value, index, f"{value:.2f}", va='center', fontsize=9, fontweight='bold')

# --- SAĞ GRAFİK: REGRESYON ANALİZİ (YENİ) ---
# İşsizlik vs Yaşam Puanı ilişkisini gösteren Regresyon Grafiği
sns.scatterplot(x=df['Issizlik'], y=df['Yasam_Puan'], s=100, color='blue', alpha=0.6, ax=axes[1], label='Şehirler')
# Kırmızı Regresyon Doğrusu
axes[1].plot(df['Issizlik'], y_pred, color='red', linewidth=2, label=f'Regresyon Doğrusu (R2={r2:.2f})')

# Şehir isimlerini noktalara ekleme
for i in range(len(df)):
    axes[1].text(df['Issizlik'].iloc[i]+0.2, df['Yasam_Puan'].iloc[i], df['Sehir'].iloc[i], fontsize=8, alpha=0.7)

axes[1].set_title('Regresyon: İşsizlik Yaşam Kalitesini Nasıl Etkiliyor?', fontsize=14)
axes[1].set_xlabel('İşsizlik Oranı (%)')
axes[1].set_ylabel('Hesaplanan Yaşam Puanı')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()