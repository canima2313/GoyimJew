import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#get location of current dir
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
policy_df = pd.read_csv(os.path.join(base_dir, 'Data_Polis.csv'))
claims_df = pd.read_csv(os.path.join(base_dir, 'Data_Klaim.csv'))

#ganti spasi di kolom jadi underscore
policy_df.columns = policy_df.columns.str.strip().str.lower().str.replace(' ', '_')
claims_df.columns = claims_df.columns.str.strip().str.lower().str.replace(' ', '_')
#gabung data
try:
    data = pd.merge(policy_df, claims_df, on='nomor_polis', how='left')
    print("Merge berhasil\n")
    print(data.head())

except Exception as e: #masa gagal si
    print(f"Merge failed because {e}")

data['nominal_klaim_yang_disetujui'] = data['nominal_klaim_yang_disetujui'].fillna(0)
data['nominal_biaya_rs_yang_terjadi'] = data['nominal_biaya_rs_yang_terjadi'].fillna(0)
data['status_klaim'] = data['status_klaim'].fillna('no-claim')

data['tanggal_lahir'] = pd.to_datetime(data['tanggal_lahir'])
data['tanggal_pasien_masuk_rs'] = pd.to_datetime(data['tanggal_pasien_masuk_rs'])
data['tanggal_pasien_keluar_rs'] = pd.to_datetime(data['tanggal_pasien_keluar_rs'])
data['tanggal_efektif_polis'] = pd.to_datetime(data['tanggal_efektif_polis'])
#usia
data['tahun_masuk'] = data['tanggal_pasien_masuk_rs'].dt.year
data['tahun_masuk'] = data['tahun_masuk'].fillna(2025)
data['usia'] = data['tahun_masuk'] - data['tanggal_lahir'].dt.year
#lama inap
data['lama_inap'] = (data['tanggal_pasien_keluar_rs'] - data['tanggal_pasien_masuk_rs']).dt.days
data['lama_inap'] = data['lama_inap'].fillna(0) #kalo rawat inap
#tenure
data['usia_polis_hari'] = (data['tanggal_pasien_masuk_rs'] - data['tanggal_efektif_polis']).dt.days
print(data.isnull().sum())
print(data.columns)
data['log_nominal_klaim'] = np.log1p(data['nominal_klaim_yang_disetujui'].fillna(0))
data['log_biaya_rs'] = np.log1p(data['nominal_biaya_rs_yang_terjadi'].fillna(0))

data['tahun_bulan'] = data['tanggal_pasien_masuk_rs'].dt.to_period('M')
data_bulanan = data.groupby('tahun_bulan').agg(
    total_claim = ('nominal_klaim_yang_disetujui', sum),
    claim_frequency = ('claim_id', 'count'),
    claim_severity = ('nominal_klaim_yang_disetujui', 'mean')
).reset_index()
#if freq is 0
data_bulanan['claim_severity'] = data_bulanan['claim_severity'].fillna(0)

doLoop = True
while doLoop:
    print("Pilih Hubungan Yang Ingin Dilihat")
    print("===-----------====------------===")
    print("1. Hubungan Gender Dengan Severitas Klaim")
    print("2. Hubungan Usia Dengan Severitas Klaim")
    print("3. Jumlah Nasabah Berdasarkan Plan Code")
    print("4. Heatmap Korelasi Faktor Nasabah vs Klaim")
    print("5. Distribusi biaya klaim asuransi kesehatan")
    print("0. Keluar")
    option = input("Enter your option (0/1/2/3/4/5): ").strip()

    if option == '1':
        # boxplot gender dgn severity
        plt.figure(figsize=(8,5))
        sns.boxplot(
            x='gender',
            y='log_nominal_klaim',
            data=data[data['nominal_klaim_yang_disetujui'] > 0]
        )
        plt.title('Hubungan Gender Dengan Severitas Klaim')
        plt.show()

    elif option == '2':
        # scatter plot usia dengan nominal klaim
        plt.figure(figsize=(8,5))
        sns.scatterplot(
            x='usia',
            y='log_nominal_klaim',
            data=data[data['nominal_klaim_yang_disetujui'] > 0],
            alpha=0.5
        )
        plt.title('Hubungan Usia Dengan Severitas Klaim')
        plt.show()

    elif option == '3':
        # bar plot freq klaim per plan code
        plt.figure(figsize=(10, 5))
        data['plan_code'].value_counts().plot(kind='bar', color='orange')
        plt.title('Jumlah Nasabah Berdasarkan Plan Code')
        plt.show()

    elif option == '4':
        # Mengambil 1 huruf pertama dari ICD Diagnosis sebagai Kategori Penyakit
        data['kategori_penyakit'] = data['icd_diagnosis'].astype(str).str[0]
        data['kategori_penyakit'] = data['kategori_penyakit'].replace('n', 'Sehat')
        kolom_teks = [
            'gender', 'domisili', 'plan_code',
            'kategori_penyakit', 'reimburse/cashless',
            'inpatient/outpatient'
        ]
        for col in kolom_teks:
            data[col] = data[col].fillna('Unknown')
            data[col + '_encoded'] = le.fit_transform(data[col])

        data_nasabah = data.groupby('nomor_polis').agg(
            usia=('usia', 'max'),
            gender_encoded=('gender_encoded', 'max'),
            domisili_encoded=('domisili_encoded', 'max'),
            plan_code_encoded=('plan_code_encoded', 'max'),
            total_klaim=('nominal_klaim_yang_disetujui', 'sum'),
            frekuensi_klaim=('claim_id', 'count'),
            rata_rata_lama_inap=('lama_inap', 'mean')
        ).reset_index()

        data_nasabah['rata_rata_lama_inap'] = data_nasabah['rata_rata_lama_inap'].fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            data_nasabah.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm',
            fmt=".2f"
        )
        plt.title("Heatmap Korelasi Faktor Nasabah vs Klaim")
        plt.show()

    elif option == '5':
        plt.figure(figsize=(12, 6))
        plt.hist(
            data['nominal_klaim_yang_disetujui'],
            bins=30,
            color='blue',
            alpha=0.7
        )
        plt.yscale('log')
        mean_val = data['nominal_klaim_yang_disetujui'].mean()
        median_val = data['nominal_klaim_yang_disetujui'].median()
        plt.axvline(mean_val, color='red', label=f'rata-rata : {mean_val:.2f}')
        plt.axvline(median_val, color='green', label=f'median : {median_val:.2f}')
        plt.title("Distribusi biaya klaim asuransi kesehatan")
        plt.xlabel("charges")
        plt.ylabel("freq")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    elif option == '0':
        print("Keluar dari menu.")
        doLoop = False

    else:
        print("Pilihan tidak valid, coba lagi.")
        print("")
