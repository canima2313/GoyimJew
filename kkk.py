import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

except Exception as e: #masa gagal si # wokwokwkowok
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
data_bulanan['tahun_bulan'] = data_bulanan['tahun_bulan'].astype(str)
#if freq is 0
data_bulanan['claim_severity'] = data_bulanan['claim_severity'].fillna(0)
output_list = []

for _, row in data_bulanan.iterrows():
    bulan = row['tahun_bulan']

    output_list.append([f"{bulan}_Claim_Frequency", row['claim_frequency']])
    output_list.append([f"{bulan}_Claim_Severity", row['claim_severity']])
    output_list.append([f"{bulan}_Total_Claim", row['total_claim']])

doLoop = True
while doLoop:
    print("Pilih Hubungan Yang Ingin Dilihat")
    print("===-----------====------------===")
    print("1. Hubungan Gender Dengan Severitas Klaim")
    print("2. Hubungan Usia Dengan Severitas Klaim")
    print("3. Jumlah Nasabah Berdasarkan Plan Code")
    print("4. Heatmap Korelasi Faktor Nasabah vs Klaim")
    print("5. Distribusi biaya klaim asuransi kesehatan")
    print("6. Prediksi Nilai Klaim (Severity)")
    print("0. Keluar")
    option = input("Enter your option (1/2/3/4/5/6/0): ").strip()


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
    
    elif option == '6':
        
        kolom_teks = [
            'gender', 'domisili', 'plan_code',
            'kategori_penyakit', 'reimburse/cashless',
            'inpatient/outpatient'
        ]

        #mastiin ada ya boy datanya
        if 'kategori_penyakit' not in data.columns:
            data['kategori_penyakit'] = data['icd_diagnosis'].astype(str).str[0]
            data['kategori_penyakit'] = data['kategori_penyakit'].replace('n', 'Sehat')

        for col in kolom_teks:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
                data[col + '_encoded'] = le.fit_transform(data[col])
        #Mastiin data valid
        ml_data = data[data['nominal_klaim_yang_disetujui'] > 0].copy()
        #fitur!!!
        features = ['usia',
                    'lama_inap',
                    'usia_polis_hari',
                    'gender_encoded',
                    'plan_code_encoded',
                    'domisili_encoded']

        #no NANS bro
        ml_data = ml_data.dropna(subset=features)

        X = ml_data[features]
        y = ml_data['log_nominal_klaim'] 

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4 , random_state= 42)      

        print("KITA TRAINNN")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)  

        #Evaluasi
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Buat dataframe dari hasil agregasi
        output_df = pd.DataFrame(output_list, columns=['id', ' value'])

        sample = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))

        output_df = output_df.set_index('id')

        final_submission = sample.copy()

        final_submission = final_submission.merge(
            output_df,
            on='id',
            how='left'
        )

        final_submission.to_csv("submission.csv", index=False)

        print("Submission berhasil dibuat dengan", len(final_submission), "baris")
        
        print("\n----------Evaluasi Model--------")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2 Score): {r2:.4f}")
        print("--------------------------------\n")

        #Visualisasi
        plt.figure(figsize=(10, 5))
        sns.barplot(x=model.feature_importances_, y=features, palette='viridis')
        plt.title('Pentingnya Fitur dalam Memprediksi Nilai Klaim (Feature Importance)')
        plt.xlabel('Tingkat Kepentingan')
        plt.ylabel('Fitur')
        plt.tight_layout()
        plt.show()

        output_df = pd.DataFrame(output_list, columns=['id', 'value'])

        output_path = os.path.join(base_dir, "monthly_claim_summary.csv")
        output_df.to_csv(output_path, index=False, header=True)

        print("File berhasil dibuat:", output_path)

    elif option == '0':
        print("Keluar dari menu.")
        doLoop = False

    else:
        print("Pilihan tidak valid, coba lagi.")
        print("")
