import pandas as pd
import seaborn as sns
import matplotlib as plt

#masukin data
policy_df = pd.read_csv(r'E:\codes\goofing\Data_Polis.csv')
claims_df = pd.read_csv(r'E:\codes\goofing\Data_Klaim.csv')

#bersihin data
policy_df.columns = policy_df.columns.str.strip().str.lower().str.replace(' ', '_')
claims_df.columns = claims_df.columns.str.strip().str.lower().str.replace(' ', '_')

try:
    data = pd.merge(claims_df, policy_df, on='nomor_polis', how='left')
    print(data.head())
    
    print('missing values:')
    print(data.isnull().sum())
except Exception as e:
    print(f"merge failed because {e}")
    
# Convert a column to 'datetime' objects
data['tanggal_pembayaran'] = pd.to_datetime(data['tanggal_pembayaran_klaim'])
print(f"Data ranges from {data['tanggal_pembayaran'].min()} to {data['tanggal_pembayaran'].max()}")
#isi data kosong
most_common_gender = data['gender'].mode()[0]
data['gender'] = data['gender'].fillna(most_common_gender)
data['nominal_klaim_yang_disetujui'] = data['nominal_klaim_yang_disetujui'].fillna(0)
#RUSAK
# plt.figure(figsize=(10,6))
# sns.histplot(data['nominal_klaim_yang_disetujui'], bins=50, kde=True)
# plt.title('Distribution of approved claim amounts')
# plt.show()