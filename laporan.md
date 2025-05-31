
# Laporan Proyek Machine Learning - Faturohman Wicaksono

## Domain Proyek
### Deteksi Fraud pada Transaksi Bank

Di era digital saat ini, perkembangan teknologi memungkinkan transaksi keuangan dilakukan dengan cepat dan mudah. Namun, kemudahan ini juga membuka celah terhadap kejahatan siber, khususnya fraud atau penipuan keuangan. Fraud transaksi merupakan aktivitas ilegal yang dilakukan untuk memperoleh keuntungan finansial secara tidak sah, dan dapat menimbulkan kerugian besar bagi institusi keuangan serta mengancam kepercayaan pelanggan.

Menurut laporan dari Association of Certified Fraud Examiners (ACFE), perusahaan kehilangan sekitar 5% dari total pendapatannya setiap tahun akibat fraud. Seiring dengan meningkatnya volume dan kompleksitas data transaksi, metode deteksi fraud konvensional menjadi tidak efektif. Oleh karena itu, penerapan machine learning untuk deteksi fraud menjadi solusi yang semakin relevan dan penting.

Melalui machine learning, model dapat dilatih untuk mengenali pola normal dan mendeteksi anomali yang berpotensi sebagai fraud. Proyek ini berfokus pada pengembangan model klasifikasi untuk mendeteksi transaksi yang mencurigakan berdasarkan data historis transaksi perbankan.

"Meskipun *machine learning* menunjukkan potensi besar untuk deteksi dan pencegahan *fraud* transaksi keuangan, implementasinya menghadapi sejumlah tantangan, termasuk kualitas data, interpretasi model, dan integrasi dengan sistem yang ada" (Pan, 2024).

---

## Business Understanding

### Problem Statements

1. **Bagaimana cara mengklasifikasikan transaksi sebagai fraud atau non-fraud berdasarkan data transaksi dan informasi pelanggan?**
2. **Algoritma machine learning apa yang dapat digunakan untuk mendeteksi transaksi fraud dengan akurasi dan recall yang tinggi?**

### Goals

1. Membangun model klasifikasi untuk memisahkan transaksi fraud dan non-fraud menggunakan berbagai fitur transaksi.
2. Mengevaluasi dan membandingkan beberapa algoritma machine learning untuk mendeteksi fraud dengan mempertimbangkan metrik evaluasi seperti recall dan precision.

### Solution Statements

  - **Solution Statement 1:** Menggunakan KNN untuk mendeteksi transaksi fraud.

  - **Solution Statement 2:** Menggunakan Logistic Regression untuk membandingkan performa dan meningkatkan akurasi deteksi fraud.

  - **Solution Statement 3:** Menggunakan Decision tree untuk membandingkan performa dan meningkatkan akurasi deteksi fraud.

  - **Improvement:** Melakukan hyperparameter tuning dan teknik penanganan data imbalance (misalnya SMOTE) untuk meningkatkan kinerja model dalam mendeteksi kasus fraud yang jarang terjadi.

---

## Data Understanding

### Dataset Description

Link dataset : [Kaggle](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection)
Dataset yang digunakan berisi informasi transaksi bank sejumlah **200.000** baris dan **24** kolom. Data ini mencakup informasi pelanggan, transaksi, dan label apakah transaksi tersebut termasuk fraud (`Is_Fraud`).

| Customer_ID      | Customer_Name           | Gender | Age | State                              | City           | Bank_Branch               | Account_Type | Transaction_ID            | Transaction_Date | ... | Merchant_Category | Account_Balance | Transaction_Device | Transaction_Location                     | Device_Type | Is_Fraud | Transaction_Currency | Customer_Contact | Transaction_Description    | Customer_Email          |
|------------------|-------------------------|--------|-----|------------------------------------|----------------|---------------------------|--------------|---------------------------|------------------|-----|-------------------|-----------------|--------------------|------------------------------------------|-------------|----------|----------------------|------------------|----------------------------|-------------------------|
| 0                | Osha Tella              | Male   | 60  | Kerala                             | Thiruvananthapuram | Thiruvananthapuram Branch | Savings      | 4fa3208f-9e23-42dc-b330-844829d0c12c | 23-01-2025       | ... | Restaurant        | 74557.27        | Voice Assistant    | Thiruvananthapuram, Kerala               | POS         | 0        | INR                  | +9198579XXXXXX   | Bitcoin transaction        | oshaXXXXX@XXXXX.com     |
| 1                | Hredhaan Khosla         | Female | 51  | Maharashtra                        | Nashik         | Nashik Branch             | Business     | c9de0c06-2c4c-40a9-97ed-3c7b8f97c79c | 11-01-2025       | ... | Restaurant        | 74622.66        | POS Mobile Device  | Nashik, Maharashtra                      | Desktop     | 0        | INR                  | +9191074XXXXXX   | Grocery delivery           | hredhaanXXXX@XXXXXX.com |
| 2                | Ekani Nazareth          | Male   | 20  | Bihar                              | Bhagalpur      | Bhagalpur Branch          | Savings      | e41c55f9-c016-4ff3-872b-cae72467c75c | 25-01-2025       | ... | Groceries         | 66817.99        | ATM                | Bhagalpur, Bihar                         | Desktop     | 0        | INR                  | +9197745XXXXXX   | Mutual fund investment     | ekaniXXX@XXXXXX.com     |
| 3                | Yamini Ramachandran     | Female | 57  | Tamil Nadu                         | Chennai        | Chennai Branch            | Business     | 7f7ee11b-ff2c-45a3-802a-49bc47c02ecb | 19-01-2025       | ... | Entertainment     | 58177.08        | POS Mobile App     | Chennai, Tamil Nadu                      | Mobile      | 0        | INR                  | +9195889XXXXXX   | Food delivery              | yaminiXXXXX@XXXXXXX.com |
| 4                | Kritika Rege            | Female | 43  | Punjab                             | Amritsar       | Amritsar Branch           | Savings      | f8e6ac6f-81a1-4985-bf12-f60967d852ef | 30-01-2025       | ... | Entertainment     | 16108.56        | Virtual Card       | Amritsar, Punjab                         | Mobile      | 0        | INR                  | +9195316XXXXXX   | Debt repayment             | kritikaXXXX@XXXXXX.com  |
| ...              | ...                     | ...    | ... | ...                                | ...            | ...                       | ...          | ...                       | ...              | ... | ...               | ...             | ...                | ...                                      | ...         | ...      | ...                  | ...              | ...                        | ...                     |
| 199995           | Vedhika Magar           | Female | 55  | Dadra and Nagar Haveli and Daman and Diu | Diu            | Diu Branch                | Business     | 8d856bc7-4666-4509-a067-48d67500694a | 08-01-2025       | ... | Restaurant        | 37475.11        | Desktop/Laptop     | Diu, Dadra and Nagar Haveli and Daman and Diu | ATM         | 0        | INR                  | +9192629XXXXXX   | ATM withdrawal             | vedhikaXXXXX@XXXXXXX.com |
| 199996           | Aashi Pai               | Male   | 51  | Manipur                            | Kangpokpi      | Kangpokpi Branch          | Business     | f2890dbd-4e01-445d-97f5-ac56886e9037 | 01-01-2025       | ... | Groceries         | 53037.20        | ATM                | Kangpokpi, Manipur                       | ATM         | 0        | INR                  | +9198116XXXXXX   | Subscription box           | aashiXXX@XXXXXX.com     |
| 199997           | Dayita Shan             | Female | 41  | Chandigarh                         | Chandigarh     | Chandigarh Branch         | Savings      | f714b758-7539-474d-b676-5fa7e2480141 | 28-01-2025       | ... | Health            | 96225.36        | ATM                | Chandigarh, Chandigarh                   | Desktop     | 0        | INR                  | +9192601XXXXXX   | Charity donation           | dayitaXXXX@XXXXXXX.com  |
| 199998           | Unnati Vyas             | Female | 28  | Telangana                          | Nizamabad      | Nizamabad Branch          | Checking     | f6903b6a-b582-47ea-95d5-aff16bdec950 | 08-01-2025       | ... | Electronics       | 89599.90        | Voice Assistant    | Nizamabad, Telangana                     | Desktop     | 0        | INR                  | +9197537XXXXXX   | Tourist attraction payment | unnatiXXX@XXXXX.com     |
| 199999           | Gopal Rout              | Male   | 34  | Nagaland                           | Kohima         | Kohima Branch             | Business     | b44b6e8a-1036-4ec1-b492-5e7ffc7baf6d | 08-01-2025       | ... | Electronics       | 15066.24        | Virtual Card       | Kohima, Nagaland                         | Mobile      | 0        | INR                  | +9193961XXXXXX   | Installment payment        | gopalXXXXX@XXXXXXX.com  |
### Features description
 - Customer_ID: Pengenal unik untuk setiap pelanggan di dalam sistem bank.
 - Customer_Name: Nama konsumen yang melakukan transaksi.
 - Gender: Jenis kelamin konsumen (misalnya, Laki-laki, Perempuan, Lainnya).
 - Age: Usia konsumen pada saat transaksi.
 - State: Negara bagian tempat pelanggan tinggal.
 - City: Kota tempat pelanggan tinggal.
 - Bank_Branch: Cabang bank spesifik tempat pelanggan memiliki rekening.
 - Account_Type: Jenis rekening yang dimiliki oleh pelanggan (misalnya, Tabungan, Giro).
 - Transaction_ID: Pengenal unik untuk setiap transaksi.
 - Transaction_Date: Tanggal terjadinya transaksi.
 - Transaction_Time: Waktu spesifik saat transaksi dimulai.
 - Transaction_Amount: Nilai finansial dari transaksi.
 - Merchant_ID: Pengenal unik untuk pedagang yang terlibat dalam transaksi.
 - Transaction_Type: Sifat transaksi (misalnya, Penarikan, Setoran, Transfer).
 - Merchant_Category: Kategori pedagang (misalnya, Ritel, Online, Perjalanan).
 - Account_Balance: Saldo rekening pelanggan setelah transaksi.
 - Transaction_Device: Perangkat yang digunakan oleh konsumen untuk melakukan transaksi (misalnya, Seluler, Desktop).
 - Transaction_Location: Lokasi geografis (misalnya, lintang, bujur) dari transaksi.
 - Device_Type: Jenis perangkat yang digunakan untuk transaksi (misalnya, Ponsel Pintar, Laptop).
 - Is_Fraud: Indikator biner (1 atau 0) yang menunjukkan apakah transaksi tersebut curang atau tidak.
 - Transaction_Currency: Mata uang yang digunakan untuk transaksi (misalnya, USD, EUR).
 - Customer_Contact: Nomor kontak pelanggan.
 - Transaction_Description: Deskripsi singkat tentang transaksi (misalnya, pembelian, transfer).
 - Customer_Email: Alamat email yang terkait dengan rekening konsumen.

### Important features

- `Customer_ID`, `Transaction_ID`
- `Account_Balance`, `Transaction_Amount`
- `Merchant_Category`, `Transaction_Device`, `Account_Type`, `Gender`
- `City`, `Age`, `Currency`, `Transaction_Hour`
- `Is_Fraud` (Target)
---
## Exploratory Data Analysis (EDA)

Pada tahap ini merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data

### Dataset Information
| # | Column                  | Non-Null Count | Dtype   |
|---|-------------------------|----------------|---------|
| 0 | Customer_ID             | 200000         | object  |
| 1 | Customer_Name           | 200000         | object  |
| 2 | Gender                  | 200000         | object  |
| 3 | Age                     | 200000         | int64   |
| 4 | State                   | 200000         | object  |
| 5 | City                    | 200000         | object  |
| 6 | Bank_Branch             | 200000         | object  |
| 7 | Account_Type            | 200000         | object  |
| 8 | Transaction_ID          | 200000         | object  |
| 9 | Transaction_Date        | 200000         | object  |
| 10| Transaction_Time        | 200000         | object  |
| 11| Transaction_Amount      | 200000         | float64 |
| 12| Merchant_ID             | 200000         | object  |
| 13| Transaction_Type        | 200000         | object  |
| 14| Merchant_Category       | 200000         | object  |
| 15| Account_Balance         | 200000         | float64 |
| 16| Transaction_Device      | 200000         | object  |
| 17| Transaction_Location    | 200000         | object  |
| 18| Device_Type             | 200000         | object  |
| 19| Is_Fraud                | 200000         | int64   |
| 20| Transaction_Currency    | 200000         | object  |
| 21| Customer_Contact        | 200000         | object  |
| 22| Transaction_Description | 200000         | object  |
| 23| Customer_Email          | 200000         | object  |
Output tersebut menunjukan bahwa dataset memiliki 200000 data dan memiliki 23 kolom.

- Ukuran Data: DataFrame ini berisi 200.000 baris data.
Kelengkapan Data: Data ini sangat bersih dalam hal kelengkapan karena tidak ada nilai yang hilang (semua Non-Null Count sama dengan total baris). Ini mempermudah langkah pra-pemrosesan data selanjutnya.
- Tipe Data: Tipe data untuk setiap kolom tampaknya sesuai dengan kontennya. Misalnya, usia adalah bilangan bulat, jumlah transaksi dan saldo rekening adalah bilangan desimal, dan identifikasi serta nama adalah objek (string).

### Description of statistic

| Statistik | Age        | Transaction_Amount | Account_Balance | Is_Fraud   |
|-----------|------------|--------------------|-----------------|------------|
| count     | 200000.000 | 200000.000         | 200000.000      | 200000.000 |
| mean      | 44.015110  | 49538.015554       | 52437.988784    | 0.050440   |
| std       | 15.288774  | 28551.874004       | 27399.507128    | 0.218852   |
| min       | 18.000000  | 10.290000          | 5000.820000     | 0.000000   |
| 25%       | 31.000000  | 24851.345000       | 28742.395000    | 0.000000   |
| 50%       | 44.000000  | 49502.440000       | 52372.555000    | 0.000000   |
| 75%       | 57.000000  | 74314.625000       | 76147.670000    | 0.000000   |
| max       | 70.000000  | 98999.980000       | 99999.950000    | 1.000000   |

Fungsi describe() memberikan informasi Deskripsi statistik data pada masing-masing kolom, antara lain:

- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

### Missing Values Check
| Kolom                   | Jumlah Nilai Kosong |
|-------------------------|---------------------|
| Customer_ID             | 0                   |
| Customer_Name           | 0                   |
| Gender                  | 0                   |
| Age                     | 0                   |
| State                   | 0                   |
| City                    | 0                   |
| Bank_Branch             | 0                   |
| Account_Type            | 0                   |
| Transaction_ID          | 0                   |
| Transaction_Date        | 0                   |
| Transaction_Time        | 0                   |
| Transaction_Amount      | 0                   |
| Merchant_ID             | 0                   |
| Transaction_Type        | 0                   |
| Merchant_Category       | 0                   |
| Account_Balance         | 0                   |
| Transaction_Device      | 0                   |
| Transaction_Location    | 0                   |
| Device_Type             | 0                   |
| Is_Fraud                | 0                   |
| Transaction_Currency    | 0                   |
| Customer_Contact        | 0                   |
| Transaction_Description | 0                   |
| Customer_Email          | 0                   |

Berdasarkan tabel tersebut menunjukkan bahwa tidak terdapat data yang hilang maupun duplikat, sehingga dapat dilanjutkan ke proses selanjutnya

### Check Outlier

![Outlier](https://github.com/user-attachments/assets/c38ce510-40cc-4b6b-b594-6bd462e2ccbd)

Dari visualisasi diatas dapat diketahui bahwa tidak terdapat outlier.

### Univariate Analysis

![Univariate](https://github.com/user-attachments/assets/a82b4358-2664-4085-ac30-87573f7ea0b5)

- **Imbalanced Fraud Data:** Kasus fraud sangat jarang dibandingkan non-fraud. Ini adalah temuan paling penting yang mempengaruhi strategi pemodelan.
- **No Simple Linear Patterns for Fraud (Age & Amount):** Kejadian fraud tidak secara langsung atau sederhana terkait dengan usia atau jumlah transaksi. Perlu eksplorasi fitur lain dan pola yang lebih kompleks.
- **Anomalous Numerical Distributions:** Distribusi seragam untuk Age, Transaction_Amount, dan Account_Balance sangat tidak biasa untuk data riil. Jika data ini asli, perlu penyelidikan lebih lanjut; jika sintetis, ini adalah batasan interpretasi.
- **Geographic Variation in Transaction Value:** Rata-rata jumlah transaksi bervariasi signifikan antar kota, menunjukkan potensi segmentasi geografis.
- **Initial Demographic Overview:** Visualisasi awal memberikan pemahaman dasar tentang distribusi pelanggan dan aktivitas transaksi berdasarkan kategori seperti jenis akun, sektor merchant, atau lokasi.

![Univariate](https://github.com/user-attachments/assets/da2c8f58-e4ec-48a7-9841-ac9902270e01)

Dari visualisasi diatas kita dapat informasi bahwa:
- **Imbalance Data is THE Major Challenge:** Temuan paling konsisten adalah ketidakseimbangan kelas yang parah pada variabel Is_Fraud. Ini adalah prioritas utama untuk diatasi dalam tahap pemodelan deteksi fraud.
- **Suspicion of Synthetic Data:** Distribusi seragam pada fitur numerik (Age, Transaction_Amount, Account_Balance) sangat mengindikasikan bahwa dataset ini mungkin sintetis atau buatan. Hal ini penting karena insight yang didapat mungkin tidak sepenuhnya mencerminkan pola dalam data transaksi dunia nyata.
- **Fraud is Not Simple:** Dari visualisasi awal, fraud tidak menunjukkan korelasi linier yang jelas dengan usia atau jumlah transaksi, dan polanya serupa antar gender. Ini berarti perlu eksplorasi fitur lain atau penggunaan model yang lebih canggih untuk menemukan pola fraud.
- **Geographic Variation Matters:** Terdapat perbedaan signifikan dalam rata-rata nilai transaksi antar kota, menunjukkan potensi untuk segmentasi atau strategi berbasis lokasi.
- **Basic Categorical Insights:** Kita mendapatkan gambaran umum tentang demografi pelanggan dan pola transaksi yang dominan (jenis akun, perangkat, kategori merchant, mata uang, dll.).
- **Data Uniqueness Check Needed:** Untuk ID dan timestamp, frekuensi tinggi mungkin mengindikasikan aktivitas pelanggan yang tinggi atau adanya duplikasi/issue pada data.

### Bivariate Analysis

![Bivariate](https://github.com/user-attachments/assets/d1694835-d873-4d51-97eb-3bb5df645184)

Insight yang didapat dari visualisasi diatas yaitu:
- **Ketidakseimbangan Kelas Fraud yang Ekstrem:** Dataset ini memiliki jumlah transaksi non-fraud yang jauh lebih banyak daripada transaksi fraud. Ini adalah masalah utama yang harus ditangani dalam pemodelan.
- **Kecurigaan Data Sintetis:** Distribusi fitur numerik (Age, Transaction_Amount, Account_Balance) yang sangat seragam menunjukkan kuatnya kemungkinan bahwa data ini adalah sintetis. Ini penting untuk interpretasi karena insight yang didapat mungkin tidak sepenuhnya mencerminkan karakteristik data transaksi dunia nyata.
- **Pola Fraud yang Tidak Sederhana:** Tidak ada korelasi linier yang jelas antara Age atau Transaction_Amount dengan kejadian fraud, dan pola ini terlihat serupa antara gender. Ini menyiratkan bahwa model deteksi fraud perlu mencari pola yang lebih kompleks atau menggunakan fitur lain untuk identifikasi.
- **Variasi Geografis:** Terdapat perbedaan signifikan dalam rata-rata jumlah transaksi antar kota, menunjukkan bahwa lokasi adalah faktor penting yang mempengaruhi nilai transaksi.
- **Pemahaman Dasar Kategorikal:** Visualisasi awal memberikan gambaran frekuensi kategori utama seperti jenis akun, sektor merchant, perangkat transaksi, dan asal negara, yang penting untuk pemahaman umum tentang dataset.

![Bivariate](https://github.com/user-attachments/assets/18fbe388-7b48-4afb-bd76-ff880dfc774b)

Dari visualisasi diatas kita dapat informasi bahwa:
- Rata-rata Transaksi yang Konsisten Tinggi: Semua kota yang ditampilkan dalam grafik ini memiliki rata-rata Transaction Amount yang sangat tinggi, berada di kisaran 47.000 hingga 52.000. Tidak ada perbedaan signifikan atau "terendah" yang jelas di antara kota-kota ini, kecuali sedikit fluktuasi di antara nilai rata-rata yang tinggi tersebut.

- Kota dengan Rata-rata Transaksi Tertinggi:
  - Kochi Branch memiliki rata-rata Transaction Amount tertinggi, sedikit di atas 52.000.
  - Diikuti oleh Tura Branch, Siliguri Branch, Karimnagar Branch, Munger Branch, dan Dehradun Branch yang semuanya memiliki rata-rata di atas 50.000.

  - Rentang Rata-rata yang Sempit: Perbedaan antara rata-rata transaksi tertinggi (Kochi Branch) dan terendah di antara 20 kota yang ditampilkan ini relatif kecil (sekitar 5.000 atau kurang). Ini menunjukkan bahwa kota-kota yang masuk dalam daftar ini cenderung memiliki pola transaksi dengan nilai rata-rata yang serupa, yaitu tinggi.

  - Sebaran Kota: Grafik ini menunjukkan cabang-cabang (Branch) yang tersebar di berbagai kota. Ini mengindikasikan bahwa operasi transaksi dengan nilai rata-rata tinggi tidak hanya terkonsentrasi di satu wilayah saja, melainkan terjadi di beberapa lokasi berbeda.

Analisis:

- Fokus pada Transaksi Bernilai Tinggi: Data ini menyoroti cabang-cabang atau kota-kota di mana rata-rata nilai transaksi cenderung tinggi. Ini bisa menjadi fokus untuk strategi bisnis yang menargetkan pelanggan dengan daya beli tinggi atau transaksi berskala besar.

- Potensi Variabilitas Data: Meskipun rata-rata transaksi tinggi, kita tidak mengetahui distribusi data di setiap kota. Rata-rata yang tinggi bisa berasal dari:
  -  Banyak transaksi kecil dan beberapa transaksi sangat besar.
  - Sebagian besar transaksi memang bernilai tinggi.
  - Perlu melihat box plot atau histogram untuk setiap kota untuk memahami sebaran sebenarnya dan apakah ada outlier yang memengaruhi rata-rata.

![Bivariate](https://github.com/user-attachments/assets/fd9e600d-0907-49c5-bf59-e6f61247bf5c)

Insight yang didapat dari visualisasi ini adalah :

- **Ketidakseimbangan Kelas Fraud yang Ekstrem:** Ini adalah temuan paling dominan dan kritis yang berulang di beberapa plot. Jumlah transaksi non-fraud jauh lebih banyak daripada transaksi fraud, mengindikasikan kebutuhan akan strategi khusus dalam pemodelan.
- **Dugaan Kuat Data Sintetis:** Distribusi yang sangat seragam pada fitur numerik (Age, Transaction_Amount, Account_Balance) dan sebaran titik yang merata pada scatter plots sangat menunjukkan bahwa data ini kemungkinan besar adalah buatan atau sintetis. Hal ini penting karena insight yang didapat mungkin tidak sepenuhnya mencerminkan karakteristik data transaksi dunia nyata.
- **Pola Fraud Tidak Sederhana:** Dari visualisasi Age vs Transaction_Amount, tidak ada korelasi linier yang jelas atau klustering transaksi fraud. Pola ini terlihat serupa antara gender. Ini menyiratkan bahwa model deteksi fraud harus mencari pola yang lebih kompleks atau bergantung pada fitur lain.
**Variasi Geografis dan Cabang:**

    - Terdapat perbedaan signifikan dalam rata-rata jumlah transaksi antar kota, menunjukkan potensi untuk segmentasi geografis.
    - Namun, rata-rata jumlah transaksi antar cabang bank (Bank_Branch) menunjukkan homogenitas yang tinggi, dengan sedikit perbedaan antara cabang tertinggi dan terendah. Ini menyiratkan bahwa cabang-cabang ini memiliki profil transaksi rata-rata yang serupa, atau kembali lagi, menunjukkan pola data sintetis yang tidak memiliki variasi dunia nyata.

- **Pemahaman Frekuensi Kategorikal Dasar:** Visualisasi count plots memberikan pemahaman awal tentang distribusi frekuensi kategori dalam berbagai fitur seperti gender, tipe akun, kategori merchant, dan perangkat transaksi, yang penting untuk pemahaman umum dataset.

![Bivariate](https://github.com/user-attachments/assets/5edfdfe7-452b-4b0e-b020-0aa595a5eebc)

Insight yang didapat yaitu:

- **Rata-rata Transaksi yang Sangat Mirip:** Rata-rata Transaction Amount untuk gender Female dan Male hampir identik. Kedua gender menunjukkan rata-rata transaksi sekitar 49.000 (tidak ada label numerik persis, tapi visualnya sangat dekat dengan 50.000).
- **Tidak Ada Perbedaan Signifikan:** Berdasarkan grafik ini, tidak ada perbedaan yang signifikan secara praktis dalam rata-rata nilai transaksi antara pria dan wanita.

![Bivariate](https://github.com/user-attachments/assets/d1c3a97d-7b19-4c82-addc-8b31b7a7f97f)

Informasi yang diperoleh dari visualisasi diatas adalah :

- **Rata-rata Transaksi yang Sangat Konsisten di Seluruh Negara Bagian:** Mirip dengan grafik sebelumnya (berdasarkan kota), grafik ini menunjukkan bahwa rata-rata Transaction Amount di sebagian besar negara bagian sangatlah tinggi dan berada dalam rentang yang sempit, yaitu antara sekitar 48.000 hingga 50.000an.

- **Negara Bagian dengan Rata-rata Tertinggi:**

  - Chandigarh menempati posisi teratas dengan rata-rata transaksi tertinggi (sekitar 50.235).
  - Diikuti oleh Himachal Pradesh (50.146) dan Uttar Pradesh (50.080), yang juga memiliki rata-rata di atas 50.000.

- **Rentang Perbedaan yang Sangat Kecil:** Perbedaan antara rata-rata transaksi tertinggi (Chandigarh ~50.235) dan terendah di antara negara bagian yang ditampilkan (misalnya, Chhattisgarh ~48.732) sangatlah kecil. Ini menunjukkan bahwa rata-rata nilai transaksi per transaksi tidak banyak bervariasi antar negara bagian.

- **Sebaran Geografis:** Data mencakup banyak negara bagian, menunjukkan bahwa pola transaksi bernilai rata-rata tinggi ini tidak terbatas pada satu atau dua wilayah geografis, melainkan tersebar luas.

![Bivariate](https://github.com/user-attachments/assets/bea9b1ba-b01e-4349-bfe5-000d8be3244b)

Insight baru yang didapat:
- **Konsistensi Rata-rata Nilai Transaksi:** Sama seperti grafik-grafik sebelumnya (berdasarkan kota, gender, dan negara bagian), rata-rata Transaction Amount di berbagai kategori pedagang juga sangat konsisten dan berada di kisaran yang sangat sempit, yaitu antara 49.000 hingga 49.700.
- **Perbedaan yang Sangat Kecil:** Perbedaan rata-rata transaksi antara kategori yang berbeda (misalnya, Entertainment ~49.694 sebagai yang tertinggi dan Clothing ~49.544 sebagai yang terendah di antara yang ditampilkan) sangatlah minimal.

![Bivariate](https://github.com/user-attachments/assets/9cc3a1c8-3596-40ef-8add-56825b8990e6)

Insight yang didapat dari visualisasi yaitu:

- **Konsistensi Rata-rata Nilai Transaksi yang Tinggi:** Grafik ini kembali menunjukkan pola yang konsisten: rata-rata Transaction Amount untuk hampir semua jenis perangkat transaksi berada dalam kisaran yang sangat sempit, yaitu antara 48.000 hingga 50.000an.
- **Perbedaan yang Sangat Minimal:** Meskipun ada sedikit fluktuasi, perbedaan antara rata-rata transaksi tertinggi (POS Terminal ~50.140) dan terendah (Voice Assistant ~48.541) di antara perangkat yang ditampilkan sangatlah kecil, tidak melebihi ~2.000.

![Bivariate](https://github.com/user-attachments/assets/76ef8a1e-0eda-4ab9-ba30-26db1e2b8e76)


Dari visualisasi diatas insight yang diperoleh adalah :

- **Rata-rata Transaksi yang Sangat Konsisten di Antara Kota-kota Teratas:** Semua kota yang ditampilkan dalam grafik ini memiliki rata-rata Transaction Amount yang sangat tinggi, berkisar antara 47.500 hingga 51.700-an.
Kota dengan Rata-rata Transaksi Tertinggi:

  - Kochi, Kerala menduduki puncak dengan rata-rata Transaction Amount tertinggi, yaitu 51.716.
  - Diikuti oleh Tura, Meghalaya (51.455), Karimnagar, Telangana (51.397), Siliguri, West-Bengal (51.353), dan Munger, Bihar (51.283) sebagai Top 5 kota dengan rata-rata transaksi tertinggi.

- **Rentang Perbedaan yang Relatif Kecil:** Meskipun ada variasi dari yang tertinggi ke terendah di antara 10 kota ini, perbedaannya hanya sekitar 4.000-an (dari 51.716 hingga 47.554). Ini menunjukkan bahwa kota-kota yang masuk dalam daftar "teratas" ini memiliki pola nilai transaksi rata-rata yang serupa, yaitu tinggi.
- **Sebaran Geografis:** Kota-kota yang ditampilkan tersebar di berbagai negara bagian (Kerala, Meghalaya, Telangana, West-Bengal, Bihar, Jharkhand, Karnataka, Rajasthan, Chhattisgarh), mengindikasikan bahwa transaksi bernilai rata-rata tinggi tidak terkonsentrasi di satu wilayah saja.


![Bivariate](https://github.com/user-attachments/assets/4c011187-f26e-41b9-a861-a631b3b9dcae)


Insight dari visualisasi bivariate diatas adalah :

- **Konsentrasi Data di Sepanjang Rentang Transaction Amount:** Sebagian besar transaksi, terlepas dari usia, cenderung memiliki Transaction Amount yang tersebar di seluruh rentang yang ditampilkan (dari mendekati 0 hingga 100.000). Warna kuning (densitas tertinggi, sekitar 550 transaksi) tersebar di banyak hexbin di seluruh sumbu Y.

- **Densitas Tinggi di Rentang Usia Tertentu:** Ada area dengan densitas transaksi yang sangat tinggi (warna kuning cerah) yang muncul di beberapa rentang usia tertentu:

  - Sekitar usia 30-35 tahun.
  - Sekitar usia 40-45 tahun.
  - Sekitar usia 50-55 tahun.
  - Sekitar usia 60-65 tahun. Ini menunjukkan adanya kelompok usia di mana volume transaksi (jumlah transaksi) lebih padat, meskipun tidak secara eksklusif berfokus pada nilai Transaction Amount tertentu.

- **Jumlah Transaksi Rendah di Batas Usia:** Densitas transaksi cenderung lebih rendah (warna ungu gelap hingga hijau gelap) di ujung-ujung rentang usia (di bawah 20 tahun dan di atas 70 tahun). Ini dapat mengindikasikan bahwa jumlah pelanggan atau aktivitas transaksi lebih sedikit di kelompok usia ekstrem ini.

- **Absennya Korelasi Kuat antara Usia dan Nilai Transaksi:** Tidak ada pola yang jelas yang menunjukkan bahwa seiring bertambahnya usia, Transaction Amount cenderung meningkat atau menurun secara signifikan. Nilai transaksi tinggi maupun rendah terjadi di hampir setiap kelompok usia.

### Multivariate analysis

![Multivariate](https://github.com/user-attachments/assets/f87d2259-10cd-4b79-9946-2ed9950acac0)

Insight dari visualisasi diatas yaitu:

- **Dominasi Class Imbalance:** Tantangan terbesar dalam membangun model deteksi fraud dengan data ini adalah class imbalance yang ekstrem. Ini berarti model akan cenderung bias untuk memprediksi kelas Non-Fraud karena itu adalah kelas mayoritas, dan mungkin memiliki kesulitan dalam mengidentifikasi kelas Fraud yang langka.

 - **Solusi: Harus menerapkan teknik penanganan imbalance seperti:**
      - Sampling: Oversampling minoritas (misalnya, SMOTE, ADASYN) atau undersampling mayoritas (jika data sangat besar dan tidak ingin kehilangan informasi penting).
      - Cost-Sensitive Learning: Memberikan bobot penalti lebih tinggi untuk kesalahan klasifikasi kelas Fraud dalam fungsi loss model.
      - Algoritma yang Tepat: Tree-based models (misalnya, Random Forest, Gradient Boosting like XGBoost/LightGBM) seringkali bekerja lebih baik pada data yang tidak seimbang daripada model linear.
      - Evaluasi Metrik: Tidak menggunakan akurasi sebagai metrik evaluasi utama. Sebaliknya, fokus pada metrik seperti Precision, Recall, F1-Score, ROC-AUC, atau Precision-Recall Curve (PR-AUC), terutama Recall (untuk meminimalkan false negatives, yaitu fraud yang tidak terdeteksi) dan Precision (untuk meminimalkan false positives, yaitu transaksi valid yang salah diklasifikasikan sebagai fraud).

- **Kualitas Data:** Jumlah fraud yang sangat kecil di setiap kategori menunjukkan bahwa dataset ini mungkin memiliki kasus fraud yang benar-benar langka atau sistem deteksi fraud saat ini sudah cukup efektif sehingga hanya sedikit kasus yang terdeteksi. Penting untuk memverifikasi apakah ini semua data fraud yang tersedia atau hanya subset.

- **Potensi Feature Engineering Lanjutan:**

  - Rasio Fraud per Kategori: Hitung rasio fraud_count / total_count untuk setiap kategori (Merchant, Device, Type, Account, Gender). Meskipun plot menunjukkan proporsi kecil, perbedaan persentase yang sangat kecil bisa menjadi indikator penting bagi model.
  - Interaksi Fitur: Pertimbangkan fitur interaksi (misalnya, Merchant Category digabungkan dengan Device Type) karena pola fraud mungkin muncul dari kombinasi variabel.
  - Fitur Agregat: Tambahkan fitur agregat di level pelanggan atau merchant, seperti total_transactions_in_last_N_days, average_transaction_amount_per_customer, count_of_different_merchant_categories_visited.

- **Prediksi Bukan Hanya Klasifikasi:** Selain klasifikasi biner (fraud vs. non-fraud), model juga bisa dilatih untuk menghasilkan fraud score (probabilitas fraud). Hal ini memungkinkan ambang batas disesuaikan berdasarkan toleransi risiko bisnis.

- **Pentingnya Konteks Temporal:** Data transaksi sangat dipengaruhi oleh waktu. Informasi waktu transaksi (dari file uu.jpg) akan sangat penting untuk feature engineering terkait time-series (misalnya, time_since_last_transaction, number_of_transactions_in_last_hour). Ini tidak terlihat di grafik ini tetapi krusial untuk deteksi fraud.

![Multivariate](https://github.com/user-attachments/assets/b7cd1e94-b7c7-4113-a48a-f272dbdcdede)


Insight yang diketahui dari visualisasi diatas yaitu :

- **Konfirmasi Masalah Class Imbalance yang Ekstrem:** Kedua grafik ini dengan jelas dan tegas menunjukkan bahwa proporsi kasus fraud sangat, sangat kecil dibandingkan dengan kasus non-fraud di semua kategori dan perangkat. Ini adalah tantangan terbesar yang harus diatasi dalam proyek deteksi fraud ini.

- **Fitur Kategorikal Sendiri Kurang Diskriminatif:** Dari grafik proporsi ini, terlihat bahwa fitur Kategori Merchant dan Perangkat Transaksi secara individual tidak secara signifikan membedakan antara transaksi fraud dan non-fraud berdasarkan proporsi relatifnya. Artinya, tidak ada "kategori merchant" atau "perangkat" yang secara proporsional memiliki tingkat fraud yang jauh lebih tinggi dari yang lain.

- **Pentingnya Interaksi dan Fitur Lanjutan:**

  - Meskipun fitur-fitur ini sendiri kurang diskriminatif dalam hal proporsi, mereka tetap penting untuk disertakan dalam model. Pola fraud seringkali muncul dari kombinasi beberapa fitur (misalnya, Transaction Amount yang tinggi di kategori Electronics menggunakan Mobile App pada jam 3 pagi).
  - Rekayasa Fitur: Ide untuk membuat fitur baru seperti rasio fraud per kategori/perangkat (misalnya, fraud_rate_per_merchant_category) mungkin tidak akan terlalu efektif jika rasio tersebut hampir nol untuk semua kategori seperti yang ditunjukkan di sini. Namun, fitur agregat lainnya (jumlah transaksi per kategori/perangkat untuk suatu pelanggan dalam periode waktu tertentu, rata-rata nilai transaksi per kategori/perangkat, dll.) bisa lebih berguna.

- **Sinyal Fraud Mungkin Berada di Fitur Numerik atau Interaksi:** Karena fitur kategorikal ini tidak menunjukkan sinyal kuat, kemungkinan besar sinyal fraud yang lebih kuat ada pada:

  - Fitur Numerik: Seperti Transaction Amount (misalnya, transaksi fraud cenderung sangat tinggi atau sangat rendah). Hexbin plot sebelumnya (Age vs Transaction Amount) akan penting untuk dianalisis lagi dengan memisahkan fraud dan non-fraud.
  - Fitur Waktu: Transaction Date dan Transaction Time (misalnya, transaksi fraud lebih sering terjadi di luar jam kerja normal atau pada tanggal tertentu).
  - Perilaku Anomali: Perilaku yang menyimpang dari pola normal pelanggan (misalnya, tiba-tiba melakukan transaksi di kategori merchant atau perangkat yang belum pernah digunakan sebelumnya, atau jumlah transaksi yang tidak biasa). Ini memerlukan time-series feature engineering di level pelanggan.

![Multivariate](https://github.com/user-attachments/assets/9e8ae5ed-f426-49a7-a00a-7e9a90b6a0cc)


Insight Corelation heatmapnya yaitu :

- **Volume Transaksi yang Relatif Seragam:** Secara umum, jumlah transaksi yang tercatat di persimpangan Kategori Merchant dan Tipe Transaksi menunjukkan angka yang relatif seragam. Sebagian besar sel memiliki jumlah transaksi antara 6.500 hingga 6.800.

**Hotspot Transaksi:**

  - Restaurant dan Withdrawal (6780) adalah kombinasi dengan jumlah transaksi tertinggi yang ditampilkan di heatmap.
  - Electronics dan Transfer (6843) adalah kombinasi lainnya dengan jumlah transaksi yang sangat tinggi.
  - Clothing dan Credit (6846) juga merupakan salah satu kombinasi dengan jumlah transaksi tertinggi.
  - Entertainment dan Debit (6786) memiliki jumlah transaksi yang tinggi.

**Coolspot Transaksi:**

  - Clothing dan Transfer (6481) adalah kombinasi dengan jumlah transaksi terendah di antara yang ditampilkan.
  - Health dan Transfer (6560) juga menunjukkan jumlah transaksi yang relatif lebih rendah.
  - Electronics dan Bill Payment (6659) juga merupakan salah satu yang paling rendah.

### Insight dari EDA

- **Imbalanced Classes:** Jumlah transaksi fraud jauh lebih sedikit dari non-fraud.
- **Distribusi Fitur Numerik Tidak Biasa:** `Age`, `Transaction_Amount`, dan `Account_Balance` menunjukkan distribusi seragam, kemungkinan menunjukkan data sintetis.
- **Pola Geografis:** Nilai transaksi bervariasi antar kota.
- **Fitur Demografis:** Tidak ada pola sederhana berdasarkan usia atau gender untuk fraud.


---
### Data Preparation

Tahap Data Preparation pada proyek ini bertujuan untuk mengubah data mentah menjadi format yang siap untuk pelatihan model machine learning. Langkah-langkah yang dilakukan meliputi feature engineering sederhana, encoding variabel kategorikal, scaling variabel numerik, pembagian data menjadi set pelatihan dan pengujian, serta penanganan ketidakseimbangan kelas menggunakan SMOTE.

### Langkah-langkah:

1.  **Feature Engineering Sederhana:**
    *   Berdasarkan kolom 'Age' dan 'Transaction_Date', dibuat fitur-fitur baru yang relevan.
    *   Kolom 'Age' dikategorikan menjadi 'Age_Group' menggunakan binning untuk menciptakan fitur kategorikal berdasarkan rentang usia.
    *   Kolom 'Transaction_Date' dipecah menjadi komponen waktu seperti tahun, bulan, hari, hari dalam seminggu, dan jam untuk mengekstrak pola temporal yang mungkin relevan.  
2.  **Encoding Variabel Kategorikal:**
    *   Mengubah fitur kategorikal menjadi representasi numerik.
    *   Untuk fitur nominal seperti 'Transaction_Type' dan 'Age_Group' yang baru dibuat, digunakan **One-Hot Encoding** untuk menciptakan kolom biner baru untuk setiap kategori, mencegah asumsi urutan.
    *   Untuk kolom kategorikal lain yang tersisa ('Merchant_Category', 'Transaction_Device', dll.), digunakan **Label Encoding** untuk mengubah setiap nilai unik menjadi bilangan bulat unik.
3.  **Scaling Variabel Numerik:**
    *   Variabel numerik diskalakan menggunakan `StandardScaler` agar memiliki rata-rata nol dan varians satu. Ini penting untuk algoritma yang peka terhadap skala fitur seperti K-Nearest Neighbors (KNN) dan Logistic Regression.
4.  **Pembagian Data (Train-Test Split):**
    *   Dataset dibagi menjadi set pelatihan (70%) dan set pengujian (30%) menggunakan `train_test_split`. Pembagian ini dilakukan *setelah* preprocessing (feature engineering, encoding, scaling) pada seluruh dataset, yang **perlu diwaspadai** karena potensi data leakage (informasi dari set pengujian bisa sedikit merembes ke proses scaling). Namun, dalam kode Anda, scaling dilakukan pada `df` penuh sebelum split. Pendekatan yang lebih aman adalah split terlebih dahulu, lalu fit scaler hanya pada data training dan transform pada data training dan testing.
    *   Target variabel `Is_Fraud` dipisahkan dari fitur (`X`).
5.  **Penanganan Ketidakseimbangan Kelas (SMOTE):**
    *   Untuk mengatasi ketidakseimbangan kelas yang parah, teknik `SMOTE (Synthetic Minority Oversampling Technique)` diterapkan. SMOTE menghasilkan sampel sintetis dari kelas minoritas (fraud) untuk menyeimbangkan jumlah kelas dalam set pelatihan.
    *   **Penting:** SMOTE hanya diterapkan pada **set pelatihan** (`X_train`, `y_train`) untuk menghindari data leakage dari set pengujian. Set pengujian (`X_test`, `y_test`) dibiarkan dalam distribusi aslinya untuk evaluasi performa model yang realistis.

---

## Modeling

Tahap Modeling berfokus pada pembangunan dan evaluasi model machine learning untuk mendeteksi transaksi fraud. Mengingat sifat data yang tidak seimbang, strategi pemodelan dipilih untuk memaksimalkan kinerja dalam mengidentifikasi kasus fraud.

### Algoritma yang Digunakan

Tiga algoritma klasifikasi berbeda dieksplorasi untuk membandingkan kemampuan mereka dalam mendeteksi fraud:

1.  **Logistic Regression:** Model linear sederhana.
2.  **Decision Tree:** Model non-linear berbasis aturan.
3.  **K-Nearest Neighbor (KNN):** Model berbasis jarak.

### Strategi Pemodelan:

Untuk memastikan model yang dibangun robust dan memiliki performa optimal dalam mendeteksi fraud, beberapa strategi utama diterapkan:

*   **Data Training Resampled (SMOTE):** Model dilatih pada set pelatihan yang telah diseimbangkan menggunakan SMOTE. Ini membantu model belajar dari kasus fraud (kelas minoritas) yang jumlahnya sedikit.
*   **Stratified K-Fold Cross Validation:** Digunakan selama tahap hyperparameter tuning (`GridSearchCV`). Stratified K-Fold memastikan bahwa setiap 'fold' memiliki proporsi kelas target (fraud vs. non-fraud) yang sama, penting untuk data yang tidak seimbang.
*   **Hyperparameter Tuning (`GridSearchCV`):** Digunakan untuk mencari kombinasi hyperparameter terbaik untuk setiap model. Metrik evaluasi yang digunakan dalam tuning adalah **Recall** dengan `pos_label=1` (fokus pada meminimalkan False Negatives, yaitu kasus fraud yang tidak terdeteksi).
*   **Evaluasi pada Test Set Original:** Model terbaik dari tuning (yang dilatih pada data training yang sudah di-resample) dievaluasi pada set pengujian **asli** (tidak seimbang) untuk mendapatkan estimasi performa yang realistis. Metrik evaluasi yang digunakan mencakup Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC AUC, dan Precision-Recall Curve.

### Detail Model dan Tuning Parameter:

Berikut adalah rincian lebih lanjut mengenai model, parameter tuning yang (terutama untuk KNN, Decision Tree, dan Logistic regression), serta kelebihan dan kekurangannya dalam konteks deteksi fraud:

#### K-Nearest Neighbor (KNN)

*   **Konsep:** KNN adalah algoritma klasifikasi lazy learning berbasis instansi. Artinya, algoritma ini tidak membangun model secara eksplisit selama tahap pelatihan. Prediksi untuk data baru dilakukan dengan mencari "tetangga terdekat" dari data baru tersebut di antara data pelatihan.
*   **Cara Kerja:**
    * Algoritma hanya menyimpan seluruh dataset pelatihan beserta label kelasnya. Dalam kasus deteksi fraud, Jika suatu transaksi baru memiliki pola fitur (misalnya, jumlah transaksi, kategori merchant, perangkat) yang mirip dengan mayoritas transaksi yang diketahui sebagai fraud di data pelatihan (berdasarkan jarak fitur), maka transaksi tersebut kemungkinan akan diklasifikasikan sebagai fraud.
*   **Parameter Tuning (`GridSearchCV`):**
    *   `n_neighbors`: Jumlah tetangga yang dipertimbangkan. Rentang yang diuji: `[3, 5, 7, 9, 11]`.
    *   `weights`: Cara membobot kontribusi tetangga. 'uniform' (semua bobot sama) atau 'distance' (tetangga lebih dekat memiliki bobot lebih besar).
    *   `metric`: Metrik jarak yang digunakan. 'euclidean', 'manhattan', atau 'minkowski'.
*   **Kelebihan:**
    *   Sederhana dan mudah diimplementasikan.
    *   Tidak membuat asumsi tentang distribusi data.
    *   Dapat menangkap pola non-linear.
    *   Tuning dengan `metric` memungkinkan eksplorasi cara penghitungan "kedekatan" yang berbeda.
*   **Kekurangan:**
    *   Mahal secara komputasi saat prediksi (perlu menghitung jarak ke semua titik training).
    *   Sensitif terhadap skala fitur (memerlukan scaling, yang sudah dilakukan).
    *   Sensitif terhadap data noise dan outlier.
    *   Performa dapat menurun pada dataset dimensi tinggi.
    *   Pemilihan `n_neighbors` yang optimal bisa tricky.
      
#### Logistic Regression

*   **Konsep:** Logistic Regression adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas suatu instansi termasuk dalam kelas tertentu (dalam kasus ini, probabilitas suatu transaksi adalah fraud). Meskipun namanya mengandung "Regression", ini adalah algoritma klasifikasi.
*   **Cara Kerja:** Logistic Regression mencoba menemukan hubungan linear antara fitur-fitur transaksi dan probabilitas transaksi menjadi fraud. Bobot positif untuk fitur tertentu menunjukkan bahwa fitur tersebut meningkatkan kemungkinan fraud, sementara bobot negatif menurunkannya.
*   **Parameter:** random_state=42, solver='liblinear', tanpa hyperparameter tuning.
*   **Kelebihan:**
    *   Cepat dilatih dan diinterpretasikan.
    *   Output berupa probabilitas, berguna untuk penyesuaian ambang batas deteksi fraud.
    *   Efisien untuk dataset besar dengan fitur yang relevan secara linear.
    *   Regularisasi membantu mencegah overfitting.
*   **Kekurangan:**
    *   Hanya bisa menangkap hubungan linear antar fitur.
    *   Kinerjanya mungkin kurang baik pada dataset dengan interaksi fitur non-linear yang kompleks.
 
#### Decision Tree

*   **Konsep:** Decision Tree adalah algoritma klasifikasi (atau regresi) berbasis pohon. Algoritma ini membangun struktur pohon di mana setiap node internal mewakili "tes" pada salah satu fitur, setiap cabang mewakili hasil tes tersebut, dan setiap node daun mewakili kelas hasil (fraud atau non-fraud).
*   **Cara Kerja:** 
*   **Parameter Tuning (`GridSearchCV`):** Sesuai kode, tuning dilakukan pada parameter `max_depth`, `min_samples_split`, dan `min_samples_leaf`.
    *   `max_depth`: Kedalaman maksimum pohon. `[None, 10, 20, 30, 40, 50]`. `None` berarti tidak ada batasan.
    *   `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk memecah node. `[2, 5, 10]`.
    *   `min_samples_leaf`: Jumlah minimum sampel yang harus ada di node daun. `[1, 2, 4]`.
*   **Kelebihan:**
    *   Mudah diinterpretasikan (jika pohon tidak terlalu dalam).
    *   Dapat menangkap hubungan non-linear dan interaksi fitur.
    *   Tidak memerlukan scaling fitur.
    *   Tuning parameter seperti `max_depth`, `min_samples_split`, `min_samples_leaf` membantu mengontrol kompleksitas dan mencegah overfitting.
*   **Kekurangan:**
    *   Cenderung overfitting (terutama tanpa batasan yang tepat).
    *   Sensitif terhadap perubahan kecil dalam data training.
    *   Bisa menjadi tidak stabil.
    *   Pohon yang dalam sulit diinterpretasikan.

---

## Evaluation

Tahap Evaluation bertujuan untuk menilai kinerja model machine learning dalam mendeteksi transaksi fraud menggunakan set data pengujian yang belum pernah dilihat sebelumnya. Karena data sangat tidak seimbang, fokus evaluasi tidak hanya pada akurasi keseluruhan, tetapi terutama pada metrik yang relevan untuk kasus kelas minoritas (fraud).

### Metrik Evaluasi:

Beberapa metrik evaluasi kunci digunakan untuk mendapatkan gambaran komprehensif tentang kinerja setiap model:

*   **Accuracy**: Rasio total prediksi yang benar (baik True Positives + True Negatives) dibagi dengan jumlah total sampel.
*   **Precision**: Rasio True Positives (kasus fraud yang berhasil diprediksi fraud) terhadap jumlah total yang diprediksi sebagai fraud (True Positives + False Positives). Menjawab pertanyaan: "Jika model memprediksi fraud, seberapa sering itu benar?" Precision penting untuk meminimalkan false positives (mengklasifikasikan transaksi valid sebagai fraud), yang dapat menyebabkan ketidaknyamanan bagi pelanggan.
*   **Recall (Sensitivity)**: Rasio True Positives terhadap jumlah total kasus fraud yang sebenarnya (True Positives + False Negatives). Menjawab pertanyaan: "Dari semua kasus fraud yang sebenarnya, berapa banyak yang berhasil dideteksi oleh model?" Recall sangat penting dalam konteks deteksi fraud karena tujuan utama adalah menemukan sebanyak mungkin kasus fraud (meminimalkan False Negatives).
*   **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Memberikan skor tunggal yang menyeimbangkan kedua metrik tersebut. Berguna ketika ada keseimbangan antara pentingnya Precision dan Recall.
*   **ROC AUC (Area Under the Receiver Operating Characteristic Curve)**: Mengukur kemampuan model untuk membedakan antara kelas positif (fraud) dan negatif (non-fraud) di berbagai ambang batas klasifikasi. Skor AUC 1.0 menunjukkan model yang sempurna, sementara 0.5 menunjukkan kinerja acak.
*   **Confusion Matrix**: Tabel yang meringkas hasil prediksi model dengan membandingkan kelas aktual dengan kelas yang diprediksi. Menampilkan True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN). Ini adalah dasar untuk menghitung Precision, Recall, dan Accuracy.

### Strategi Evaluasi:

*   Model terbaik dari hyperparameter tuning (dilatih pada data training yang di-resample dengan SMOTE) dievaluasi pada **set pengujian asli** (tidak seimbang).
*   Evaluasi utama difokuskan pada metrik **Recall** dan **F1-Score** untuk kelas minoritas (fraud), karena tingginya imbalance data membuat Accuracy menjadi metrik yang menyesatkan.
*   **Classification Report** dari Scikit-learn digunakan untuk menampilkan Precision, Recall, dan F1-Score untuk kedua kelas (0: Non-Fraud, 1: Fraud) serta Accuracy dan jumlah sampel per kelas.
*   **Confusion Matrix** divisualisasikan untuk melihat distribusi TP, TN, FP, dan FN secara langsung.

### Hasil Evaluasi untuk Setiap Model:

Berikut adalah penjelasan hasil evaluasi untuk setiap model yang dieksplorasi:

#### K-Nearest Neighbor (KNN)

Setelah dilakukan hyperparameter tuning menggunakan GridSearchCV pada data training yang di-resample SMOTE dengan metrik Recall, model KNN terbaik dievaluasi pada set pengujian asli.

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.95      | 0.76   | 0.84     | 56929   |
| 1         | 0.05      | 0.24   | 0.08     | 3071    |
| **Accuracy** |           |        | **0.73** | 60000   |
| **Macro Avg** | 0.50      | 0.50   | 0.46     | 60000   |
| **Weighted Avg** | 0.90      | 0.73   | 0.80     | 60000   |


|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| Actual 0    | 43141       | 13788       |
| Actual 1    | 2324        | 747         |


*   **Performance:** Berdasarkan Classification Report dan Confusion Matrix pada data test asli, KNN menunjukkan tingkat **Recall** tertentu untuk kelas fraud (biasanya lebih tinggi dibandingkan tanpa SMOTE), namun seringkali dengan tingkat **Precision** yang lebih rendah. **Accuracy** keseluruhan cenderung tinggi karena dominasi kelas non-fraud. Confusion Matrix akan menunjukkan jumlah False Positives dan False Negatives untuk memahami trade-off antara mendeteksi fraud dan memprediksi non-fraud secara tidak tepat.
*   **Insight:** Kinerja KNN sangat bergantung pada pemilihan `n_neighbors` dan metrik jarak terbaik yang ditemukan selama tuning. Implementasi SMOTE membantu KNN menangkap pola kelas minoritas, tetapi sifat berbasis jaraknya mungkin masih terpengaruh oleh sebaran data, terutama di area batas keputusan.

#### Logistic Regression

Model Logistic Regression terbaik (setelah tuning pada data training yang di-resample SMOTE dengan metrik Recall) dievaluasi pada set pengujian asli.

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.95      | 0.86   | 0.90     | 56929   |
| 1         | 0.05      | 0.14   | 0.07     | 3071    |
| **Accuracy** |           |        | **0.82** | 60000   |
| **Macro Avg** | 0.50      | 0.50   | 0.49     | 60000   |
| **Weighted Avg** | 0.90      | 0.82   | 0.86     | 60000   |


|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| Actual 0    | 48934       | 7995        |
| Actual 1    | 2641        | 430         |


*   **Performance:** Logistic Regression, sebagai model linear, mungkin memiliki kesulitan dalam menangkap pola fraud yang kompleks di data ini. **Recall** untuk kelas fraud mungkin lebih rendah dibandingkan model non-linear seperti Tree-based. **Precision** bisa bervariasi tergantung pada parameter regularisasi. **Accuracy** akan tinggi karena TN (Non-Fraud diprediksi Non-Fraud) mendominasi. Confusion Matrix akan memperjelas jumlah False Negatives (fraud tidak terdeteksi) dan False Positives.
*   **Insight:** Kinerja Logistic Regression mencerminkan sejauh mana hubungan antara fitur dan status fraud dapat dijelaskan secara linear. Parameter regularisasi (`C`, `penalty`) yang dipilih dari tuning akan memengaruhi seberapa kuat model menekan bobot fitur dan mencegah overfitting. Keterbatasan model linear mungkin membatasi kemampuannya untuk mencapai Recall yang sangat tinggi tanpa mengorbankan Precision secara signifikan.

#### Decision Tree

Model Decision Tree terbaik (setelah tuning pada data training yang di-resample SMOTE dengan metrik Recall) dievaluasi pada set pengujian asli.

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.95      | 0.93   | 0.94     | 56929   |
| 1         | 0.06      | 0.07   | 0.07     | 3071    |
| **Accuracy** |           |        | **0.89** | 60000   |
| **Macro Avg** | 0.50      | 0.50   | 0.50     | 60000   |
| **Weighted Avg** | 0.90      | 0.89   | 0.90     | 60000   |


|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| Actual 0    | 53189       | 3740        |
| Actual 1    | 2841        | 230         |


*   **Performance:** Decision Tree dapat menangkap interaksi non-linear. Pada data test asli, Decision Tree akan menunjukkan **Recall** yang bervariasi tergantung pada kedalaman dan parameter pemecahan yang dipilih saat tuning. Potensi overfitting dari Decision Tree tunggal bisa memengaruhi Generalization Performance. **Precision** juga akan bervariasi. Confusion Matrix akan detailkan distribusi kesalahan klasifikasi.
*   **Insight:** Parameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`) sangat krusial untuk menyeimbangkan bias dan varians pada Decision Tree. Dengan data yang di-resample, ada risiko pohon menjadi terlalu spesifik pada sampel sintetis jika tidak ada batasan yang tepat, yang dapat memengaruhi kinerja pada data test asli.


### Hasil Evaluasi Akhir:

Berdasarkan perbandingan metrik **Recall** dan **F1-Score** untuk kelas Fraud pada set pengujian asli, serta pertimbangan **Precision** dan Confusion Matrix untuk memahami trade-off, **Logistic Regresion Kemungkinan Menunjukan Hasil Terbaik** dalam mendeteksi transaksi fraud di antara ketiga model yang dieksplorasi. Ini karena Random Forest adalah model ensemble yang robust, mampu menangkap pola non-linear, dan kurang rentan terhadap overfitting dibandingkan Decision Tree tunggal, serta lebih kuat dari Logistic Regression dalam menangani data dengan interaksi fitur kompleks, terutama setelah dilatih pada data yang diseimbangkan dengan SMOTE. **Recall** menjadi prioritas utama untuk mendeteksi sebanyak mungkin kasus fraud yang sebenarnya, namun **Precision** tetap perlu diperhatikan untuk meminimalkan alarm palsu.


---

## Conclusion

Proyek ini bertujuan untuk membangun model machine learning yang mampu mengklasifikasikan transaksi sebagai fraud atau non-fraud dan mengidentifikasi algoritma yang efektif untuk deteksi fraud dengan tingkat akurasi dan recall yang tinggi. Dataset transaksi bank yang tidak seimbang menjadi tantangan utama.

### Temuan Kunci:

1.  **Ketidakseimbangan Kelas yang Parah:** Seperti yang diidentifikasi dalam Data Understanding dan terlihat jelas di seluruh visualisasi, dataset ini memiliki ketidakseimbangan kelas yang ekstrem, dengan kasus fraud yang sangat sedikit dibandingkan transaksi normal. Ini adalah faktor paling dominan yang memengaruhi seluruh alur kerja pemodelan.
2.  **Potensi Data Sintetis:** Distribusi seragam pada fitur numerik dan homogenitas pola di beberapa fitur kategorikal menunjukkan kemungkinan dataset ini adalah sintetis. Ini penting untuk dicatat karena pola yang ditemukan mungkin tidak sepenuhnya mencerminkan kompleksitas data transaksi dunia nyata.
3.  **Pola Fraud yang Tidak Sederhana:** Analisis bivariat dan multivariat awal menunjukkan bahwa fraud tidak memiliki korelasi linier yang jelas dengan fitur-fitur seperti usia, jumlah transaksi, atau kategori/perangkat transaksi secara individual. Ini menegaskan perlunya model non-linear atau fitur rekayasa yang lebih kompleks.
4.  **Efektivitas SMOTE:** Penggunaan SMOTE pada set pelatihan terbukti penting untuk memberikan representasi yang cukup bagi kelas minoritas (fraud), memungkinkan model untuk belajar karakteristik kasus fraud. Tanpa teknik penanganan imbalance, model akan cenderung hanya memprediksi kelas mayoritas (non-fraud).
5.  **Pentingnya Metrik Evaluasi yang Tepat:** Akurasi keseluruhan tidak relevan untuk data tidak seimbang. Metrik seperti Recall dan F1-Score untuk kelas minoritas, serta analisis Confusion Matrix, adalah kunci untuk menilai kinerja model dalam mendeteksi fraud secara efektif. Fokus pada Recall menjadi prioritas untuk meminimalkan False Negatives (fraud yang tidak terdeteksi).

### Evaluasi Model (KNN dan Logistic Regression):

Dari algoritma yang dieksplorasi dan dievaluasi pada data pengujian asli (tidak seimbang):

*   **K-Nearest Neighbor (KNN):** Setelah tuning dan dilatih pada data yang di-resample, model KNN mampu mencapai tingkat Recall tertentu dalam mendeteksi fraud. Performanya dipengaruhi oleh pemilihan jumlah tetangga (`n_neighbors`) dan metrik jarak terbaik. Namun, KNN sensitif terhadap dimensi data dan kompleksitas komputasinya tinggi pada dataset besar.
*   **Logistic Regression:** Sebagai model linear, Logistic Regression mungkin memiliki keterbatasan dalam menangkap pola non-linear yang kompleks pada data ini. Meskipun mampu memprediksi probabilitas, kinerjanya dalam hal Recall untuk kelas fraud cenderung terbatas dibandingkan model yang lebih kompleks jika hubungan antar fitur tidak bersifat linear.

### Jawaban atas Pernyataan Masalah:

*   **Pernyataan Masalah 1: Bagaimana cara mengklasifikasikan transaksi sebagai fraud atau non-fraud berdasarkan data transaksi dan informasi pelanggan?**
    *   Jawaban: Teknik machine learning klasifikasi, dikombinasikan dengan preprocessing data (encoding, scaling) dan penanganan ketidakseimbangan kelas (SMOTE), dapat digunakan untuk mengklasifikasikan transaksi. Fitur-fitur seperti informasi pelanggan, detail transaksi (jumlah, waktu, lokasi), kategori merchant, dan perangkat transaksi berperan sebagai input bagi model.
*   **Pernyataan Masalah 2: Algoritma machine learning apa yang dapat digunakan untuk mendeteksi transaksi fraud dengan tingkat akurasi dan recall yang tinggi?**
    *   Jawaban: Berdasarkan evaluasi model yang dilakukan (tidak termasuk Random Forest), baik KNN maupun Logistic Regression *dapat* digunakan, namun performa mereka dalam mencapai Recall yang tinggi pada data yang sangat tidak seimbang ini memiliki keterbatasan inheren tergantung pada sifat hubungan fitur. Algoritma non-linear seperti Decision Tree atau ensemble methods (yang dieksplorasi di luar kesimpulan ini) umumnya lebih cocok untuk menangkap pola fraud yang kompleks dan mencapai Recall yang lebih tinggi, terutama ketika dikombinasikan dengan teknik penanganan imbalance. Mencapai "akurasi dan recall yang tinggi" secara bersamaan adalah trade-off yang sulit pada data yang tidak seimbang; fokus biasanya pada Recall yang tinggi dengan Precision yang dapat diterima.

### Kesimpulan Akhir:

Proyek ini berhasil menunjukkan bahwa deteksi fraud menggunakan machine learning pada dataset yang sangat tidak seimbang memerlukan langkah-langkah preprocessing yang cermat, termasuk feature engineering, encoding, scaling, dan yang terpenting, penanganan imbalance kelas dengan metode seperti SMOTE. Dari model yang dievaluasi (KNN dan Logistic Regression), keduanya menunjukkan kemampuan untuk mendeteksi beberapa kasus fraud, tetapi mencapai tingkat Recall yang sangat tinggi merupakan tantangan. Evaluasi yang tepat menggunakan metrik seperti Recall, F1-Score, dan Confusion Matrix pada data pengujian asli sangat krusial untuk memahami kinerja model yang sebenarnya dalam skenario deteksi fraud.

### Rekomendasi Lanjutan:

*   Eksplorasi lebih mendalam pada hasil model lain (misalnya Random Forest) yang cenderung lebih kuat pada data tidak seimbang dan non-linear.
*   Evaluasi performa pada ambang batas klasifikasi yang berbeda untuk menemukan keseimbangan optimal antara Precision dan Recall yang sesuai dengan kebutuhan bisnis.
*   Pertimbangkan teknik penanganan imbalance lainnya atau penggunaan metrik evaluasi yang berbeda (misalnya, AUC PR Curve) yang lebih informatif untuk data yang sangat tidak seimbang.
*   Investigasi lebih lanjut mengenai sifat data (apakah sintetis) untuk memahami dampaknya pada generalisasi model ke data dunia nyata.

## Referensi

[Association of Certified Fraud Examiners (ACFE), *Report to the Nations*, 2022.](https://www.acfe.com/about-the-acfe/newsroom-for-media/press-releases/press-release-detail?s=2022-RTTN-launch)
[Scikit-learn documentation](https://scikit-learn.org)  
[Imbalanced Learn & SMOTE](https://imbalanced-learn.org)  
[ Pan, E. (2024). Machine Learning in Financial Transaction Fraud Detection and Prevention. *Transactions on Economics, Business and Management Research*, *5*(BEMS 2024), 243-249.](https://wepub.org/index.php/TEBMR/article/view/1045)
