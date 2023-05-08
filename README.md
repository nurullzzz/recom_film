![image](https://user-images.githubusercontent.com/75351301/236828265-ec21ab75-5736-48cf-8d04-ebe261fb2bb1.png)
Proyek Terakhir Pada Kelas Machine Learning Terapan (Kelas Expert) Dicoding 

# Laporan Proyek Machine Learning - Nurul Tazkiyah Adam 

## Daftar Isi

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling and Result](#modeling-and-result)
- [Evaluation](#evaluation)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Project Overview

Sistem rekomendasi adalah metode yang digunakan untuk memberikan rekomendasi pada sebuah produk seperti buku, musik dan film dengan memberikan nilai prediksi tertinggi pada penggunanya. Kualitas sistem rekomendasi masih terus ditingkatkan hingga saat ini, dengan harapan metode baru dapat lebih meningkatkan lagi nilai relevansi dari hasil rekomendasi yang diberikan daripada sistem-sistem sebelumnya. Selain itu perkembangan film didunia juga setiap harinya semakin meningkat dengan berbagai variasi jenis genre yang dimiliki, sehingga membuat para penonton film merasa kesulitan untuk memilih film apa yang akan ditonton. 

Maka dari itu, proyek ini akan membahas  sistem rekomendasi film kepada para pengguna berdasarkan sistem rekomendasi menggunakan data yang telah dikumpulkan baik berdasarkan data pengguna maupun data dari film itu sendiri. Misalnya ketika pelanggan menonton sebuah film dengan genre tertentu, maka sistem dapat memberikan rekomendasi film lain dengan genre yang sama. Selain itu, sistem juga dapat memberikan rekomendasi berdasarkan rating dan preferensi pengguna lain dengan ciri-ciri ataupun profil pengguna satu yang masih relevan, mirip ataupun sama. 

Adapun referensi yang diambil untuk proyek overview ini yakni hasil riset oleh Kiki Ratna Sari, Wildan Suharso, dan Yufiz Azhar dari Teknik Informatika Universitas Muhammadiyah Malang yang mengembangkan sistem rekomendasi film menggunakan Item based Collaborative Filtering untuk memberikan hasil rekomendasi yang sangat mendekati dengan preferensi nilai yang diberikan oleh penggunanya dan telah diuji sistem tersebut dengan hasil nilai akurasi sebesar 97%

Selanjutnya hasil riset oleh Muhammad Fajriansyah, Putra Pandu Adikara, dan Agus Wahyu Widodo dari Fakultas Ilmu Komputer Universitas Brawijaya yang menggunakan algoritme content based filtering dengan mencari kemiripan bobot dari term pada bag of words hasil pre-processing sinopsis film dan judul film.

Hasil riset dapat dilihat melalui taautan berikut:
- [Pembuatan Sistem Rekomendasi Film dengan Menggunakan Metode Item Based Collaborative Filtering pada Apache Mahout](https://www.researchgate.net/publication/343197618_Pembuatan_Sistem_Rekomendasi_Film_dengan_Menggunakan_Metode_Item_Based_Collaborative_Filtering_pada_Apache_Mahout)
- [Sistem Rekomendasi Film Menggunakan Content Based Filtering](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka diperoleh problem statement pada proyek ini, yakni:
1. Bagaimana cara melakukan pengolahan data yang baik sehingga dapat digunakan untuk membuat model sistem rekomendasi yang bagus?
3. Bagaimana cara membangun model machine learning agar mendapatkan rekomendasi film yang mungkin disukai pengguna berdasarkan data rating atau penilaian pengguna terhadap film?

### Goals

Berdasarkan rumusan masalah di atas, maka diperoleh tujuan dari proyek ini, yaitu:
1. Melakukan pengolahan data yang baik agar dapat digunakan dalam membangun model sistem rekomendasi yang baik.
3. Membangun model machine learning gar mendapatkan rekomendasi film yang mungkin disukai pengguna berdasarkan data rating atau penilaian pengguna terhadap film.

### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:

- Content Based Filtering adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit.
    - TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency Vectorizer) digunakan untuk menemukan representasi fitur penting dari setiap kategori film. TF-IDF Vectorizer dari library scikit-learn akan melakukan vektorisasi nilai dengan menggunakan metode `fit_transform` dan `transform`, serta melakukan tokenisasi data secara langsung.
    - Cosine similarity merupakan suatu pengukuran untuk mencari tingkat derajat kesamaan antar dua buah vektor dalam ruang dimensi dari nilai cosinus. Metode pencarian derajat kesamaan menggunakan cosine similarity memiliki nilai akurasi yang cukup tinggi karena tidak berpengaruh pada panjang atau pendeknya suatu dokumen. 
- Collaborative Filtering merupakan teknik merekomendasikan item yang mirip dengan preferensi pengguna yang sama di masa lalu, misalnya berdasarkan penilaian film yang telah diberikan oleh seorang pengguna. Sistem akan merekomendasikan film berdasarkan riwayat penilaian pengguna tersebut terhadap film dan genrenya.

    
## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari platform [Kaggle ](kaggle.com/datasets/rohan4050/movie-recommendation-data) dengan total 6 file yakni:
*   pada folder ml-latest-small, berisi file:
    - links.csv
    - movies.csv
    - ratings.csv
    - tags.csv
*   file movies_metadata.csv

namun pada proyek ini hanya akan menggunakan file `movies.csv` dan `ratings.csv`.

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

- **Dataset Movies** (`movies.csv`)
    - Jumlah data movies sebanyak 9742 
    - Jumlah variabel pada data movies sebanyak 3 variabel, yakni:
            - `movieId` -> ID film (int64)
            - `title` -> Judul film (object)
            - `genres` -> Genre film (object)
    - Data Missing Value Pada Variable Dataset Movies:
            - `movieId` sebanyak 0
            - `title` sebanyak 0
            - `movieId` sebanyak 0
    - Duplikasi Dataset Movies: 0 duplikasi
    - Data Unique Pada Variable Dataset Movies:
            - `movieId` sebanyak 9742
            - `title` sebanyak 9737
            - `movieId` sebanyak 951

- **Dataset Ratings** (`ratings.csv`)
    - Jumlah data ratings sebanyak 100836 
    - Jumlah variabel pada data ratings sebanyak 4 variabel yakni
        - `userId` -> ID User pemberi rating (int64)
        - `movieId` -> ID film yang dirating (int64)
        - `rating` -> Rating film yang diberikan user (float64)
        - `timestamp` -> Waktu rating terekam (int64)
    - Skala Rating Film yakni rating 0,5 sampai rating 5.
    - Data Missing Value Pada Variable Dataset Ratings:
        - `userId` sebanyak 0
        - `movieId` sebanyak 0
        - `rating` sebanyak 0
        - `timestamp` sebanyak 0
    - Duplikasi Dataset Ratings: 0 duplikasi
    - Data Unique Pada Variable Dataset Ratings:
        - `userId` sebanyak 610
        - `movieId` sebanyak 9724
        - `rating` sebanyak 10
        - `timestamp` sebanyak 85043
        

## Data Preparation

Data preparation diperlukan untuk mempersiapkan data agar ketika dilakukan proses pengembangan model akurasi model yang didapatkan semakin baik dan meminimalisir terjadinya bias pada data. Berikut ini merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data:

- **Pembersihan Missing Value**
Penanganan yang penulis lakukan pada missing value yaitu dengan melakukan drop data. Tetapi karena dataset yang digunakan cukup bersih, missing value hanya terdapat ketika proses penggabungan dataset.

- **Sorting Data Rating berdasarkan User ID Kemudian Menjadikan Integer**
Melakukan pengurutan data rating berdasarkan ID Pengguna agar mempermudah dalam melakukan penghapusan data duplikat nantinya.

- **Pembersihan Duplikasi Data**
Melakukan penghapusan data duplikat agar tidak terjadi bias pada data nantinya.

- **Penggabungan Dataset**
Melakukan penggabungan data yang sudah diolah sebelumnya untuk membangun model. Lalu menghapus data yang memiliki missing value pada variabel genre dan melihat jumlah data setelah digabungkan, terlihat data memiliki 100830 baris dengan 5 kolom /variable.

- **Menggunakan TfidfVectorizer dan Cosine Similarity**.
Teknik TfidfVectorizer dan Cosine Similarity ini digunakan untuk model Content Based Filtering.

- **Encoding dan Mapping Fitur**
proses encoding fitur userId pada dataset ratings dan fitur movieId pada dataset menjadi sebuah array. Lalu hasil encoding tersebut akan dilakukan pemetaan atau mapping fitur yang telah dilakukan encoding tersebut ke dalam dataset.

Berdasarkan hasil encoding dan mapping tersebut, diperoleh jumlah user sebesar 32, jumlah film sebesar 2427, nilai rating minimal sebesar 0.5, dan nilai rating maksimal yaitu 5.0.

- **Pembagian Data Training dan Validasi**
Untuk melatih model diperlukan pembagian dataset latih dan juga dataset validasi, untuk dataset latih penulis berikan 80% dari total keseluruhan jumlah data sedangkan dataset validasi sebesar 20% dari keseluruhan data. Hal ini diperlukan untuk pengembangan pada model Collaborative Filtering.
 

## Modeling and Result

Tahap pengembangan model machine learning atau modeling sistem rekomendasi dilakukan untuk memberikan hasil rekomendasi film terbaik kepada pengguna tertentu berdasarkan *rating* atau penilaian pengguna terhadap film tersebut. Tahap modeling yang dilakukan menggunakan teknik pendekatan *content-based filtering recommendation* dan *collaborative filtering recommendation*.

- **Content-based Filtering Recommendation**
   
   Pada modeling Content Based Filtering, langkah pertama yang dilakukan yakni menggunakan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap genre film. dengan menggunakan fungsi `tfidfvectorizer()` dari library sklearn. Selanjutnya, melakukan fit dan transformasi ke dalam bentuk matriks. Output matriks yang dihasilkan berukuran (9737, 23). 9737 merupakan angka dari ukuran data dan 23 merupakan matriks genre film. Untuk menghitung derajat kesamaan (similarity degree) antar movie, menggunakan teknik cosine similarity dengan fungsi `cosine_similarity dari` library sklearn. Rumusnya berikut ini: 

    ![rumus cosim](https://raw.githubusercontent.com/nurullzzz/recom_film/main/Screenshot%202022-12-06%20192952.png)
    
    selanjutnya yaitu menggunakan argpartition untuk mengambil sejumlah nilai k tertinggi dari similarity data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Kemudian menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi movies yang mirip dari film yang ingin dicari.
    - Kelebihan
      Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
    - Kekurangan
      Hanya dapat digunakan untuk fitur yang sesuai, seperti film, dan buku, kemudian tidak mampu menentukan profil dari user baru.
    
   **Hasil pengujian dengan langkah diambil sebuah judul film dengan detail berikut** 
     |       | movieId | title        | genres |
     |-------|---------|--------------|--------|
     | 1734 | 2329	  | American History X (1998 | Crime, Drama  |
     
     Lalu didapatkan hasil Top 10 rekomendasi judul film berdasarkan genre yang sama.
     
     |   | title                      | movieId  | genres  |
     |---|----------------------------|---------|---------|
     | 0 | People I Know (2002)	       |6330	| Crime, Drama |
     | 1 | United States of Leland, The (2003) |7377 | Crime, Drama |
     | 2 | Above the Rim (1994)     | 409	  | Crime, Drama |
     | 3 |Road to Perdition (2002)       | 5464 | Crime, Drama  |
     | 4 | Virgin Spring, The (Jungfrukällan) (1960)   | 7820  | Crime, Drama |
     | 5 | Who'll Stop the Rain (1978)   | 4695  | Crime, Drama |
     | 6 | Prophet, A (Un Prophète) (2009)    | 73344 | Crime, Drama |
     | 7 | Tsotsi (2005)	     | 44204  | Crime, Drama |
     | 8 | Godfather: Part II, The (1974)      | 1221  | Crime, Drama |
     | 9 |Tattooed Life (Irezumi ichidai) (1965)  | 63768	 | Crime, Drama |

    Hasil uji model rekomendasi content based filtering di atas berhasil. Dapat dilihat bahwa sistem yang telah dirancang memberikan rekomendasi 10 judul film yang sesuai berdasarkan genre yang sama dengan satu judul film yang dipilih. Pada kasus ini menggunakan sample film *American History X (1998)* yang memiliki genre Crime dan Drama.

- **Collaborative Filtering Recommendation**
   
   Pertama  dilakukan dengan proses *encoding* fitur userId dan fitur movieId pada dataset menjadi *array*. Lalu hasil *encoding* tersebut akan dilakukan pemetaan atau *mapping* fitur yang telah dilakukan *encoding* tersebut ke dalam *dataset* *ratings*.
     
     Berdasarkan hasil encoding dan mapping tersebut, diperoleh 
        - Jumlah Pengguna Sebanyak 610
        - Jumlah Movie Sebanyak 9719
        - Nilai Rating Terkecil: 0.0
        - Nilai Rating Terbesar: 5.0
     
   Setelah itu tahap pembagian *dataset* atau *split* *dataset* diawali dengan mengacak *dataset*, kemudian melakukan pembagian menjadi data latih (*training data*) dan data validasi (*validation data*), dengan rasio 80:20.
    
    Selanjutnya,  proses embedding terhadap data film dan pengguna. Lalu lakukan operasi perkalian dot product antara embedding pengguna dan film. Serta, menambahkan bias untuk setiap pengguna dan film. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Untuk mendapatkan rekomendasi film dengan mengambil sampel user secara acak dan mendefinisikan variabel *movie_not_watched* yang merupakan daftar film yang belum pernah ditonton oleh pengguna.

    - Kelebihan
        Tidak memerlukan atribut untuk setiap itemnya, dapat membuat rekomendasi tanpa harus selalu menggunakan dataset yang lengkap, unggul dari segi kecepatan dan skalabilitas, rekomendasi tetap akan berkerja dalam keadaan dimana konten sulit dianalisi sekalipun,
    - Kekurangan
        membutuhkan parameter rating, sehingga jika ada item baru sistem tidak akan merekomendasikan item tersebut.
    
    **Hasil Pengujian Rekomendasi Film dengan Pendekatan Collaborative Filtering**
     ```
     297/297 [==============================] - 0s 1ms/step
     Hasil Rekomendasi Berdasarkan Model Collaborative Filtering
     ============================================================
     Rekomendasi Untuk User dengan ID 20 (dipilih secara acak)
     ============================================================
     5 Movie dengan Rating Tertinggi dari user 20
     ----------------------------------------
     Aladdin (1992) : Adventure|Animation|Children|Comedy|Musical
     Sword in the Stone, The (1963) : Animation|Children|Fantasy|Musical
     Toy Story 2 (1999) : Adventure|Animation|Children|Comedy|Fantasy
     Muppets Take Manhattan, The (1984) : Children|Comedy|Musical
     Spirited Away (Sen to Chihiro no kamikakushi) (2001) : Adventure|Animation|Fantasy
     ============================================================
     Untuk itu, Top 10 Rekomendasi Movie untuk user 20 adalah berikut
     ----------------------------------------
     Pulp Fiction (1994) : Comedy|Crime|Drama|Thriller
     Shawshank Redemption, The (1994) : Crime|Drama
     Godfather, The (1972) : Crime|Drama
     Star Wars: Episode V - The Empire Strikes Back (1980) : Action|Adventure|Sci-Fi
     Apocalypse Now (1979) : Action|Drama|War
     Star Wars: Episode VI - Return of the Jedi (1983) : Action|Adventure|Sci-Fi
     Third Man, The (1949) : Film-Noir|Mystery|Thriller
     Goodfellas (1990) : Crime|Drama 
     Ran (1985) : Drama|War
     Godfather: Part II, The (1974) : Crime|Drama
     ```

## Evaluation

Evaluasi yang akan penulis lakukan disini yaitu evaluasi dengan Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE) pada Collaborative Filtering dan Precision Content Based Filtering

- **Content-based Filtering Recommendation**
   
   Tahap evaluasi untuk sistem rekomendasi dengan *content-based filtering* dapat menggunakan metrik *precision*.
   
   ![Precision Metric Formula](https://raw.githubusercontent.com/nurullzzz/recom_film/main/Screenshot%202022-12-06%20192926.png)
   
     |       | movieId | title        | genres |
     |-------|---------|--------------|--------|
     | 0 | 1 | 	Toy Story (1995)	 | Adventure, Animation, Children, Comedy, Fantas |
   
    mengecek precision metrik dengan sample film **Toy Story (1995)**  yang memiliki 4 Genre yakni Adventure, Animation, Children, Comedy, dan Fantasy. Kemudian, didapatkan hasil berikut ini

     |   | title                      | movieId  | genres  |
     |---|----------------------------|---------|---------|
     | 0 | Shrek the Third (2007)  | 53121 | Adventure, Animation, Children, Comedy, Fantasy |
     | 1 | Asterix and the Vikings (Astérix et les Viking... |91355	 | Adventure, Animation, Children, Comedy, Fantasy |
     | 2 | Tale of Despereaux, The (2008) | 65577 | Adventure, Animation, Children, Comedy, Fantasy |
     | 3 | Monsters, Inc. (2001)	   | 4886 | Adventure, Animation, Children, Comedy, Fantasy |
     | 4 | Moana (2016) |166461| Adventure, Animation, Children, Comedy, Fantasy |
     | 5 | Emperor's New Groove, The (2000) | 4016 | Adventure, Animation, Children, Comedy, Fantasy |
     | 6 | Turbo (2013) | 103755| Adventure, Animation, Children, Comedy, Fantasy |
     | 7 | The Good Dinosaur (2015) | 136016 | Adventure, Animation, Children, Comedy, Fantasy |
     | 8 | Toy Story 2 (1999) | 3114  | Adventure, Animation, Children, Comedy, Fantasy |
     | 9 | Little Nemo: Adventures in Slumberland (1992) | 2800	 | Adventure, Animation, Children, Drama, Fantasy |
     
dari hasil tersebut didapatkan **9 dari 10 Film** mendapatkan rekomendasi genre yang sama (similar) dengan 4 kategori genre pada film **Toy Story (1995)**. Sehingga, dapat dikatakan **Precision Sistem yang telah dirancang pada proyek ini sebesar 90%**
   
- **Collaborative Filtering Recommendation**
   
   Tahap evaluasi untuk sistem rekomendasi dengan *collaborative filtering* menggunakan metrik MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error). Rumus untuk mencari nilai RMSE sebagai berikut,
   
   - Formula MAE
   ![mae](https://raw.githubusercontent.com/nurullzzz/recom_film/main/Screenshot%202022-12-06%20193127.png)
   Mean Absolute Error (MAE), Mengukur besarnya rata-rata kesalahan dalam serangkaian prediksi yang sudah dilatih kepada data yang akan dites, tanpa mempertimbangkan arahnya. Semakin rendah nilai MAE (Mean Absolute Error) maka semakin baik dan akurat model yang dibuat. Visualisasi MAE pada proyek ini bisa dilihat melalui plot berikut
    ![plot mae](https://raw.githubusercontent.com/nurullzzz/recom_film/main/download%20(1).png)

    Berdasarkan hasil fitting nilai konvergen metrik MAE berada dibawah 0.145 untuk data training dan sedikit diatas 0.155 untuk data validasi.
    
   - Formula RMSE
   ![rmse](https://raw.githubusercontent.com/nurullzzz/recom_film/main/Screenshot%202022-12-06%20193111.png)
   Root Mean Squared Error (RMSE), Aturan penilaian kuadrat yang juga mengukur besarnya rata-rata kesalahan. Sama seperti MAE, semakin rendahnya nilai root mean square error juga menandakan semakin baik model tersebut dalam melakukan prediksi. Visualisasi RMSE pada proyek ini bisa dilihat melalui plot berikut
    ![plot rmse](https://raw.githubusercontent.com/nurullzzz/recom_film/main/download%20(2).png)
    Berdasarkan hasil fitting nilai konvergen metrik RMSE berada dibawah 0.185 untuk data training dan kisaran 0.195 untuk data validasi.

## Kesimpulan

Dapat disimpulkan bahwa sistem berhasil melakukan rekomendasi baik dengan pendekatan *content-based filtering* maupun *collaborative filtering*. *Collaborative filtering* membutuhkan data penilaian film dari pengguna, sedangkan pada *content-based filtering*, data *rating* tidak dibutuhkan karena sistem akan merekomendasikan berdasarkan konten film tersebut, yaitu genre.

## Referensi

[1] [Pembuatan Sistem Rekomendasi Film dengan Menggunakan Metode Item Based Collaborative Filtering pada Apache Mahout](https://www.researchgate.net/publication/343197618_Pembuatan_Sistem_Rekomendasi_Film_dengan_Menggunakan_Metode_Item_Based_Collaborative_Filtering_pada_Apache_Mahout)

[2] [Sistem Rekomendasi Film Menggunakan Content Based Filtering](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163).

[3] https://github.com/AzharRizky/Movie-Recommendation-System

[4] https://github.com/onedayxzn/Rekomendasi-FilmA.

[5] https://github.com/chelizaaa/movie-recommender-system

