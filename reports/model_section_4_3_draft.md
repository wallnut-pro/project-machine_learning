# Draft Bab 4.3

## 4.3 Pembangunan Model Baseline

Pada tahap ini, model dibangun menggunakan artefak hasil preprocessing dari Milestone 1, yaitu `X_train.npz`, `X_test.npz`, `y_train.npy`, `y_test.npy`, `feature_names.csv`, dan `preprocessor.joblib` untuk masing-masing dataset UCI dan Secondary. Dengan demikian, tidak ada preprocessing ulang pada tahap pemodelan dan test set tidak digunakan untuk melatih model.

Dua algoritma baseline yang digunakan adalah Decision Tree dan Random Forest. Decision Tree dibangun menggunakan `DecisionTreeClassifier(random_state=42)`, sedangkan Random Forest dibangun menggunakan `RandomForestClassifier(random_state=42, n_jobs=-1)`. Parameter lain dibiarkan pada nilai default karena hyperparameter tuning belum dilakukan pada milestone ini.

Secara konsep, Decision Tree bekerja dengan memilih split yang menurunkan impurity node secara maksimal. Dalam implementasi ini, impurity diukur menggunakan Gini Impurity. Pada setiap node, model mengevaluasi kandidat split pada fitur input hasil One-Hot Encoding, kemudian memilih pemisahan yang memberikan penurunan impurity terbesar. Random Forest memperluas pendekatan ini dengan membangun banyak pohon keputusan dari bootstrap sample berbeda, menambahkan acak fitur pada setiap split, lalu mengambil keputusan akhir melalui majority voting.

## 4.3.1 Struktur Model Aktual

Hasil pelatihan menunjukkan bahwa root node aktual Decision Tree pada UCI adalah `categorical__odor_n`, yang kembali dipetakan ke fitur asli `odor`. Temuan ini sejalan dengan hasil EDA yang menunjukkan bahwa `odor` memiliki asosiasi kategorikal tertinggi terhadap target, tetapi pada bab ini penting ditegaskan bahwa penetapan root node berasal dari model yang benar-benar dilatih, bukan dari asumsi awal.

Pada Secondary Dataset, root node aktual Decision Tree adalah `numeric__stem-width`, yang dipetakan ke fitur asli `stem-width`. Temuan ini juga konsisten dengan Milestone 2, di mana `stem-width` memperoleh skor mutual information tertinggi pada analisis train split.

Ringkasan struktur model baseline adalah sebagai berikut:

- UCI Decision Tree:
  - jumlah fitur input: 117
  - depth: 7
  - jumlah leaf: 14
  - root feature asli: `odor`
- Secondary Decision Tree:
  - jumlah fitur input: 105
  - depth: 25
  - jumlah leaf: 206
  - root feature asli: `stem-width`
- UCI Random Forest:
  - jumlah estimator: 100
- Secondary Random Forest:
  - jumlah estimator: 100

## 4.3.2 Feature Importance Aktual

Feature importance pada Random Forest dihitung dari model yang telah dilatih dan harus dibaca sebagai kontribusi fitur terhadap keputusan model, bukan hubungan sebab-akibat.

Pada UCI Random Forest, fitur agregat terpenting adalah `odor` (0.2462), `gill-size` (0.1194), `spore-print-color` (0.0894), `stalk-surface-below-ring` (0.0764), dan `ring-type` (0.0662). Untuk fitur encoded, kontribusi tertinggi berasal dari `categorical__odor_n` (0.1413).

Pada Secondary Random Forest, fitur agregat terpenting adalah `cap-surface` (0.1156), `gill-attachment` (0.1070), `stem-width` (0.0865), `gill-color` (0.0854), dan `stem-color` (0.0807). Untuk fitur encoded, kontribusi tertinggi berasal dari `numeric__stem-width` (0.0865).

Perbedaan antara ranking EDA dan ranking model perlu dicatat. EDA mengukur asosiasi statistik atau informasi terhadap target, sedangkan Random Forest importance mengukur kontribusi fitur dalam struktur ensemble yang benar-benar dilatih. Karena itu, kesamaan antara EDA dan model dapat dibahas sebagai konsistensi sinyal, tetapi perbedaan di antara keduanya bukan kontradiksi metodologis.

## 4.3.3 Metrik Baseline

Metrik evaluasi menggunakan kelas poisonous sebagai kelas positif (`pos_label=1`).

Hasil baseline menunjukkan:

- Secondary Decision Tree:
  - train accuracy = 1.0000
  - test accuracy = 0.9990
  - precision poisonous = 0.9991
  - recall poisonous = 0.9991
  - f1 poisonous = 0.9991
- Secondary Random Forest:
  - train accuracy = 1.0000
  - test accuracy = 1.0000
  - precision poisonous = 1.0000
  - recall poisonous = 1.0000
  - f1 poisonous = 1.0000
- UCI Decision Tree:
  - train accuracy = 1.0000
  - test accuracy = 1.0000
  - precision poisonous = 1.0000
  - recall poisonous = 1.0000
  - f1 poisonous = 1.0000
- UCI Random Forest:
  - train accuracy = 1.0000
  - test accuracy = 1.0000
  - precision poisonous = 1.0000
  - recall poisonous = 1.0000
  - f1 poisonous = 1.0000

## 4.3.4 Sanity-Check Metodologis

Sebelum hasil baseline ditutup final, dilakukan audit metodologis tambahan.

Pertama, kolom target `class` dipastikan tidak masuk ke feature matrix. Hasil pemeriksaan menunjukkan `target_in_preprocessor_inputs = False` dan `target_in_feature_names = False` untuk semua model. Selain itu, metadata model menunjukkan bahwa proses `fit` hanya menggunakan jumlah sampel yang sama dengan `X_train`, sehingga tidak ada indikasi bahwa test set digunakan saat pelatihan.

Kedua, exact overlap antara train dan test setelah preprocessing diperiksa. Pada UCI maupun Secondary, jumlah unique feature-vector overlap adalah 0 dan jumlah pola fitur+label identik lintas train-test juga 0. Hasil ini menunjukkan bahwa skor tinggi baseline tidak dapat dijelaskan oleh duplikasi vektor fitur yang identik antara train dan test.

Ketiga, dilakukan shuffled-label control dengan mengacak `y_train` menggunakan `random_state=42`, melatih model sementara, lalu mengevaluasi pada `y_test`. Hasilnya turun mendekati performa acak. Pada UCI, Decision Tree menghasilkan accuracy 0.5212 dan Random Forest 0.4917. Pada Secondary, Decision Tree menghasilkan accuracy 0.5158 dan Random Forest 0.5157. Penurunan ini mendukung kesimpulan bahwa baseline utama tidak memperoleh skor tinggi karena target leakage eksplisit.

Keempat, dilakukan validasi silang 5-fold hanya pada training set. Hasilnya tetap sangat konsisten: UCI Decision Tree mencapai accuracy rata-rata 0.9997, UCI Random Forest 1.0000, Secondary Decision Tree 0.9982, dan Secondary Random Forest 1.0000. Standar deviasi metrik juga sangat kecil. Dengan demikian, performa tinggi tidak hanya muncul pada satu split tunggal.

Kelima, confusion matrix numerik menunjukkan kesalahan yang sangat sedikit atau nol. Secondary Decision Tree menghasilkan FP=6 dan FN=6, sedangkan tiga baseline lain menghasilkan FP=0 dan FN=0 pada test split yang tersedia.

Keenam, dilakukan eksperimen diagnostik tambahan pada UCI dengan menghapus seluruh encoded feature yang berasal dari `odor`. Menariknya, akurasi, recall poisonous, dan F1 poisonous tetap 1.0000 untuk Decision Tree maupun Random Forest. Ini menunjukkan bahwa UCI memiliki redundansi sinyal yang sangat kuat: `odor` memang penting, tetapi dataset masih dapat dipisahkan sempurna oleh kombinasi fitur lain.

## 4.3.5 Interpretasi Hasil Baseline

Secara keseluruhan, sanity-check menunjukkan bahwa skor sangat tinggi pada baseline bukan artefak target leakage yang jelas. Namun, interpretasi hasil tetap harus berhati-hati. Pada Secondary Dataset tidak tersedia species/group identifier eksplisit, sehingga walaupun exact overlap train-test bernilai nol, kemiripan struktural antarsampel pada level grup tidak bisa diuji. Pada UCI, hasil sempurna meskipun `odor` dihilangkan memperlihatkan bahwa dataset memiliki redundansi informasi yang besar.

Karena itu, hasil baseline ini cukup dapat dipercaya untuk split dan pipeline yang digunakan, tetapi tetap belum cukup untuk menyimpulkan generalisasi di luar data yang tersedia. Milestone berikutnya harus memfokuskan diri pada hyperparameter tuning, evaluasi robustnes, dan analisis tambahan terhadap generalisasi model.
