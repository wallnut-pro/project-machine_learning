# Draft Bab 4.4 dan 4.5

## 4.4 Optimasi Model

Pada tahap ini dilakukan optimasi hyperparameter untuk Decision Tree dan Random Forest pada dua dataset, yaitu UCI Mushroom Dataset dan Secondary Mushroom Dataset. Optimasi tidak menggunakan test set. Seluruh pencarian parameter dilakukan hanya pada `X_train` dan `y_train` hasil Milestone 1, dengan `GridSearchCV` dan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

Skor utama yang digunakan adalah recall untuk kelas poisonous (`pos_label=1`). Pemilihan skor ini didasarkan pada risiko kesalahan false negative, yaitu kondisi ketika jamur poisonous diprediksi edible. Dalam konteks klasifikasi jamur, false negative lebih berbahaya daripada false positive karena berpotensi menghasilkan rekomendasi yang salah terhadap sampel beracun.

Grid yang digunakan untuk Decision Tree meliputi kombinasi `max_depth`, `min_samples_split`, `min_samples_leaf`, dan `class_weight`. Untuk Random Forest, grid meliputi `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, dan `class_weight`. Seluruh grid dijalankan penuh tanpa reduksi.

Hasil terbaik pada UCI Decision Tree diperoleh pada parameter `class_weight="balanced"`, `max_depth=None`, `min_samples_leaf=2`, dan `min_samples_split=2` dengan nilai recall cross-validation 1.0000. Pada UCI Random Forest, parameter terbaik adalah `n_estimators=100`, `max_depth=None`, `max_features="sqrt"`, `min_samples_split=2`, dan `class_weight=None`, juga dengan recall cross-validation 1.0000.

Pada Secondary Decision Tree, parameter terbaik adalah `class_weight="balanced"`, `max_depth=None`, `min_samples_leaf=1`, dan `min_samples_split=2`, dengan recall cross-validation 0.9983. Pada Secondary Random Forest, parameter terbaik kembali sama dengan baseline, yaitu `n_estimators=100`, `max_depth=None`, `max_features="sqrt"`, `min_samples_split=2`, dan `class_weight=None`, dengan recall cross-validation 1.0000.

Temuan penting dari hasil tuning adalah bahwa kedua Random Forest tuned kembali memilih konfigurasi baseline. Dengan demikian, tuning tidak memberikan perubahan konfigurasi maupun peningkatan performa yang terukur pada model tersebut.

## 4.5 Evaluasi Model

Evaluasi dilakukan terhadap delapan model: empat baseline dan empat tuned. Metrik yang dihitung meliputi accuracy, precision poisonous, recall poisonous, F1 poisonous, serta confusion matrix numerik berupa TN, FP, FN, dan TP.

Pada UCI, seluruh model baseline dan tuned menghasilkan accuracy 1.0000, precision poisonous 1.0000, recall poisonous 1.0000, dan F1 poisonous 1.0000. Semua model juga menghasilkan FP=0 dan FN=0. Dengan demikian, tuning tidak meningkatkan performa UCI. Walaupun tuned Decision Tree sedikit mengubah struktur pohon dari depth 7 menjadi 6, jumlah leaf justru naik dari 14 menjadi 19. Karena metrik identik, baseline Decision Tree tetap lebih layak dipilih untuk penggunaan praktis karena lebih cepat dilatih dan lebih mudah dijelaskan.

Pada Secondary Dataset, baseline Decision Tree menghasilkan accuracy 0.9990 dengan FP=6 dan FN=6. Setelah tuning, accuracy naik sedikit menjadi 0.9991, recall poisonous meningkat dari 0.9991 menjadi 0.9996, dan FN turun dari 6 menjadi 3. Namun, precision poisonous turun sedikit dari 0.9991 menjadi 0.9988 dan FP naik dari 6 menjadi 8. Selain itu, model tuned menjadi lebih kompleks, dengan depth meningkat dari 25 menjadi 32 dan jumlah leaf meningkat dari 206 menjadi 247. Jadi, tuning Decision Tree memberi peningkatan kecil pada recall poisonous, tetapi peningkatan ini tidak gratis karena ada trade-off pada FP dan kompleksitas model.

Baseline Random Forest pada Secondary sudah mencapai accuracy, precision, recall, dan F1 sebesar 1.0000, dengan FP=0 dan FN=0. Model tuned Random Forest menghasilkan hasil yang sama persis, karena parameter terbaik yang ditemukan identik dengan baseline. Oleh sebab itu, tuning tidak memberikan peningkatan terukur pada Random Forest Secondary.

Jika fokus utama adalah meminimalkan false negative, maka hasil paling kuat pada Milestone 4 berasal dari Random Forest untuk Secondary dan seluruh model pada UCI, karena semuanya mencapai FN=0. Namun, untuk pemilihan model akhir, performa harus dibaca bersama biaya dan kompleksitas. Pada UCI, baseline Decision Tree adalah pilihan paling rasional karena performanya sempurna tetapi struktur dan waktu latihnya lebih ringan. Pada Secondary, baseline Random Forest adalah model terbaik karena sudah memberi hasil sempurna sebelum tuning, sementara tuning hanya mengonfirmasi bahwa konfigurasi baseline tersebut memang sudah optimal dalam grid yang diuji.

Secara keseluruhan, Milestone 4 menunjukkan bahwa optimasi tidak selalu meningkatkan performa. Pada beberapa kasus, tuning hanya memverifikasi bahwa model baseline sudah berada pada konfigurasi yang sangat kuat. Karena itu, pembahasan hasil tuning harus berhati-hati dan tidak boleh otomatis menyatakan adanya peningkatan bila metrik akhir tetap sama atau perubahannya sangat kecil.
