# Penerjemah Bahasa Isyarat SIBI Real-Time

Aplikasi ini bertujuan untuk menerjemahkan gestur huruf dari Sistem Isyarat Bahasa Indonesia (SIBI) secara real-time menggunakan input dari webcam. Hasil terjemahan akan ditampilkan sebagai teks dan juga dapat diucapkan sebagai output suara.

## Fitur Utama

* **Deteksi Tangan Real-Time**: Menggunakan MediaPipe untuk mendeteksi tangan dan mengekstrak titik-titik penting (*landmarks*) secara akurat.
* **Pengenalan Gestur Huruf SIBI**: Menggunakan model *Deep Learning* (LSTM) yang dilatih untuk mengenali berbagai gestur huruf SIBI.
* **Output Teks dan Suara**: Menampilkan huruf/kata yang terprediksi sebagai teks dan mengucapkannya menggunakan gTTS dengan Pygame untuk audio.
* **Pembuatan Dataset**: Dilengkapi dengan skrip untuk membantu pembuatan dataset:
    * Merekam video gestur dari webcam.
    * Mengekstrak *landmarks* dari video yang sudah direkam.
    * Mengekstrak *landmarks* dari gambar statis.
* **Normalisasi Data**: Skrip untuk normalisasi data *landmarks* sebelum training.
* **Training Model**: Skrip untuk melatih model LSTM dari dataset *landmarks* yang telah diproses.

## Teknologi yang Digunakan

* Python 3.x
* OpenCV (`opencv-python`)
* MediaPipe
* TensorFlow (Keras API)
* NumPy
* Scikit-learn
* gTTS (Google Text-to-Speech)
* Pygame (untuk memutar audio)

## Persiapan dan Instalasi

1.  **Prasyarat**:
    * Python (versi 3.8 atau lebih baru direkomendasikan).
    * `pip` (Python package installer).
    * Koneksi internet (untuk gTTS dan instalasi pustaka).
    * Webcam yang berfungsi.

2.  **Klon Repositori**:
    ```bash
    git clone [https://github.com/NAMA_PENGGUNA_ANDA/NAMA_REPOSITORI_ANDA.git](https://github.com/NAMA_PENGGUNA_ANDA/NAMA_REPOSITORI_ANDA.git)
    cd NAMA_REPOSITORI_ANDA
    ```

3.  **Buat dan Aktifkan Virtual Environment** (Direkomendasikan):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Instal Dependensi**:
    Pastikan Anda memiliki file `requirements.txt` di direktori root proyek Anda (konten untuk file ini ada di bawah). Kemudian jalankan:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Proyek ini memerlukan dataset gestur huruf SIBI untuk melatih model.

* **Jenis Data**: Disarankan untuk membuat dataset video untuk setiap huruf SIBI. Video ini kemudian akan diproses untuk mengekstrak sekuens *landmarks* tangan.
* **Skrip Pembuatan Dataset**:
    * `perekam_dataset_video.py`: Untuk merekam video gestur huruf SIBI dari webcam dan menyimpannya ke folder berdasarkan label huruf.
    * `video_ke_landmarks.py`: Untuk memproses dataset video yang sudah ada menjadi dataset sekuens *landmarks* (file `.npy`).
    * `gambar_ke_landmarks.py`: Untuk memproses dataset gambar statis menjadi dataset *landmarks* (file `.npy`, setiap gambar menjadi 1 frame).
* **Struktur Folder (Contoh)**:
    * Dataset Video Mentah: `dataset_video_sibi/<LABEL_HURUF>/nama_video.mp4`
    * Dataset Landmarks Mentah (dari video/gambar): `dataset_landmarks_sibi/<LABEL_HURUF>/nama_sampel.npy`
    * Dataset Landmarks Ternormalisasi: `dataset_landmarks_sibi_normalized/<LABEL_HURUF>/nama_sampel.npy`
* **Normalisasi**: Setelah membuat dataset *landmarks* mentah, jalankan `normalisasi_dataset.py` untuk membuat versi yang ternormalisasi.
* **Catatan**: Dataset mentah (video/gambar) dan dataset *landmarks* biasanya tidak disertakan langsung di repositori GitHub jika ukurannya besar. Pengguna diharapkan membuat dataset sendiri menggunakan skrip yang disediakan.

## Penggunaan Skrip

1.  **Pembuatan Dataset (Urutan Umum)**:
    * Jalankan `perekam_dataset_video.py` (jika merekam video baru dari webcam).
        ```bash
        python perekam_dataset_video.py
        ```
    * Jalankan `video_ke_landmarks.py` (jika Anda memiliki dataset video yang sudah ada).
        ```bash
        python video_ke_landmarks.py
        ```
    * (Opsional) Jalankan `gambar_ke_landmarks.py` (jika Anda memiliki dataset gambar).
        ```bash
        python gambar_ke_landmarks.py
        ```
    * Jalankan `normalisasi_dataset.py` pada folder output *landmarks* mentah.
        ```bash
        python normalisasi_dataset.py
        ```
        *(Pastikan path input dan output di dalam skrip ini sudah benar)*

2.  **Training Model**:
    Jalankan skrip training (misalnya, `training_sibi_lstm.py` - sesuaikan dengan nama file Anda).
    ```bash
    python training_sibi_lstm.py
    ```
    Ini akan memuat data *landmarks* yang sudah dinormalisasi, melatih model, dan menyimpan model terbaik (`sibi_lstm_model_best.keras`) serta file kelas label encoder (`sibi_label_encoder_classes.npy`).
    *(Pastikan path dataset di dalam skrip training sudah benar)*

3.  **Menjalankan Penerjemah Real-Time**:
    Jalankan program utama (misalnya, `penerjemah_sibi_realtime.py` - sesuaikan dengan nama file Anda).
    ```bash
    python penerjemah_sibi_realtime.py
    ```
    Program akan menggunakan webcam untuk deteksi dan terjemahan. Pastikan file model (`.keras`) dan file label encoder (`.npy`) berada di lokasi yang benar sesuai path di dalam skrip.

## Struktur Proyek (Contoh)
.
├── dataset_video_sibi/             # (Opsional, dibuat pengguna untuk menyimpan video asli)
├── dataset_landmarks_from_video/     # (Dihasilkan oleh video_ke_landmarks.py)
├── dataset_landmarks_from_images/  # (Dihasilkan oleh gambar_ke_landmarks.py)
├── dataset_landmarks_sibi_normalized/ # (Dihasilkan oleh normalisasi_dataset.py)
│
├── perekam_dataset_video.py        # Skrip merekam video gestur SIBI
├── video_ke_landmarks.py           # Skrip konversi dataset video ke landmarks
├── gambar_ke_landmarks.py          # Skrip konversi dataset gambar ke landmarks
├── normalisasi_dataset.py          # Skrip normalisasi dataset landmarks
├── training_sibi_lstm.py           # Skrip untuk melatih model LSTM (sesuaikan nama)
├── penerjemah_sibi_realtime.py     # Program utama penerjemah SIBI (sesuaikan nama)
│
├── sibi_lstm_model_best.keras      # Model terlatih (hasil training)
├── sibi_label_encoder_classes.npy  # Kelas label encoder (hasil training)
│
├── requirements.txt                # Daftar pustaka Python yang dibutuhkan
└── README.md                       # File ini
## Model

* **Arsitektur**: Model pengenalan gestur menggunakan arsitektur Long Short-Term Memory (LSTM).
* **File Model**: Model yang telah dilatih disimpan sebagai `sibi_lstm_model_best.keras`.
* **Label Encoder**: Pemetaan kelas label disimpan di `sibi_label_encoder_classes.npy`.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan lakukan *fork* pada repositori, buat *branch* baru untuk fitur atau perbaikan Anda, dan kemudian buat *Pull Request*.

## Lisensi

Proyek ini dilisensikan di bawah [LISENSI_ANDA, misalnya MIT License] - lihat file `LICENSE` untuk detailnya.

---
