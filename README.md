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
    git clone https://github.com/Aditya-Agung-T/VoiceSign-Indonesia.git
    cd VoiceSign-Indonesia
    ```

3.  **Buat dan Aktifkan Virtual Environment** (Opsional) (Direkomendasikan):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Requirements**:
    Pastikan Anda memiliki file `requirements.txt` di direktori. Kemudian jalankan:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Proyek ini memerlukan dataset gestur huruf SIBI untuk melatih model.

* **Jenis Data**: Disarankan untuk membuat dataset video untuk setiap huruf SIBI. Video ini kemudian akan diproses untuk mengekstrak sekuens *landmarks* tangan.
* **Skrip Pembuatan Dataset**:
    * `pembuatan_dataset_video.py`: Untuk merekam video gestur huruf SIBI dari webcam dan menyimpannya ke folder berdasarkan label huruf.
    * `video_ke_landmarks.py`: Untuk memproses dataset video yang sudah ada menjadi dataset sekuens *landmarks* (file `.npy`).
    * `cv_gambar_ke_landmarks.py`: Untuk memproses dataset gambar statis menjadi dataset *landmarks* (file `.npy`, setiap gambar menjadi 1 frame).
* **Struktur Folder (Contoh)**:
    * Dataset Video Mentah: `dataset_video_sibi/<LABEL_HURUF>/nama_video.mp4`
    * Dataset Landmarks Mentah (dari video/gambar): `dataset_landmarks_sibi/<LABEL_HURUF>/nama_sampel.npy`
    * Dataset Landmarks Ternormalisasi: `dataset_landmarks_sibi_normalized/<LABEL_HURUF>/nama_sampel.npy`
* **Normalisasi**: Setelah membuat dataset *landmarks* mentah, jalankan `normalisasi_datasets_landmarks.py` untuk membuat versi yang ternormalisasi.
* **Catatan**: Dataset mentah (video/gambar) dan dataset *landmarks* tidak disertakan langsung di repositori GitHub. Pengguna diharapkan membuat dataset sendiri menggunakan skrip yang disediakan.

## Penggunaan Skrip

1.  **Pembuatan Dataset (Urutan Umum)**:
    * Jalankan `pembuatan_dataset_video.py` (jika merekam video baru dari webcam).
        ```bash
        python pembuatan_dataset_video.py
        ```
    * Jalankan `video_ke_landmarks.py` (jika Anda memiliki dataset video yang sudah ada).
        ```bash
        python video_ke_landmarks.py
        ```
    * (Opsional) Jalankan `cv_gambar_ke_landmarks.py` (jika Anda memiliki dataset gambar).
        ```bash
        python cv_gambar_ke_landmarks.py
        ```
    * Jalankan `normalisasi_dataset.py` pada folder output *landmarks* mentah.
        ```bash
        python normalisasi_datasets_landmarks.py
        ```
        *(Pastikan path input dan output di dalam skrip ini sudah benar)*

2.  **Training Model**:
    Jalankan skrip training (`training_data_landmarks.py`).
    ```bash
    python training_data_landmarks.py
    ```
    Ini akan memuat data *landmarks* yang sudah dinormalisasi, melatih model, dan menyimpan model terbaik (`sibi_lstm_model_best.keras`) serta file kelas label encoder (`sibi_label_encoder_classes.npy`).
    *(Pastikan path dataset di dalam skrip training sudah benar)*

3.  **Menjalankan Penerjemah Real-Time**:
    Jalankan program utama (`main_program.py`).
    ```bash
    python main_program.py
    ```
    Program akan menggunakan webcam untuk deteksi dan terjemahan. Pastikan file model (`.keras`) dan file label encoder (`.npy`) berada di lokasi yang benar sesuai path di dalam skrip.

## Model

* **Arsitektur**: Model pengenalan gestur menggunakan arsitektur Long Short-Term Memory (LSTM).
* **File Model**: Model yang telah dilatih disimpan sebagai `sibi_lstm_model_best.keras`.
* **Label Encoder**: Pemetaan kelas label disimpan di `sibi_label_encoder_classes.npy`.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan lakukan *fork* pada repositori, buat *branch* baru untuk fitur atau perbaikan Anda, dan kemudian buat *Pull Request*.

## Lisensi

Proyek ini dilisensikan di bawah Creative Commons Zero v1.0 Universal

---
