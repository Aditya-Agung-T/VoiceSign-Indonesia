import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # Untuk one-hot encoding label
from tensorflow.keras.preprocessing.sequence import pad_sequences # Untuk padding
from sklearn.preprocessing import LabelEncoder
import collections

# Path ke dataset landmarks yang sudah dinormalisasi
DATASET_PATH = "dataset_landmarks_sibi_normalized"

# Daftar untuk menyimpan sekuens landmarks dan labelnya
sequences = []
labels = []

# Tentukan panjang maksimum sekuens (MAX_SEQ_LENGTH)
# Anda mungkin perlu menganalisis dataset Anda untuk menentukan nilai yang baik.
# Misalnya, jika sebagian besar sekuens memiliki ~60 frame (dari 2 detik @ 30fps),
# Anda bisa memilih nilai ini atau sedikit lebih tinggi.
# Untuk contoh, kita set ke 60. Jika sekuens lebih pendek, akan di-padding.
# Jika lebih panjang, akan dipotong (atau Anda bisa memilih strategi lain).
MAX_SEQ_LENGTH = 60 # Sesuaikan berdasarkan dataset Anda!

print("Memuat dataset...")
# Iterasi melalui setiap folder gestur (label)
for gesture_label in os.listdir(DATASET_PATH):
    gesture_path = os.path.join(DATASET_PATH, gesture_label)
    if os.path.isdir(gesture_path):
        print(f"  Memuat gestur: {gesture_label}")
        # Iterasi melalui setiap file .npy (sampel)
        for sample_file in os.listdir(gesture_path):
            if sample_file.endswith(".npy"):
                file_path = os.path.join(gesture_path, sample_file)
                try:
                    landmark_sequence = np.load(file_path)
                    if landmark_sequence.ndim == 3 and landmark_sequence.shape[1:] == (21, 3):
                        if landmark_sequence.shape[0] > 0: # Pastikan tidak kosong
                            sequences.append(landmark_sequence)
                            labels.append(gesture_label)
                        # else:
                        #     print(f"    Peringatan: Sekuens kosong ditemukan di {sample_file}, dilewati.")
                    # else:
                    #     print(f"    Peringatan: Format data tidak sesuai di {sample_file} (shape: {landmark_sequence.shape}), dilewati.")
                except Exception as e:
                    print(f"    Error saat memuat {sample_file}: {e}, dilewati.")

if not sequences:
    print("Tidak ada data yang berhasil dimuat! Pastikan path dataset benar dan berisi file .npy yang valid.")
    exit()

print(f"\nTotal {len(sequences)} sekuens dimuat dari {len(set(labels))} kelas gestur.")

# B. Mengubah Label menjadi Format Numerik
# LabelEncoder akan mengubah label string (misalnya 'A', 'B') menjadi angka (0, 1, ...)
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)

# Simpan pemetaan label untuk digunakan nanti saat inferensi
# np.save('label_encoder_classes.npy', label_encoder.classes_)
# print(f"Kelas label yang ditemukan dan di-encode: {list(label_encoder.classes_)}")

# One-hot encode labels (misalnya, jika ada 3 kelas: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1])
# Ini biasanya dibutuhkan untuk loss function 'categorical_crossentropy'
num_classes = len(label_encoder.classes_)
one_hot_labels = to_categorical(integer_encoded_labels, num_classes=num_classes)

print(f"Shape dari one_hot_labels: {one_hot_labels.shape}") # (jumlah_sampel, jumlah_kelas)

# C. Padding dan Reshaping Sekuens
# Setiap sampel (sekuens landmarks) memiliki bentuk (jumlah_frame, 21, 3)
# Kita perlu me-reshape setiap frame menjadi satu vektor fitur tunggal: 21 * 3 = 63 fitur.
# Jadi, bentuk per sampel menjadi (jumlah_frame, 63)
reshaped_sequences = []
for seq in sequences:
    num_frames = seq.shape[0]
    reshaped_seq = seq.reshape(num_frames, -1) # -1 akan otomatis menghitung 21*3 = 63
    reshaped_sequences.append(reshaped_seq)

# Sekarang, lakukan padding agar semua sekuens memiliki panjang yang sama (MAX_SEQ_LENGTH)
# 'pre' padding berarti menambahkan nol di awal jika sekuens lebih pendek.
# 'post' truncating berarti memotong dari akhir jika sekuens lebih panjang.
padded_sequences = pad_sequences(reshaped_sequences, maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='post', dtype='float32')

X = np.array(padded_sequences)
y = one_hot_labels

print(f"Shape data input (X) setelah padding dan reshape: {X.shape}") # (jumlah_sampel, MAX_SEQ_LENGTH, 63)
print(f"Shape data label (y): {y.shape}")

# D. Membagi Dataset
# Bagi data menjadi set pelatihan dan set validasi (misalnya, 80% train, 20% validation)
# Jika Anda punya banyak data, buat juga test set terpisah.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nData siap untuk training:")
print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Diasumsikan variabel berikut sudah ada dari langkah persiapan data:
# X_train, X_val, y_train, y_val
# num_classes (jumlah kelas gestur unik)
# MAX_SEQ_LENGTH (panjang sekuens setelah padding)
# label_encoder (objek LabelEncoder yang sudah di-fit)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np # Pastikan numpy diimpor jika belum

# --- Langkah 3: Membangun Model LSTM ---

# Dapatkan jumlah fitur per frame (timestep)
# X_train memiliki shape (jumlah_sampel, MAX_SEQ_LENGTH, jumlah_fitur_per_frame)
# jumlah_fitur_per_frame adalah 63 (dari 21 landmarks * 3 koordinat)
input_shape = (X_train.shape[1], X_train.shape[2]) # (MAX_SEQ_LENGTH, 63)

model = Sequential([
    Input(shape=input_shape, name='input_layer'),
    # Lapisan LSTM pertama
    # 'units' adalah dimensi output dari LSTM (jumlah sel memori).
    # 'return_sequences=True' jika Anda ingin menumpuk lapisan LSTM lain atau menggunakan
    # lapisan TimeDistributed setelah ini. Untuk lapisan LSTM terakhir sebelum Dense,
    # biasanya False (default).
    LSTM(units=64, return_sequences=True, name='lstm_1'),
    Dropout(0.2, name='dropout_1'), # Dropout untuk regularisasi

    # Lapisan LSTM kedua (opsional, bisa menambah kompleksitas dan kemampuan belajar)
    LSTM(units=128, return_sequences=False, name='lstm_2'), # return_sequences=False karena setelah ini Dense layer
    Dropout(0.2, name='dropout_2'),

    # Lapisan Dense
    Dense(units=128, activation='relu', name='dense_1'),
    Dropout(0.3, name='dropout_3'), # Dropout lebih besar sebelum output layer

    # Lapisan Output
    # 'num_classes' adalah jumlah huruf SIBI yang unik
    # 'softmax' untuk klasifikasi multi-kelas, menghasilkan probabilitas untuk setiap kelas
    Dense(units=num_classes, activation='softmax', name='output_layer')
])

# Mencetak ringkasan model
model.summary()

# --- Langkah 4: Mengompilasi Model ---
# Optimizer: 'adam' adalah pilihan yang baik dan umum digunakan.
# Loss function: 'categorical_crossentropy' karena label kita one-hot encoded.
#                Jika label Anda integer (0, 1, 2,..), gunakan 'sparse_categorical_crossentropy'.
# Metrics: 'accuracy' untuk melihat seberapa sering model membuat prediksi yang benar.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Langkah 5: Melatih Model ---

# Callbacks untuk meningkatkan proses training:
# ModelCheckpoint: Menyimpan model (atau hanya bobotnya) setiap kali ada peningkatan
#                  pada metrik yang dipantau (misalnya, val_accuracy).
# EarlyStopping: Menghentikan training jika tidak ada peningkatan pada metrik yang dipantau
#                setelah sejumlah 'patience' epoch, untuk mencegah overfitting dan hemat waktu.

# Tentukan path untuk menyimpan model terbaik
model_checkpoint_path = 'sibi_lstm_model_best.keras' # Menggunakan format .keras modern

callbacks_list = [
    ModelCheckpoint(filepath=model_checkpoint_path,
                    monitor='val_accuracy', # Pantau akurasi pada data validasi
                    save_best_only=True,    # Hanya simpan jika ada peningkatan
                    verbose=1),             # Tampilkan pesan saat menyimpan
    EarlyStopping(monitor='val_loss',     # Pantau loss pada data validasi
                  patience=15,            # Jumlah epoch tanpa peningkatan sebelum berhenti
                  verbose=1,
                  restore_best_weights=True) # Kembalikan bobot terbaik saat berhenti
]

# Parameter training
EPOCHS = 100  # Jumlah epoch (iterasi penuh melalui seluruh dataset training)
             # EarlyStopping mungkin akan menghentikannya lebih awal.
BATCH_SIZE = 32 # Jumlah sampel yang diproses sebelum model memperbarui bobotnya.
                # Sesuaikan berdasarkan ukuran memori GPU/CPU Anda.

print("\n--- Memulai Training Model ---")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val), # Data untuk validasi di setiap akhir epoch
                    callbacks=callbacks_list,
                    verbose=1) # Tampilkan progress bar

print("\n--- Training Selesai ---")

# Muat model terbaik yang disimpan oleh ModelCheckpoint (jika EarlyStopping mengembalikan bobot non-terbaik)
print(f"Memuat model terbaik dari: {model_checkpoint_path}")
model = tf.keras.models.load_model(model_checkpoint_path)
import matplotlib.pyplot as plt

# Diasumsikan 'history' adalah objek yang dikembalikan oleh model.fit()
# dan 'model' adalah model terbaik yang sudah dimuat.

#-----------------------------------------------------------
# Fungsi untuk plot riwayat training
#-----------------------------------------------------------
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc)) # Jumlah epoch yang sebenarnya dijalankan

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Akurasi Training')
    plt.plot(epochs_range, val_acc, label='Akurasi Validasi')
    plt.legend(loc='lower right')
    plt.title('Akurasi Training dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Loss Training')
    plt.plot(epochs_range, val_loss, label='Loss Validasi')
    plt.legend(loc='upper right')
    plt.title('Loss Training dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.suptitle('Riwayat Training Model', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

# Panggil fungsi untuk menampilkan plot
if 'history' in locals() and history is not None:
    plot_training_history(history)
else:
    print("Variabel 'history' tidak ditemukan atau None. Tidak bisa menampilkan plot.")

#-----------------------------------------------------------
# Evaluasi model pada data validasi (atau test set jika ada)
#-----------------------------------------------------------
print("\n--- Mengevaluasi Model pada Data Validasi ---")
# Diasumsikan X_val dan y_val sudah ada dan diproses dengan benar
try:
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f"\nLoss pada Data Validasi: {val_loss:.4f}")
    print(f"Akurasi pada Data Validasi: {val_accuracy*100:.2f}%")
except NameError:
    print("Variabel X_val atau y_val tidak ditemukan. Tidak bisa melakukan evaluasi.")
except Exception as e:
    print(f"Error saat evaluasi: {e}")

# --- Langkah 7: Menyimpan Label Encoder ---
# Ini penting agar kita bisa mengubah prediksi numerik model kembali menjadi label huruf SIBI.
try:
    np.save('sibi_label_encoder_classes.npy', label_encoder.classes_)
    print("Kelas LabelEncoder berhasil disimpan ke 'sibi_label_encoder_classes.npy'")
except NameError:
    print("PERHATIAN: Variabel 'label_encoder' tidak ditemukan. Pastikan sudah di-fit dan tersedia.")
    print("Anda perlu menyimpan 'label_encoder.classes_' secara manual jika ingin memuatnya nanti.")
except Exception as e:
    print(f"Error saat menyimpan kelas LabelEncoder: {e}")


# Anda bisa melanjutkan dengan evaluasi model pada test set (jika ada)
# atau menganalisis history training (kurva loss dan akurasi).
# Contoh:
# loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
# print(f"\nAkurasi Model pada Data Validasi (setelah memuat model terbaik): {accuracy*100:.2f}%")
# print(f"Loss Model pada Data Validasi: {loss:.4f}")