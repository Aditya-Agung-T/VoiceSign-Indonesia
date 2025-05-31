import numpy as np
import os
import shutil # Untuk menyalin struktur direktori jika diperlukan

# --- Fungsi Normalisasi Landmarks (sama seperti sebelumnya) ---
def normalize_landmarks(landmarks_data):
    """
    Normalisasi data landmarks tangan untuk satu frame.
    Membuat landmarks relatif terhadap pergelangan tangan dan menormalkan skalanya.

    Args:
        landmarks_data (np.array): Array NumPy dengan shape (21, 3) berisi
                                     koordinat x, y, z dari 21 landmarks tangan.

    Returns:
        np.array: Array NumPy dengan shape (21, 3) berisi landmarks yang sudah dinormalisasi.
                  Mengembalikan array nol jika input tidak valid atau tidak bisa dinormalisasi.
    """
    if landmarks_data is None or not isinstance(landmarks_data, np.ndarray) or landmarks_data.shape != (21, 3):
        # print(f"Data landmarks tidak valid untuk normalisasi: shape {landmarks_data.shape if isinstance(landmarks_data, np.ndarray) else 'Bukan array'}")
        return np.zeros((21, 3), dtype=np.float32)

    # 1. Normalisasi Translasi (Relatif terhadap Pergelangan Tangan - landmark 0)
    wrist_landmark = landmarks_data[0].copy()  # Landmark pergelangan tangan
    normalized_landmarks = landmarks_data - wrist_landmark

    # 2. Normalisasi Skala
    # Hitung jarak antara pergelangan tangan (landmark 0, sekarang di [0,0,0])
    # dan pangkal jari tengah (landmark 9) sebagai referensi skala.
    middle_finger_mcp = normalized_landmarks[9]
    
    scale_distance = np.linalg.norm(middle_finger_mcp)

    if scale_distance < 1e-6: # Hindari pembagian dengan nol atau nilai yang sangat kecil
        # print("Peringatan: Jarak skala sangat kecil, normalisasi mungkin tidak stabil.")
        return np.zeros((21, 3), dtype=np.float32)

    normalized_landmarks = normalized_landmarks / scale_distance
    
    return normalized_landmarks

# --- Pengaturan Path Dataset ---
INPUT_DATASET_PATH = "dataset_landmarks_from_images"  # Ganti jika nama folder Anda berbeda
OUTPUT_DATASET_PATH = "dataset_landmarks_sibi_normalized"

# Fungsi untuk membuat folder jika belum ada
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder dibuat: {path}")

# --- Fungsi Utama untuk Memproses Seluruh Dataset ---
def process_entire_dataset(input_base_path, output_base_path):
    """
    Memuat, menormalisasi, dan menyimpan kembali seluruh dataset landmarks.
    """
    create_folder_if_not_exists(output_base_path)
    
    processed_files = 0
    skipped_files_load_error = 0
    skipped_files_empty_sequence = 0

    # Iterasi melalui setiap folder gestur di dalam direktori input
    for gesture_label in os.listdir(input_base_path):
        gesture_input_path = os.path.join(input_base_path, gesture_label)
        gesture_output_path = os.path.join(output_base_path, gesture_label)

        if os.path.isdir(gesture_input_path): # Pastikan itu adalah direktori
            create_folder_if_not_exists(gesture_output_path)
            print(f"\nMemproses gestur: {gesture_label}")

            # Iterasi melalui setiap file .npy di dalam folder gestur
            for filename in os.listdir(gesture_input_path):
                if filename.endswith(".npy"):
                    file_input_path = os.path.join(gesture_input_path, filename)
                    file_output_path = os.path.join(gesture_output_path, filename)

                    try:
                        # 1. Muat data landmarks mentah dari file .npy
                        raw_landmark_sequence = np.load(file_input_path)

                        if raw_landmark_sequence is None or raw_landmark_sequence.ndim != 3 or raw_landmark_sequence.shape[1:] != (21,3):
                            print(f"  -> Melewati (format data tidak sesuai): {filename}. Shape: {raw_landmark_sequence.shape if isinstance(raw_landmark_sequence, np.ndarray) else 'Data tidak valid'}")
                            skipped_files_load_error +=1
                            continue
                        
                        if raw_landmark_sequence.shape[0] == 0: # Cek apakah sekuens kosong
                            print(f"  -> Melewati (sekuens kosong): {filename}")
                            skipped_files_empty_sequence += 1
                            # Simpan file kosong juga di output jika ingin konsisten
                            # np.save(file_output_path, raw_landmark_sequence) 
                            continue

                        # 2. Inisialisasi list untuk menyimpan sekuens yang sudah dinormalisasi
                        normalized_sequence_for_sample = []

                        # 3. Iterasi melalui setiap frame dalam sampel
                        for frame_idx in range(raw_landmark_sequence.shape[0]):
                            landmarks_one_frame = raw_landmark_sequence[frame_idx]
                            normalized_landmarks_frame = normalize_landmarks(landmarks_one_frame.copy()) # Kirim copy
                            normalized_sequence_for_sample.append(normalized_landmarks_frame)
                        
                        # 4. Konversi list menjadi array NumPy kembali
                        normalized_sequence_output = np.array(normalized_sequence_for_sample, dtype=np.float32)

                        # 5. Simpan data yang sudah dinormalisasi
                        np.save(file_output_path, normalized_sequence_output)
                        processed_files += 1
                        if processed_files % 20 == 0: # Cetak progres setiap 20 file
                             print(f"  ... {processed_files} file telah diproses ...")


                    except Exception as e:
                        print(f"  -> Gagal memproses file {filename}: {e}")
                        skipped_files_load_error += 1
                        
    print(f"\n--- Proses Normalisasi Selesai ---")
    print(f"Total file berhasil diproses dan disimpan: {processed_files}")
    print(f"Total file dilewati (error load/format): {skipped_files_load_error}")
    print(f"Total file dilewati (sekuens kosong): {skipped_files_empty_sequence}")
    print(f"Dataset yang sudah dinormalisasi disimpan di: {output_base_path}")


# --- Jalankan Program ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DATASET_PATH):
        print(f"Error: Direktori input dataset '{INPUT_DATASET_PATH}' tidak ditemukan.")
        print("Pastikan Anda sudah membuat dataset landmarks menggunakan skrip perekam_landmarks.py sebelumnya,")
        print("atau ubah variabel INPUT_DATASET_PATH sesuai dengan lokasi dataset Anda.")
    else:
        process_entire_dataset(INPUT_DATASET_PATH, OUTPUT_DATASET_PATH)