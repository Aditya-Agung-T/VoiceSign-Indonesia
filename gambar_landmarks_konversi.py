import cv2
import mediapipe as mp
import numpy as np
import os

# --- Pengaturan Awal ---
INPUT_IMAGE_DATASET_PATH = "SIBI"  # GANTI DENGAN PATH DATASET GAMBAR ANDA
OUTPUT_LANDMARK_DATASET_PATH = "dataset_landmarks_from_images" # Folder output untuk landmarks

# Inisialisasi MediaPipe Hands
# Untuk gambar statis, static_image_mode=True bisa lebih optimal
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, # Penting untuk pemrosesan gambar individual
    max_num_hands=1,        # Asumsi satu tangan per gambar untuk huruf SIBI
    min_detection_confidence=0.5) # Sesuaikan jika perlu

# Fungsi untuk membuat folder jika belum ada
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder dibuat: {path}")

def extract_landmarks_from_image_data(hand_landmarks_instance):
    """Mengekstrak koordinat x, y, z dari objek landmarks tangan MediaPipe."""
    landmarks_list = []
    for landmark in hand_landmarks_instance.landmark:
        landmarks_list.append([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks_list, dtype=np.float32)

# --- Fungsi Utama untuk Memproses Dataset Gambar ---
def process_image_dataset(input_base_path, output_base_path):
    """
    Memuat gambar dari dataset, mengekstrak landmarks, dan menyimpannya.
    """
    if not os.path.exists(input_base_path):
        print(f"Error: Direktori input dataset gambar '{input_base_path}' tidak ditemukan.")
        return

    create_folder_if_not_exists(output_base_path)
    
    processed_images = 0
    skipped_no_hand = 0
    skipped_error_load = 0
    valid_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # Iterasi melalui setiap folder gestur di dalam direktori input gambar
    for gesture_label in os.listdir(input_base_path):
        gesture_input_image_path = os.path.join(input_base_path, gesture_label)
        gesture_output_landmark_path = os.path.join(output_base_path, gesture_label)

        if os.path.isdir(gesture_input_image_path): # Pastikan itu adalah direktori
            create_folder_if_not_exists(gesture_output_landmark_path)
            print(f"\nMemproses gestur: {gesture_label}")

            # Iterasi melalui setiap file gambar di dalam folder gestur
            for image_filename in os.listdir(gesture_input_image_path):
                if image_filename.lower().endswith(valid_image_extensions):
                    image_file_path = os.path.join(gesture_input_image_path, image_filename)
                    
                    # Buat nama file output .npy
                    npy_filename = os.path.splitext(image_filename)[0] + ".npy"
                    landmark_file_output_path = os.path.join(gesture_output_landmark_path, npy_filename)

                    try:
                        # 1. Muat gambar
                        image = cv2.imread(image_file_path)
                        if image is None:
                            print(f"  -> Gagal memuat gambar: {image_filename}, dilewati.")
                            skipped_error_load += 1
                            continue

                        # 2. Ubah ke RGB dan proses dengan MediaPipe
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)

                        # 3. Ekstrak dan simpan landmarks jika terdeteksi
                        if results.multi_hand_landmarks:
                            # Asumsi hanya satu tangan yang relevan per gambar
                            hand_landmarks = results.multi_hand_landmarks[0] 
                            
                            landmarks_data = extract_landmarks_from_image_data(hand_landmarks) # Shape (21, 3)
                            
                            # Karena ini dari gambar statis, kita anggap sebagai sekuens 1 frame
                            # Bentuk output: (1, 21, 3)
                            landmarks_sequence = np.expand_dims(landmarks_data, axis=0) 
                            
                            np.save(landmark_file_output_path, landmarks_sequence)
                            processed_images += 1
                            if processed_images % 50 == 0: # Cetak progres
                                 print(f"  ... {processed_images} gambar telah diproses ...")
                        else:
                            print(f"  -> Tidak ada tangan terdeteksi di: {image_filename}, dilewati.")
                            skipped_no_hand += 1

                    except Exception as e:
                        print(f"  -> Error saat memproses file {image_filename}: {e}")
                        skipped_error_load += 1
                        
    print(f"\n--- Proses Ekstraksi Landmarks dari Gambar Selesai ---")
    print(f"Total gambar berhasil diproses dan landmarks disimpan: {processed_images}")
    print(f"Total gambar dilewati (tidak ada tangan terdeteksi): {skipped_no_hand}")
    print(f"Total gambar dilewati (error load/proses): {skipped_error_load}")
    print(f"Dataset landmarks disimpan di: {output_base_path}")

# --- Jalankan Program ---
if __name__ == "__main__":
    process_image_dataset(INPUT_IMAGE_DATASET_PATH, OUTPUT_LANDMARK_DATASET_PATH)
    
    # Penting: Tutup objek hands MediaPipe jika sudah selesai
    # (terutama jika static_image_mode=False, tapi baik untuk kebiasaan)
    if 'hands' in globals():
        hands.close()