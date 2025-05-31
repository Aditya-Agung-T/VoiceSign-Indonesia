import cv2
import mediapipe as mp
import numpy as np
import os

# --- Pengaturan Awal ---
INPUT_VIDEO_DATASET_PATH = "dataset_video_sibi"  # GANTI DENGAN PATH DATASET VIDEO ANDA
OUTPUT_LANDMARK_DATASET_PATH = "dataset_landmarks_from_video" # Folder output

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, # Kita memproses video stream (frame per frame)
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Fungsi untuk membuat folder jika belum ada
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder dibuat: {path}")

def extract_landmarks_from_hand(hand_landmarks_instance):
    """Mengekstrak koordinat x, y, z dari objek landmarks tangan MediaPipe."""
    landmarks_list = []
    for landmark in hand_landmarks_instance.landmark:
        landmarks_list.append([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks_list, dtype=np.float32)

# --- Fungsi Utama untuk Memproses Dataset Video ---
def process_video_dataset_to_landmarks(input_base_path, output_base_path):
    if not os.path.exists(input_base_path):
        print(f"Error: Direktori input dataset video '{input_base_path}' tidak ditemukan.")
        return

    create_folder_if_not_exists(output_base_path)
    
    processed_videos = 0
    skipped_videos_no_hand_mostly = 0 # Video dilewati jika sebagian besar frame tidak ada tangan
    skipped_videos_error_open = 0
    valid_video_extensions = ('.mp4', '.avi', '.mov', '.mkv') # Tambahkan ekstensi lain jika perlu

    for gesture_label in os.listdir(input_base_path):
        gesture_input_video_path = os.path.join(input_base_path, gesture_label)
        gesture_output_landmark_path = os.path.join(output_base_path, gesture_label)

        if os.path.isdir(gesture_input_video_path):
            create_folder_if_not_exists(gesture_output_landmark_path)
            print(f"\nMemproses gestur: {gesture_label}")

            for video_filename in os.listdir(gesture_input_video_path):
                if video_filename.lower().endswith(valid_video_extensions):
                    video_file_path = os.path.join(gesture_input_video_path, video_filename)
                    
                    npy_filename = os.path.splitext(video_filename)[0] + ".npy"
                    landmark_file_output_path = os.path.join(gesture_output_landmark_path, npy_filename)

                    # Jika file .npy sudah ada, lewati untuk menghemat waktu (opsional)
                    # if os.path.exists(landmark_file_output_path):
                    #     print(f"  File landmarks '{npy_filename}' sudah ada, dilewati.")
                    #     processed_videos +=1 # Anggap sudah diproses sebelumnya
                    #     continue

                    cap = cv2.VideoCapture(video_file_path)
                    if not cap.isOpened():
                        print(f"  -> Gagal membuka video: {video_filename}, dilewati.")
                        skipped_videos_error_open += 1
                        continue

                    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_video_landmark_sequence = []
                    frames_with_hand = 0

                    print(f"  Memproses video: {video_filename} ({video_total_frames} frames)")
                    
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret: # Akhir video atau error baca frame
                            break
                        
                        frame_count += 1
                        if frame_count % 100 == 0 : # Progress per 100 frame
                             print(f"    Frame {frame_count}/{video_total_frames}...")

                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)

                        if results.multi_hand_landmarks:
                            frames_with_hand += 1
                            # Ambil tangan pertama yang terdeteksi
                            hand_landmarks = results.multi_hand_landmarks[0]
                            landmarks_for_frame = extract_landmarks_from_hand(hand_landmarks)
                            current_video_landmark_sequence.append(landmarks_for_frame)
                        else:
                            # Jika tidak ada tangan terdeteksi, kita bisa:
                            # 1. Menambahkan array nol (21,3) agar panjang sekuens tetap sama dengan jumlah frame video
                            # current_video_landmark_sequence.append(np.zeros((21,3), dtype=np.float32))
                            # 2. Mengabaikan frame ini (sekuens .npy akan lebih pendek dari total frame video)
                            # Untuk konsistensi dengan perekaman webcam yang hanya menyimpan saat ada tangan,
                            # kita akan mengabaikan frame tanpa tangan.
                            pass
                    
                    cap.release()

                    # Simpan sekuens jika ada cukup frame dengan tangan terdeteksi
                    # (misalnya, minimal 10% dari total frame atau minimal N frame absolut)
                    if current_video_landmark_sequence and frames_with_hand > 0 :
                        # Minimal 5 frame dengan tangan, atau 10% dari total frame
                        min_hand_frames_threshold = max(5, int(video_total_frames * 0.10))
                        if frames_with_hand >= min_hand_frames_threshold:
                            np.save(landmark_file_output_path, np.array(current_video_landmark_sequence, dtype=np.float32))
                            processed_videos += 1
                            print(f"    Landmarks disimpan ke '{npy_filename}' ({len(current_video_landmark_sequence)} frame landmarks).")
                        else:
                            print(f"    -> Dilewati (terlalu sedikit frame dengan tangan: {frames_with_hand}/{video_total_frames}) untuk: {video_filename}")
                            skipped_videos_no_hand_mostly += 1
                    else:
                        print(f"    -> Dilewati (tidak ada landmarks yang diekstrak) untuk: {video_filename}")
                        skipped_videos_no_hand_mostly += 1
                        
    print(f"\n--- Proses Ekstraksi Landmarks dari Dataset Video Selesai ---")
    print(f"Total video berhasil diproses dan landmarks disimpan: {processed_videos}")
    print(f"Total video dilewati (error buka): {skipped_videos_error_open}")
    print(f"Total video dilewati (sedikit/tidak ada tangan terdeteksi): {skipped_videos_no_hand_mostly}")
    print(f"Dataset landmarks disimpan di: {output_base_path}")

# --- Jalankan Program ---
if __name__ == "__main__":
    process_video_dataset_to_landmarks(INPUT_VIDEO_DATASET_PATH, OUTPUT_LANDMARK_DATASET_PATH)
    if 'hands' in globals(): # Tutup objek MediaPipe
        hands.close()