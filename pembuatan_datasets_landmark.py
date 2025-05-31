import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Pengaturan Awal ---
BASE_DATA_PATH = "Datasets_landmarks"  # Folder utama untuk menyimpan dataset landmarks
RECORD_DURATION_SECONDS = 2  # Durasi perekaman setiap sampel (dalam detik)
# Sesuaikan durasi ini; 2-3 detik biasanya cukup untuk satu huruf

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Fokus pada satu tangan untuk SIBI huruf
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # Untuk menggambar landmarks

# Fungsi untuk membuat folder jika belum ada
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder dibuat: {path}")

def extract_landmarks(hand_landmarks):
    """Mengekstrak koordinat x, y, z dari landmarks tangan."""
    landmarks_list = []
    for landmark in hand_landmarks.landmark:
        landmarks_list.append([landmark.x, landmark.y, landmark.z])
    return landmarks_list

def record_landmark_sequence(gesture_label, subject_id, sample_num):
    """Merekam satu sampel sekuens landmarks untuk gestur tertentu."""
    gesture_path = os.path.join(BASE_DATA_PATH, gesture_label)
    create_folder_if_not_exists(gesture_path)

    # Nama file untuk menyimpan data landmarks (format .npy)
    landmark_filename = f"{gesture_label}_{subject_id}_{sample_num}.npy"
    landmark_filepath = os.path.join(gesture_path, landmark_filename)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return False

    print(f"\nBersiap merekam LANDMARKS untuk: GESTUR '{gesture_label}', SUBJEK '{subject_id}', SAMPEL {sample_num}")
    print(f"Data akan disimpan sebagai: {landmark_filepath}")

    recorded_sequence = [] # List untuk menyimpan landmarks dari setiap frame

    # Countdown sebelum mulai
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak bisa membaca frame dari kamera saat countdown.")
            cap.release()
            return False
        
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Mulai dalam: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Gestur: {gesture_label}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Perekaman Dataset Landmarks SIBI", frame)
        cv2.waitKey(1000)

    print(f"MULAI MEREKAM LANDMARKS! (selama {RECORD_DURATION_SECONDS} detik)")
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame.")
            break

        frame_display = cv2.flip(frame.copy(), 1) # Untuk tampilan
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe perlu RGB
        
        results = hands.process(image_rgb) # Deteksi tangan

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time <= RECORD_DURATION_SECONDS:
            if results.multi_hand_landmarks:
                for hand_landmarks_instance in results.multi_hand_landmarks: # Seharusnya hanya satu tangan
                    # Gambar landmarks di frame display
                    mp_drawing.draw_landmarks(
                        frame_display,
                        hand_landmarks_instance,
                        mp_hands.HAND_CONNECTIONS)
                    
                    # Ekstrak dan simpan landmarks
                    landmarks_for_frame = extract_landmarks(hand_landmarks_instance)
                    recorded_sequence.append(landmarks_for_frame)
            else:
                # Jika tidak ada tangan terdeteksi, kita bisa tambahkan array kosong atau 0
                # atau skip. Untuk SIBI huruf, tangan harusnya selalu ada.
                # Jika ingin data yang lebih bersih, pastikan tangan selalu dalam frame.
                # Untuk kesederhanaan, jika tangan tidak terdeteksi, frame ini tidak menambahkan data landmarks.
                # Alternatif: tambahkan np.zeros((21, 3)) untuk menandakan tidak ada deteksi.
                pass

            time_left = RECORD_DURATION_SECONDS - elapsed_time
            cv2.putText(frame_display, f"Sisa: {time_left:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Gestur: {gesture_label}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if results.multi_hand_landmarks:
                 cv2.putText(frame_display, "LANDMARKS TEREKAM", (frame_display.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else: # Waktu perekaman selesai
            print("SELESAI MEREKAM SAMPEL LANDMARKS.")
            break
        
        cv2.imshow("Perekaman Dataset Landmarks SIBI", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Perekaman dihentikan secara manual.")
            break
            
    cap.release()
    cv2.destroyWindow("Perekaman Dataset Landmarks SIBI")

    if recorded_sequence:
        np.save(landmark_filepath, np.array(recorded_sequence))
        print(f"Data landmarks disimpan ke {landmark_filepath}")
        return True
    else:
        print("Tidak ada landmarks yang terekam untuk sampel ini.")
        return False

if __name__ == "__main__":
    create_folder_if_not_exists(BASE_DATA_PATH)

    subject_id = input("Masukkan ID Subjek (misal: 'subjek01'): ").strip()
    if not subject_id:
        subject_id = "default_subject"
        print(f"ID Subjek tidak dimasukkan, menggunakan '{subject_id}'")

    while True:
        gesture_label = input("\nMasukkan LABEL GESTUR SIBI (misal 'A', 'B', atau 'selesai' untuk keluar): ").upper().strip()
        if gesture_label == "SELESAI":
            break
        if not gesture_label:
            print("Label gestur tidak boleh kosong.")
            continue

        sample_num = 1
        gesture_folder = os.path.join(BASE_DATA_PATH, gesture_label)
        create_folder_if_not_exists(gesture_folder)
        
        existing_files = [f for f in os.listdir(gesture_folder) if f.startswith(f"{gesture_label}_{subject_id}_") and f.endswith(".npy")]
        sample_num = len(existing_files) + 1
        
        if record_landmark_sequence(gesture_label, subject_id, sample_num):
            print(f"Sampel landmarks {sample_num} untuk gestur '{gesture_label}' berhasil direkam.")
        else:
            print(f"Gagal merekam sampel landmarks {sample_num} untuk gestur '{gesture_label}'.")

        next_action = input("Lanjut rekam gestur berikutnya? (y/n): ").lower().strip()
        if next_action != 'y':
            break
            
    print("\nProses pengumpulan data landmarks selesai.")
    cv2.destroyAllWindows()