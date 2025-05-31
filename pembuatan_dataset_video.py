import cv2
import os
import time

# --- Pengaturan Awal ---
BASE_DATA_PATH = "dataset_video_sibi"  # Folder utama untuk menyimpan dataset
RECORD_DURATION_SECONDS = 3  # Durasi perekaman setiap sampel video (dalam detik)
# Atau, jika Anda ingin mengontrol secara manual dengan tombol Start/Stop, kita bisa sesuaikan

# Fungsi untuk membuat folder jika belum ada
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder dibuat: {path}")

def record_video_sample(gesture_label, subject_id, sample_num):
    """Merekam satu sampel video untuk gestur tertentu."""
    gesture_path = os.path.join(BASE_DATA_PATH, gesture_label)
    create_folder_if_not_exists(gesture_path)

    video_filename = f"{gesture_label}_{subject_id}_{sample_num}.mp4"
    video_filepath = os.path.join(gesture_path, video_filename)

    cap = cv2.VideoCapture(0)  # Buka webcam
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return False

    # Dapatkan properti frame dari webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: # Jika fps tidak terdeteksi, gunakan default
        fps = 20
        print(f"FPS tidak terdeteksi, menggunakan default: {fps}")


    # Tentukan codec dan buat objek VideoWriter
    # 'mp4v' adalah codec yang umum untuk .mp4. Anda mungkin perlu 'XVID' untuk .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))

    print(f"\nBersiap merekam untuk: GESTUR '{gesture_label}', SUBJEK '{subject_id}', SAMPEL {sample_num}")
    print(f"File akan disimpan sebagai: {video_filepath}")

    start_time = None
    recording_started_signal_time = 0

    # Countdown sebelum mulai
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak bisa membaca frame dari kamera saat countdown.")
            cap.release()
            out.release()
            return False
        
        frame = cv2.flip(frame, 1) # Tampilan cermin
        cv2.putText(frame, f"Mulai dalam: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Gestur: {gesture_label}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Perekaman Dataset SIBI", frame)
        cv2.waitKey(1000) # Tunggu 1 detik

    print(f"MULAI MEREKAM! (selama {RECORD_DURATION_SECONDS} detik)")
    start_time = time.time()
    recording_started_signal_time = time.time()


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame.")
            break

        frame_display = cv2.flip(frame.copy(), 1) # Tampilan cermin untuk display

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time <= RECORD_DURATION_SECONDS:
            out.write(frame) # Simpan frame original (belum di-flip)
            
            # Tampilkan indikator MEREKAM
            if current_time - recording_started_signal_time < 1.0 : # Tampilkan "MEREKAM" selama 1 detik pertama
                 cv2.putText(frame_display, "MEREKAM", (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Tampilkan waktu tersisa
            time_left = RECORD_DURATION_SECONDS - elapsed_time
            cv2.putText(frame_display, f"Sisa: {time_left:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Gestur: {gesture_label}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        else: # Waktu perekaman selesai
            print("SELESAI MEREKAM SAMPEL.")
            break
        
        cv2.imshow("Perekaman Dataset SIBI", frame_display)

        # Tekan 'q' untuk menghentikan perekaman sampel saat ini secara manual (jika perlu)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Perekaman dihentikan secara manual.")
            break
            
    cap.release()
    out.release()
    cv2.destroyWindow("Perekaman Dataset SIBI") # Tutup hanya jendela perekaman
    return True


if __name__ == "__main__":
    create_folder_if_not_exists(BASE_DATA_PATH)

    subject_id = input("Masukkan ID Subjek (misal: 'subjek01'): ").strip()
    if not subject_id:
        subject_id = "default_subject"
        print(f"ID Subjek tidak dimasukkan, menggunakan '{subject_id}'")

    while True:
        gesture_label = input("\nMasukkan LABEL GESTUR (huruf SIBI, misal 'A', 'B', atau 'selesai' untuk keluar): ").upper().strip()
        if gesture_label == "SELESAI":
            break
        if not gesture_label:
            print("Label gestur tidak boleh kosong.")
            continue

        # Cari tahu nomor sampel berikutnya untuk gestur ini
        sample_num = 1
        gesture_folder = os.path.join(BASE_DATA_PATH, gesture_label)
        create_folder_if_not_exists(gesture_folder) # Pastikan folder gestur ada
        
        # Hitung file yang sudah ada untuk menentukan nomor sampel berikutnya
        existing_files = [f for f in os.listdir(gesture_folder) if f.startswith(f"{gesture_label}_{subject_id}_") and f.endswith(".mp4")]
        sample_num = len(existing_files) + 1
        
        if record_video_sample(gesture_label, subject_id, sample_num):
            print(f"Sampel {sample_num} untuk gestur '{gesture_label}' berhasil direkam.")
        else:
            print(f"Gagal merekam sampel {sample_num} untuk gestur '{gesture_label}'.")

        next_action = input("Lanjut rekam gestur berikutnya? (y/n): ").lower().strip()
        if next_action != 'y':
            break
            
    print("\nProses pengumpulan data selesai.")
    cv2.destroyAllWindows() # Pastikan semua jendela tertutup di akhir