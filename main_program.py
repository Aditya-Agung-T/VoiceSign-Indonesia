import cv2
import mediapipe as mp
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Impor untuk gTTS dan Pygame ---
from gtts import gTTS
import pygame # Menggantikan playsound
# ------------------------------------

# --- Pengaturan Awal & Konstanta ---
MODEL_PATH = 'sibi_lstm_model_best.keras'
LABEL_ENCODER_CLASSES_PATH = 'sibi_label_encoder_classes.npy'

try:
    temp_model = tf.keras.models.load_model(MODEL_PATH)
    MAX_SEQ_LENGTH = temp_model.input_shape[1]
    NUM_FEATURES = temp_model.input_shape[2]
    print(f"Model berhasil dimuat. MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}, NUM_FEATURES: {NUM_FEATURES}")
    del temp_model
except Exception as e:
    print(f"Tidak bisa memuat model untuk deteksi MAX_SEQ_LENGTH: {e}. Menggunakan default.")
    MAX_SEQ_LENGTH = 60 
    NUM_FEATURES = 63   

# --- Fungsi Normalisasi Landmarks (HARUS SAMA DENGAN SAAT TRAINING) ---
def normalize_landmarks(landmarks_data):
    if landmarks_data is None or not isinstance(landmarks_data, np.ndarray) or landmarks_data.shape != (21, 3):
        return np.zeros((21, 3), dtype=np.float32)
    wrist_landmark = landmarks_data[0].copy()
    normalized_landmarks = landmarks_data - wrist_landmark
    middle_finger_mcp = normalized_landmarks[9]
    scale_distance = np.linalg.norm(middle_finger_mcp)
    if scale_distance < 1e-6: return np.zeros((21, 3), dtype=np.float32)
    return normalized_landmarks / scale_distance

# --- Fungsi untuk gTTS dengan Pygame Mixer ---
pygame_mixer_initialized = False # Variabel global untuk status init mixer pygame

def speak_with_gtts_pygame(text_to_speak, lang='id', audio_filename="temp_speech.mp3"):
    """Mengucapkan teks menggunakan gTTS dan memutarnya dengan pygame.mixer."""
    global pygame_mixer_initialized
    try:
        print(f"gTTS: Membuat audio untuk '{text_to_speak}'...")
        tts = gTTS(text=text_to_speak, lang=lang, slow=False)
        tts.save(audio_filename)
        
        print(f"Pygame: Memutar audio '{audio_filename}'...")
        
        if not pygame_mixer_initialized:
            pygame.mixer.init() # Inisialisasi mixer pygame jika belum
            pygame_mixer_initialized = True
            print("Pygame mixer diinisialisasi.")

        pygame.mixer.music.load(audio_filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy(): # Tunggu audio selesai
            pygame.time.Clock().tick(10) 
            
        print("Pygame: Pemutaran audio selesai.")
        pygame.mixer.music.unload() # Unload file

        if os.path.exists(audio_filename): # Hapus file audio
            try:
                os.remove(audio_filename)
            except PermissionError:
                print(f"Peringatan: Tidak bisa menghapus '{audio_filename}', mungkin masih digunakan.")
            except Exception as e_rem:
                print(f"Error saat menghapus file audio: {e_rem}")
            
    except ConnectionError:
        print("gTTS Error: Tidak ada koneksi internet untuk menghasilkan suara.")
    except pygame.error as pg_err:
        print(f"Pygame mixer error: {pg_err}")
        if pygame_mixer_initialized: # Coba uninitialize jika error
            pygame.mixer.quit()
            pygame_mixer_initialized = False
    except Exception as e:
        print(f"gTTS Error atau Error saat memutar audio lainnya: {e}")
# ---------------------------------------------

print(f"Memuat model dari: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model berhasil dimuat.")

if not os.path.exists(LABEL_ENCODER_CLASSES_PATH):
    print(f"Error: File kelas LabelEncoder '{LABEL_ENCODER_CLASSES_PATH}' tidak ditemukan!")
    exit()
loaded_encoder_classes = np.load(LABEL_ENCODER_CLASSES_PATH, allow_pickle=True)
print(f"Kelas LabelEncoder berhasil dimuat: {loaded_encoder_classes}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Tidak bisa membuka kamera."); exit()

# Variabel Logika
current_sequence_landmarks = []
accumulated_letters = ""
last_predicted_char_display = "..."
last_valid_char_buffer = None 
last_hand_detection_time = time.time()

# Parameter yang bisa disesuaikan
PAUSE_THRESHOLD_SECONDS_FOR_SPEECH = 2.0 
MIN_FRAMES_FOR_GESTURE_COMPLETION = 15  
MIN_FRAMES_FOR_CONTINUOUS_PREDICTION = 15 
CONFIDENCE_THRESHOLD = 0.65 
confirm_window_start = 0.1 
confirm_window_end = PAUSE_THRESHOLD_SECONDS_FOR_SPEECH * 0.8 

prev_time_fps = 0
print("\n--- APLIKASI PENERJEMAH SIBI REAL-TIME (gTTS dengan Pygame) ---")
print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_time = time.time()
    fps = 1 / (current_time - prev_time_fps) if (current_time - prev_time_fps) > 0 else 0
    prev_time_fps = current_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    hand_detected_this_frame = False
    
    if results.multi_hand_landmarks:
        hand_detected_this_frame = True
        last_hand_detection_time = current_time
        
        for hand_landmarks_instance in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks_instance, mp_hands.HAND_CONNECTIONS,
                                      mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                      mp.solutions.drawing_styles.get_default_hand_connections_style())
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_instance.landmark])
            current_sequence_landmarks.append(frame_landmarks)

        if len(current_sequence_landmarks) > MAX_SEQ_LENGTH * 2:
            current_sequence_landmarks = current_sequence_landmarks[-MAX_SEQ_LENGTH * 2:]

        if len(current_sequence_landmarks) >= MIN_FRAMES_FOR_CONTINUOUS_PREDICTION:
            sequence_to_predict_on = current_sequence_landmarks[-MAX_SEQ_LENGTH:]
            
            normalized_seq = [normalize_landmarks(fl.copy()) for fl in sequence_to_predict_on]
            reshaped_seq = [ns_lm.reshape(-1) for ns_lm in normalized_seq]
            padded_sequence = pad_sequences([np.array(reshaped_seq)], maxlen=MAX_SEQ_LENGTH, 
                                            padding='pre', truncating='pre', dtype='float32')

            if padded_sequence.shape == (1, MAX_SEQ_LENGTH, NUM_FEATURES):
                prediction_probabilities = model.predict(padded_sequence, verbose=0)
                predicted_class_index = np.argmax(prediction_probabilities[0])
                prediction_confidence = prediction_probabilities[0][predicted_class_index]

                if prediction_confidence >= CONFIDENCE_THRESHOLD:
                    predicted_sibi_char = loaded_encoder_classes[predicted_class_index]
                    last_predicted_char_display = f"{predicted_sibi_char} ({prediction_confidence*100:.0f}%)"
                    last_valid_char_buffer = predicted_sibi_char 
                else:
                    last_predicted_char_display = f"? ({prediction_confidence*100:.0f}%)"
            else:
                last_predicted_char_display = "ERR_SHAPE"
        elif len(current_sequence_landmarks) > 0 :
             last_predicted_char_display = "..."
    
    if not hand_detected_this_frame:
        time_since_last_detection = current_time - last_hand_detection_time
        
        if last_valid_char_buffer is not None and \
           (not accumulated_letters or accumulated_letters[-1] != last_valid_char_buffer) and \
           (confirm_window_start < time_since_last_detection < confirm_window_end) :
            
            accumulated_letters += last_valid_char_buffer
            print(f"Huruf ditambahkan: '{last_valid_char_buffer}'. Kata sekarang: '{accumulated_letters}'")
            last_predicted_char_display = last_valid_char_buffer 
            last_valid_char_buffer = None 
            current_sequence_landmarks = []

        elif len(current_sequence_landmarks) >= MIN_FRAMES_FOR_GESTURE_COMPLETION and \
             time_since_last_detection > 0.5 : 
            if len(current_sequence_landmarks) > 0:
                 print(f"Tangan hilang, sekuens (panjang {len(current_sequence_landmarks)}) dibersihkan tanpa menambah huruf.")
                 current_sequence_landmarks = []
            if last_valid_char_buffer is None and last_predicted_char_display != "...":
                 last_predicted_char_display = "?"
        
        if accumulated_letters and time_since_last_detection > PAUSE_THRESHOLD_SECONDS_FOR_SPEECH:
            print(f"Jeda panjang. Output kata: {accumulated_letters}")
            cv2.putText(image, f"Output: {accumulated_letters}", (image_width // 2 - 100, image_height - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Penerjemah SIBI - Mode Langsung (gTTS pygame) (q: keluar)', image)
            cv2.waitKey(1) 

            speak_with_gtts_pygame(accumulated_letters, lang='id') # PANGGIL FUNGSI gTTS DENGAN PYGAME
            
            accumulated_letters = ""
            current_sequence_landmarks = []
            last_valid_char_buffer = None
            last_predicted_char_display = "..."
        
        elif not accumulated_letters and len(current_sequence_landmarks) == 0 and \
             time_since_last_detection > 0.5 and last_predicted_char_display != "...":
                last_predicted_char_display = "..."
                last_valid_char_buffer = None

    cv2.putText(image, f"Huruf: {last_predicted_char_display}", (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(image, f"Kata: {accumulated_letters}", (10, 110), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255, 0, 0), 2)
    if not hand_detected_this_frame and not accumulated_letters and len(current_sequence_landmarks) == 0 and last_predicted_char_display == "..." :
         cv2.putText(image, "Arahkan tangan membentuk huruf SIBI", (image_width // 2 - 300, image_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Penerjemah SIBI - Mode Langsung (gTTS pygame) (q: keluar)', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close(); cap.release(); cv2.destroyAllWindows()
# Pastikan mixer pygame juga di-quit saat aplikasi ditutup (jika sudah diinisialisasi)
if pygame_mixer_initialized:
    pygame.mixer.quit()
    print("Pygame mixer di-uninitialize.")

print("Aplikasi ditutup.")
if accumulated_letters: print(f"Sisa kata belum diucapkan: {accumulated_letters}")