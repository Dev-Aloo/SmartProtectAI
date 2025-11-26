import threading
import requests
import time
import numpy as np
import joblib
import pandas as pd
import sounddevice as sd
import librosa
import warnings
import os
import pygame
from datetime import datetime

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

warnings.filterwarnings("ignore")


PHY_URL = "IP"   # Phyphox IP
MODEL_PATH = r"Your model path"
SCALER_PATH = r"scaler model path"
SCREAM_MODEL_PATH = r"scream model path"
ALERT_SOUND = r"Sound path"

EMERGENCY_CONTACTS = [
    "XXXXXXXXXX"
]

TWILIO_ACCOUNT_SID = "YOUR_TWILIO_SID_HERE"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN_HERE"
TWILIO_FROM_NUMBER = "Your Twilio Provided no."  


SOS_COOLDOWN = 300  

def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def log(tag, msg):
    print(f"[{now_str()}] {tag} {msg}")

# -----------------------
# LOAD MODELS
# -----------------------
log("INFO", "Loading models...")
fall_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
scream_model = joblib.load(SCREAM_MODEL_PATH)
log("INFO", "Models loaded.")

_PYGAME_INITIALIZED = False
try:
    pygame.mixer.init()
    _PYGAME_INITIALIZED = True
    log("INFO", "Pygame mixer initialized.")
except Exception as e:
    log("WARN", f"Failed to init pygame mixer: {e}. Alerts may not play.")

_play_lock = threading.Lock()

def play_alert_sound(sound_path=ALERT_SOUND):
    """Play alert sound non-blocking. Uses a lock to avoid rapid re-loading collisions."""
    if not _PYGAME_INITIALIZED:
        log("WARN", "Pygame mixer not available; cannot play alert.")
        return
    if not os.path.exists(sound_path):
        log("WARN", f"Alert sound not found: {sound_path}")
        return

    def _play():
        with _play_lock:
            try:
                # Stop current playback and play new sound
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
                log("ALERT", f"Playing alert sound: {os.path.basename(sound_path)}")
            except Exception as e:
                log("WARN", f"Error during alert playback: {e}")

    threading.Thread(target=_play, daemon=True).start()


_last_sos_time = 0
_twilio_client = None

def init_twilio_client():
    global _twilio_client
    if not TWILIO_AVAILABLE:
        log("WARN", "Twilio package not installed; SMS disabled.")
        return
    sid = TWILIO_ACCOUNT_SID.strip()
    token = TWILIO_AUTH_TOKEN.strip()
    if sid and token:
        _twilio_client = TwilioClient(sid, token)
        log("INFO", "Twilio client initialized.")
    else:
        log("WARN", "Twilio credentials not provided; SMS disabled.")

def get_location_ipinfo():
    """Return (lat, lon, description) or (None, None, msg)"""
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        if r.status_code == 200:
            j = r.json()
            loc = j.get("loc")
            city = j.get("city")
            region = j.get("region")
            country = j.get("country")
            if loc:
                lat, lon = loc.split(",")
                desc = ", ".join([p for p in (city, region, country) if p])
                return float(lat), float(lon), desc
        return None, None, "Location unavailable"
    except Exception as e:
        return None, None, f"Location error: {e}"

def send_sos_message(event_type="ALERT", extra_text=""):
    """
    Send SOS to emergency contacts. This version is optimized for Twilio Trial accounts
    by splitting long messages into multiple smaller chunks (<=150 chars each).
    """
    global _last_sos_time
    now = time.time()
    if now - _last_sos_time < SOS_COOLDOWN:
        print("⚠️ SOS cooldown active, will not send another SOS yet.")
        return

    _last_sos_time = now
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lat, lon, desc = get_location_ipinfo()

    if lat and lon:
        map_link = f"https://maps.google.com/?q={lat:.4f},{lon:.4f}"
        loc_short = f"{lat:.4f},{lon:.4f}"
    else:
        map_link = "Location unavailable"
        loc_short = desc

    # Build the core message (short, concise)
    msg_header = f"🚨 SmartProtectAI: {event_type}"
    msg_details = f"Time: {ts}\nLoc: {loc_short}\n{desc}\n{extra_text}"
    msg_map = f"Map: {map_link}"

    messages_to_send = [msg_header, msg_details, msg_map]

    print("[SOS] Prepared multi-part SOS message:")
    for part in messages_to_send:
        print(part)
        print("-" * 40)

    # Send via Twilio if configured
    if _twilio_client and TWILIO_FROM_NUMBER and EMERGENCY_CONTACTS:
        for to_num in EMERGENCY_CONTACTS:
            for part in messages_to_send:
                try:
                    msg = _twilio_client.messages.create(
                        body=part,
                        from_=TWILIO_FROM_NUMBER,
                        to=to_num
                    )
                    print(f"✅ Sent part to {to_num} (SID: {msg.sid})")
                    time.sleep(1)  # avoid flooding
                except Exception as e:
                    print(f"⚠️ Failed to send SMS to {to_num}: {e}")
    else:
        print("⚠️ Twilio not configured or no emergency contacts; SMS not sent.")


# Initialize Twilio client (if creds)
init_twilio_client()

# -----------------------
# FALL DETECTION THREAD
# -----------------------
def fall_detection():
    log("Fall", "Starting fall detection (Phyphox)...")
    window_size = 25
    acc_data = []
    last_fall_time = 0
    fall_cooldown = 3.0
    fall_magnitude_threshold = 18.0
    motionless_threshold = 1.5

    while True:
        try:
            resp = requests.get(f"{PHY_URL}/get?accX&accY&accZ&n=10", timeout=2)
            data = resp.json()
            accX = data["buffer"]["accX"]["buffer"]
            accY = data["buffer"]["accY"]["buffer"]
            accZ = data["buffer"]["accZ"]["buffer"]

            if not accX or not accY or not accZ:
                log("[Fall]", "No data from Phyphox.")
                time.sleep(0.5)
                continue

            ax, ay, az = accX[-1], accY[-1], accZ[-1]
            acc_data.append([ax, ay, az])
            if len(acc_data) > window_size:
                acc_data = acc_data[-window_size:]

            log("[Fall]", f"ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}")

            acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

            if len(acc_data) == window_size:
                arr = np.array(acc_data)
                acc_mag_series = np.sqrt(np.sum(arr**2, axis=1))

                features = np.array([[
                    np.mean(arr[:, 0]),
                    np.mean(arr[:, 1]),
                    np.mean(arr[:, 2]),
                    np.std(arr[:, 0]),
                    np.std(arr[:, 1]),
                    np.std(arr[:, 2]),
                    np.mean(acc_mag_series),
                    np.std(acc_mag_series)
                ]])

                features_df = pd.DataFrame(features, columns=scaler.feature_names_in_)
                features_scaled = scaler.transform(features_df)
                pred = fall_model.predict(features_scaled)[0]

                now = time.time()

                # --- Strong fall detection ---
                if (
                    pred == 1
                    and acc_mag > fall_magnitude_threshold
                    and np.mean(acc_mag_series[-5:]) < motionless_threshold
                    and (now - last_fall_time > fall_cooldown)
                ):
                    log("[Fall]", "🚨 CONFIRMED FALL (Model + High impact + Stillness)")
                    play_alert_sound(ALERT_SOUND)
                    threading.Thread(
                        target=send_sos_message,
                        args=("FALL", f"Peak magnitude: {acc_mag:.2f}"),
                        daemon=True
                    ).start()
                    last_fall_time = now

                # --- Sudden high movement fallback alert ---
                elif acc_mag > 15 and (now - last_fall_time > fall_cooldown):
                    log("[Fall]", "⚠️ Strong movement! Triggering safety alert.")
                    play_alert_sound(ALERT_SOUND)
                    threading.Thread(
                        target=send_sos_message,
                        args=("SUDDEN MOTION", f"Impact magnitude: {acc_mag:.2f}"),
                        daemon=True
                    ).start()
                    last_fall_time = now

                # --- Minor motion ---
                elif acc_mag > 12:
                    log("[Fall]", "⚠️ Sudden movement detected.")
                else:
                    log("[Fall]", "✅ Normal activity")


            time.sleep(0.25)

        except KeyboardInterrupt:
            log("Fall", "KeyboardInterrupt - exiting fall detection thread.")
            break
        except Exception as e:
            log("Fall", f"Error: {e}")
            time.sleep(1)

# -----------------------
# SCREAM DETECTION THREAD
# -----------------------
def extract_features_from_audio(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

def scream_detection():
    log("Scream", "Starting scream detection (mic)...")
    duration = 2
    threshold = 0.65

    while True:
        try:
            audio = sd.rec(int(22050 * duration), samplerate=22050, channels=1)
            sd.wait()
            audio = np.squeeze(audio)

            features = extract_features_from_audio(audio, 22050)
            probs = scream_model.predict_proba(features)[0]
            pred = np.argmax(probs)
            conf = probs[pred]

            if pred == 1 and conf > threshold:
                log("[Scream]", f"No scream (conf: {conf:.2f})")
            else:
                log("[Scream]", f"🚨 SCREAM DETECTED! (conf: {conf:.2f})")
                play_alert_sound(ALERT_SOUND)
                threading.Thread(target=send_sos_message, args=("SCREAM", f"Confidence: {conf:.2f}"), daemon=True).start()

        except KeyboardInterrupt:
            log("Scream", "KeyboardInterrupt - exiting scream detection thread.")
            break
        except Exception as e:
            log("Scream", f"Error: {e}")
            time.sleep(1)

# -----------------------
# LAUNCH
# -----------------------
if __name__ == "__main__":
    try:
        t_fall = threading.Thread(target=fall_detection, daemon=True)
        t_scream = threading.Thread(target=scream_detection, daemon=True)
        t_fall.start()
        t_scream.start()

        log("MAIN", "SmartProtectAI running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        log("MAIN", "Ctrl+C received — shutting down.")
        # threads are daemon; program will exit
