import requests
import time
import numpy as np
import joblib
import pandas as pd

# === Configuration ===
PHY_URL = "http://10.12.101.237:8080"
MODEL_PATH = r"C:\Users\Devraj Singh\Desktop\SmartProtectAI\fall_model_tuned.pkl"
SCALER_PATH = r"C:\Users\Devraj Singh\Desktop\SmartProtectAI\scaler_tuned.pkl"

# === Load model & scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded successfully!")
print("📡 Starting live data stream... Move your phone to test.")

# === Parameters ===
window_size = 25               # longer window for better stability
acc_data = []
last_fall_time = 0
fall_cooldown = 3.0            # seconds between fall detections
fall_magnitude_threshold = 18  # higher spike threshold (since SIS-Fall data is high-impact)
motionless_threshold = 1.5     # detect if phone is still after spike (m/s²)

while True:
    try:
        resp = requests.get(f"{PHY_URL}/get?accX&accY&accZ&n=10", timeout=2)
        data = resp.json()
        accX = data["buffer"]["accX"]["buffer"]
        accY = data["buffer"]["accY"]["buffer"]
        accZ = data["buffer"]["accZ"]["buffer"]

        if not accX or not accY or not accZ:
            print("⏸️ No new data from Phyphox.")
            time.sleep(0.5)
            continue

        ax, ay, az = accX[-1], accY[-1], accZ[-1]
        print(f"📊 ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}")
        acc_data.append([ax, ay, az])
        if len(acc_data) > window_size:
            acc_data = acc_data[-window_size:]

        # Compute magnitude
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

        # Once we have enough samples, run model
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
            pred = model.predict(features_scaled)[0]

            # Hybrid condition for fall (SIS-Fall model logic)
            now = time.time()
            if (
                pred == 1
                and acc_mag > fall_magnitude_threshold
                and np.mean(acc_mag_series[-5:]) < motionless_threshold
                and (now - last_fall_time > fall_cooldown)
            ):
                print("🚨 FALL DETECTED! (High impact + stillness)")
                last_fall_time = now
            elif acc_mag > 12:
                print("⚠️ Sudden movement detected.")
            else:
                print("✅ Normal activity")

        time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n🛑 Exiting stream.")
        break
    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(1)
