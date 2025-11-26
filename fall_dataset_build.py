
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to your SisFall folder (change if needed)
BASE_PATH = r"xxxxxxx"

def read_file(file_path):
    """Reads one SisFall .txt file and returns accelerometer + gyro arrays."""
    try:
        # Some files use commas and semicolons — clean both
        with open(file_path, "r") as f:
            lines = f.readlines()

        clean_lines = []
        for line in lines:
            line = line.strip().replace(";", "")
            if line:
                clean_lines.append(line)

        # Load using numpy
        data = np.loadtxt(clean_lines, delimiter=",")
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        # Take first 6–9 columns (depending on file)
        df = pd.DataFrame(data)
        if df.shape[1] >= 9:
            df = df.iloc[:, :9]
            df.columns = [
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "extra1", "extra2", "extra3"
            ]
        elif df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        else:
            return None
        return df

    except Exception as e:
        print(f"⚠️ Skipping {file_path} due to error: {e}")
        return None


# -------------------------------------------------------------
# 2️⃣ Build dataset from all folders
# -------------------------------------------------------------
def build_dataset(base_path):
    data_list = []

    # Go through all SAxx and SBxx folders
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Label logic
        if folder.startswith("SA"):
            label = 1  # Fall
        elif folder.startswith("SE") or folder.startswith("SB"):
            label = 0  # Non-Fall
        else:
            continue

        print(f"📂 Processing {folder} ... ({'Fall' if label == 1 else 'Non-Fall'})")

        for file in os.listdir(folder_path):
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(folder_path, file)
            df = read_file(file_path)
            if df is not None and not df.empty:
                # Compute simple statistical features per file
                feature_row = {
                    "acc_x_mean": df["acc_x"].mean(),
                    "acc_y_mean": df["acc_y"].mean(),
                    "acc_z_mean": df["acc_z"].mean(),
                    "gyro_x_mean": df["gyro_x"].mean(),
                    "gyro_y_mean": df["gyro_y"].mean(),
                    "gyro_z_mean": df["gyro_z"].mean(),
                    "acc_mag_mean": np.sqrt((df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)).mean(),
                    "gyro_mag_mean": np.sqrt((df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)).mean(),
                    "label": label
                }
                data_list.append(feature_row)

    dataset = pd.DataFrame(data_list)
    return dataset


# -------------------------------------------------------------
# 3️⃣ Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Building dataset from SisFall...")

    if not os.path.exists(BASE_PATH):
        print(f"❌ Error: Path not found: {BASE_PATH}")
        exit()

    df = build_dataset(BASE_PATH)

    if df.empty:
        print("⚠️ No valid data found! Please check folder structure or file format.")
    else:
        print(f"✅ Completed! Final dataset shape: {df.shape}")

        # Save combined CSV
        save_path = os.path.join(os.path.dirname(BASE_PATH), "sisfall_data.csv")
        df.to_csv(save_path, index=False)
        print(f"💾 Saved combined dataset to: {save_path}")

        # Optional: visualize one signal (for sanity check)
        sample_file = None
        for folder in os.listdir(BASE_PATH):
            if folder.startswith("SA"):
                folder_path = os.path.join(BASE_PATH, folder)
                for file in os.listdir(folder_path):
                    if file.endswith(".txt"):
                        sample_file = os.path.join(folder_path, file)
                        break
                if sample_file:
                    break

        if sample_file:
            sample_df = read_file(sample_file)
            plt.figure(figsize=(10, 4))
            plt.plot(sample_df["acc_x"].values[:300], label="acc_x")
            plt.plot(sample_df["acc_y"].values[:300], label="acc_y")
            plt.plot(sample_df["acc_z"].values[:300], label="acc_z")
            plt.legend()
            plt.title(f"Sample Accelerometer Signal: {os.path.basename(sample_file)}")
            plt.xlabel("Time")
            plt.ylabel("Acceleration")
            plt.show()
