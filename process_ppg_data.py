import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# Function to preprocess PPG signals
def preprocess_ppg_signals(data_folder):
    signal_features = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_folder, file_name)
            signal = np.loadtxt(file_path)

            # Extract subject_ID and segment from file name
            parts = file_name.split("_")
            subject_id = parts[0]
            segment = parts[1].split(".")[0]  # Extract segment (e.g., "1" from "12345_1.txt")

            # Bandpass filter
            def bandpass_filter(signal, lowcut=0.5, highcut=5.0, fs=1000, order=4):
                nyquist = 0.5 * fs
                low = lowcut / nyquist
                high = highcut / nyquist
                b, a = butter(order, [low, high], btype="band")
                return filtfilt(b, a, signal)

            filtered_signal = bandpass_filter(signal)

            # Feature extraction
            features = {
                "subject_ID": subject_id,
                f"mean_segment{segment}": np.mean(filtered_signal),
                f"std_dev_segment{segment}": np.std(filtered_signal),
                f"min_segment{segment}": np.min(filtered_signal),
                f"max_segment{segment}": np.max(filtered_signal),
                f"range_segment{segment}": np.ptp(filtered_signal),
            }
            signal_features.append(features)

    # Combine features from all segments
    features_df = pd.DataFrame(signal_features)
    combined_features = features_df.groupby("subject_ID").first().reset_index()

    return combined_features

# Function to preprocess clinical metadata
def preprocess_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path)

    # Convert subject_ID to string for consistent merging
    metadata["subject_ID"] = metadata["subject_ID"].astype(str)

    # Example preprocessing (modify based on your dataset)
    metadata.dropna(inplace=True)

    return metadata

# Function to combine PPG features with clinical metadata
def combine_features(ppg_features, clinical_metadata):
    # Convert subject_ID in PPG features to string for consistent merging
    ppg_features["subject_ID"] = ppg_features["subject_ID"].astype(str)

    print("PPG Features Columns:", ppg_features.columns)
    print("Clinical Metadata Columns:", clinical_metadata.columns)

    # Perform merging using subject_ID
    combined_data = pd.merge(ppg_features, clinical_metadata, on="subject_ID", how="inner")
    return combined_data

# Main script
if __name__ == "__main__":
    data_folder = r"C:\Users\bhavy\Downloads\5459299\PPG-BP Database\Data File\0_subject"
    clinical_data_path = r"C:\Users\bhavy\Downloads\5459299\PPG-BP Database\PPG-BP dataset.csv"

    print("Preprocessing PPG signals...")
    ppg_features = preprocess_ppg_signals(data_folder)
    print("PPG signal preprocessing complete.")

    print("Preprocessing clinical metadata...")
    clinical_metadata = preprocess_metadata(clinical_data_path)
    print("Clinical metadata preprocessing complete.")

    try:
        print("Combining features...")
        final_dataset = combine_features(ppg_features, clinical_metadata)
        final_dataset.to_csv("final_dataset.csv", index=False)
        print("Final dataset saved to 'final_dataset.csv'.")
    except KeyError as e:
        print(f"Merge failed: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
