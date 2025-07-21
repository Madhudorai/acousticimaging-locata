import os 
from tqdm import tqdm
import csv 
import numpy as np 
import glob 
import pandas as pd 
import argparse

from tools import load_and_process_audio, setup_frequency_bands, compute_dynamic_ranges 
from microphone import setup_microphone_array
from beamform import setup_beamforming_directions,DAS_frequency_band, MUSIC_frequency_band, DAMAS_frequency_band,compute_time_delays
from evaluate import add_gt_to_predictions, angular_error

def main(file_path, metadata_path, band_indices, methodtype, task, acousticimagingalgo, num_channels):
    c = 343.0  
    fs = 48000 
    npoints = 1024

    if num_channels == 4:
        selected_mics = [5, 9, 25, 21]
    else:
        selected_mics = None

    positions = setup_microphone_array(selected_mics)
    unit_vectors, azimuths, elevations, num_dirs = setup_beamforming_directions(n_points=npoints)
    delays = compute_time_delays(positions, unit_vectors, c)
    stfts, freqs, num_frames, nperseg, noverlap = load_and_process_audio(file_path, selected_channels=selected_mics)
 
    # precompute freq bands and dynamic ranges ONCE
    N_MEL_BANDS = 6
    freq_bands, band_names = setup_frequency_bands(fs, 4800, N_MEL_BANDS)
    band_dynamic_ranges = compute_dynamic_ranges(stfts, delays, freqs, freq_bands, num_frames)

    max_frames = num_frames

    parts = file_path.split(os.sep)
    split_name = parts[1]  # 'dev' or 'eval'
    recording_name = [p for p in parts if p.startswith("recording")][0]
    basename = f"{split_name}_{recording_name}"


    band_str = "_".join(str(b) for b in band_indices)
    output_subdir = os.path.join(f"{acousticimagingalgo}-{num_channels}ch-task{task}-{methodtype}-bands_{band_str}", basename)
    os.makedirs(output_subdir, exist_ok=True)
    pred_csv_path = os.path.join(output_subdir, f"predictions_{methodtype.lower()}.csv")

    df_gt = pd.read_csv(metadata_path, header=None)
    df_gt.columns = ['frame_index', 'col1', 'col2', 'azimuth', 'elevation']
    powermax_records = []

    for t in tqdm(range(max_frames), desc="Processing frames"):
        row_gt = df_gt[df_gt['frame_index'] == t+1]
        gt_az, gt_el = (row_gt.iloc[0]['azimuth'], row_gt.iloc[0]['elevation']) if not row_gt.empty else (None, None)

        all_points = []
        all_power_maps = []

        for band_idx in band_indices:
            f_low, f_high = freq_bands[band_idx]

            # Compute power for this frequency band
            if acousticimagingalgo == "DAMAS":
                power = DAMAS_frequency_band(stfts, delays, freqs, t,
                                            freq_range=(f_low, f_high),
                                            weighting='triangular')
            elif acousticimagingalgo == "DAS":
                power = DAS_frequency_band(stfts, delays, freqs, t,
                                        freq_range=(f_low, f_high),
                                        weighting='triangular')
            elif acousticimagingalgo == "MUSIC":
                power = MUSIC_frequency_band(stfts, delays, freqs, t,
                                            freq_range=(f_low, f_high),
                                            weighting='triangular', 
                                            num_sources=1)
            else:
                raise ValueError(f"Unsupported algorithm: {acousticimagingalgo}")

            power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))

            mean_power = np.mean(power_db)
            std_power = np.std(power_db)
            if std_power > 0:
                power_db = (power_db - mean_power) / std_power
            else:
                power_db = power_db - mean_power  

            # ✅ Assert expected shape
            if power_db.shape != (npoints,):
                print(f"⚠️ Skipping frame {t}: unexpected shape {power_db.shape}")
                continue

            # ✅ Check for weird values (NaNs, infs, unreasonable negatives)
            if not np.all(np.isfinite(power_db)):
                print(f"⚠️ Skipping frame {t}: power contains NaN or Inf")
                continue

            if methodtype.upper() == "BEST_COMBO":
                threshold = np.percentile(power_db, 95)
                idx_top = np.where(power_db >= threshold)[0]
                for idx in idx_top:
                    all_points.append([azimuths[idx], elevations[idx], power_db[idx]])

            elif methodtype.upper() == "AVERAGE_COMBO":
                all_power_maps.append(power_db)

        if methodtype.upper() == "BEST_COMBO":
            if not all_points:
                pred_az, pred_el = 0.0, 0.0
            else:
                # all_points: list of [az, el, power] collected earlier
                all_points = np.array(all_points)
                idx_best = np.argmax(all_points[:, 2])  # highest power
                pred_az, pred_el = all_points[idx_best, 0], all_points[idx_best, 1]
                pred_az = ((pred_az + 180) % 360) - 180

        elif methodtype.upper() == "AVERAGE_COMBO":
            avg_power = np.mean(np.stack(all_power_maps, axis=0), axis=0)
            idx_max = np.argmax(avg_power)
            pred_az = ((azimuths[idx_max] + 180) % 360) - 180
            pred_el = elevations[idx_max]

        powermax_records.append([t+1, pred_az, pred_el])

    # write predictions
    with open(pred_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "azimuth", "elevation"])
        writer.writerows(powermax_records)

    pred_with_gt_path = pred_csv_path.replace(".csv", "_with_gt.csv")
    add_gt_to_predictions(pred_csv_path, metadata_path, pred_with_gt_path)

    df = pd.read_csv(pred_with_gt_path)
    df['angular_error'] = angular_error(df['azimuth'], df['elevation'], df['gt_azimuth'], df['gt_elevation'])

    df.to_csv(pred_with_gt_path, index=False)
    print(f"✅ Saved predictions with angular error: {pred_with_gt_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help="Task number (e.g., 1, 2, 3)")
    parser.add_argument("--methodtype", type=str, required=True, help="Beam combination method")
    parser.add_argument("--acousticimagingalgo", type=str, default="DAMAS", 
                    choices=["DAMAS", "DAS", "MUSIC"], 
                    help="Acoustic imaging algorithm to use")
    parser.add_argument("--num_channels", type=int, default=32, 
                        choices=[4, 32], 
                        help="Number of microphones to use (4 or 32)")
    args = parser.parse_args()

    task = args.task
    methodtype = args.methodtype

    band_indices = [2,3,4]

    file_paths = []
    metadata_paths = []

    # Search in dev and eval
    for split in ["dev", "eval"]:
        audio_root = f"LOCATA/{split}/task{task}"
        meta_root  = f"LOCATA/metadata_{split}"

        # find all audio files recursively
        audio_files = glob.glob(os.path.join(audio_root, "**", "audio_array_eigenmike.wav"), recursive=True)

        for audio_file in audio_files:
            # extract recording number
            parts = audio_file.split(os.sep)
            recording_part = [p for p in parts if p.startswith("recording")]
            if not recording_part:
                print(f"Warning: Could not determine recording number for {audio_file}")
                continue
            recording = recording_part[0].replace("recording", "")

            meta_filename = f"task{task}_recording{recording}.csv"
            meta_path = os.path.join(meta_root, meta_filename)

            if not os.path.isfile(meta_path):
                print(f"Warning: Metadata file not found for {audio_file}, expected {meta_path}")
                continue

            file_paths.append(audio_file)
            metadata_paths.append(meta_path)

    # Sanity check
    if not file_paths:
        print(f"No files found for task {task}")
        exit(1)

    # Run the main function over all files & bands
    for file_path, metadata_path in zip(file_paths, metadata_paths):
            main(file_path, metadata_path, band_indices, methodtype, task, args.acousticimagingalgo, args.num_channels)