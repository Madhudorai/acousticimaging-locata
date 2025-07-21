import os 
from tqdm import tqdm
import csv 
import numpy as np 
import glob 
import pandas as pd 
import argparse
from sklearn.cluster import KMeans

from plot_tools import setup_output_directories
from tools import load_and_process_audio, setup_frequency_bands, compute_dynamic_ranges 
from microphone import setup_microphone_array
from beamform import setup_beamforming_directions,DAS_frequency_band, MUSIC_frequency_band, DAMAS_frequency_band,compute_time_delays
from evaluate import add_gt_with_hungarian, angular_error

def merge_clusters(centroids, merge_within=15):
    """
    Merge centroids that are within `merge_within` degrees of each other.
    Returns a list of merged centroids.
    """
    centroids = list(centroids)  # make mutable
    merged = []

    while centroids:
        c = centroids.pop(0)
        to_merge = [c]
        remaining = []
        for other in centroids:
            if angular_error(c[0], c[1], other[0], other[1]) <= merge_within:
                to_merge.append(other)
            else:
                remaining.append(other)
        merged.append(np.mean(to_merge, axis=0))
        centroids = remaining

    return merged

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

    N_MEL_BANDS = 6
    freq_bands, band_names = setup_frequency_bands(fs, 4800, N_MEL_BANDS)
    band_dynamic_ranges = compute_dynamic_ranges(stfts, delays, freqs, freq_bands, num_frames)

    max_frames = num_frames

    parts = file_path.split(os.sep)
    split_name = parts[1]  # 'dev' or 'eval'
    recording_name = [p for p in parts if p.startswith("recording")][0]
    basename = f"{split_name}_{recording_name}"

    # Prediction output dir
    band_str = "_".join(str(b) for b in band_indices)
    prediction_output_dir = os.path.join(f"{acousticimagingalgo}-{num_channels}ch-task{task}-{methodtype}--bands_{band_str}", basename)
    os.makedirs(prediction_output_dir, exist_ok=True)
    pred_csv_path = os.path.join(prediction_output_dir, f"predictions_{methodtype.lower()}.csv")

    # Prepare band output dirs once
    band_dirs_map = {}
    for band_idx in band_indices:
        band_folder = f"bands/{N_MEL_BANDS}_{band_idx}"
        frame_output_dir = os.path.join(prediction_output_dir, band_folder)
        output_dir, band_dirs = setup_output_directories(frame_output_dir, band_names)
        band_dirs_map[band_idx] = band_dirs[band_idx] 

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
                                            num_sources=3)
            else:
                raise ValueError(f"Unsupported algorithm: {acousticimagingalgo}")

            power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))

            # ✅ Assert expected shape
            if power_db.shape != (npoints,):
                print(f"⚠️ Skipping frame {t}: unexpected shape {power_db.shape}")
                continue

            # ✅ Check for weird values (NaNs, infs, unreasonable negatives)
            if not np.all(np.isfinite(power_db)):
                print(f"⚠️ Skipping frame {t}: power contains NaN or Inf")
                continue
            
            mean_power = np.mean(power_db)
            std_power = np.std(power_db)
            if std_power > 0:
                power_db = (power_db - mean_power) / std_power
            else:
                power_db = power_db - mean_power

            if methodtype.upper() == "BEST_COMBO":
                n_points = len(power_db)
                top_k = max(3, int(0.05 * n_points))
                idx_sorted = np.argsort(power_db)[-top_k:]
                idx_top = idx_sorted[np.argsort(-power_db[idx_sorted])]

                for idx in idx_top:
                    all_points.append([azimuths[idx], elevations[idx], power_db[idx]])

            elif methodtype.upper() == "AVERAGE_COMBO":
                all_power_maps.append(power_db)

        pred_az_list, pred_el_list = [], []

        if methodtype.upper() == "BEST_COMBO":
            if len(all_points) >= 3:
                points = np.array(all_points)
                kmeans = KMeans(n_clusters=min(3, len(points)), random_state=0).fit(points[:, :2])
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
                merged_centroids = merge_clusters(centroids)

                for c in merged_centroids:
                    azi_best = ((c[0] + 180) % 360) - 180
                    ele_best = c[1]
                    pred_az_list.append(azi_best)
                    pred_el_list.append(ele_best)
                    powermax_records.append([t+1,azi_best, ele_best])

        elif methodtype.upper() == "AVERAGE_COMBO":
            if all_power_maps:
                avg_power = np.mean(np.stack(all_power_maps, axis=0), axis=0)
                top_k = 15
                idx_top = np.argsort(avg_power)[-top_k:] 
                if len(idx_top) >= 3:
                    points = np.array([
                        [azimuths[idx], elevations[idx], avg_power[idx]] for idx in idx_top
                    ])
                    kmeans = KMeans(n_clusters=min(3, len(points)), random_state=0).fit(points[:, :2])
                    labels = kmeans.labels_
                    centroids = kmeans.cluster_centers_
                    merged_centroids = merge_clusters(centroids)

                    for c in merged_centroids:
                        azi_best = ((c[0] + 180) % 360) - 180
                        ele_best = c[1]
                        pred_az_list.append(azi_best)
                        pred_el_list.append(ele_best)
                        powermax_records.append([t+1,azi_best, ele_best])

        # collect GTs for this frame
        df_gt_frame = df_gt[df_gt['frame_index'] == t+1]
        gt_az_list = df_gt_frame['azimuth'].to_list()
        gt_el_list = df_gt_frame['elevation'].to_list()

    # Save predictions
    with open(pred_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "azimuth", "elevation"])
        writer.writerows(powermax_records)

    pred_with_gt_path = pred_csv_path.replace(".csv", "_with_gt.csv")
    add_gt_with_hungarian(pred_csv_path, metadata_path, pred_with_gt_path)

    df = pd.read_csv(pred_with_gt_path)

    df['angular_error'] = angular_error(
        df['azimuth'], df['elevation'],
        df['gt_azimuth'], df['gt_elevation']
    )

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