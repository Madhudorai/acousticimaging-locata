import os 
from tqdm import tqdm
import csv 
import numpy as np 
import glob 
import pandas as pd 
import argparse
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from tools import load_and_process_audio, setup_frequency_bands, compute_dynamic_ranges 
from plot_tools import setup_output_directories, create_frame_visualization
from microphone import setup_microphone_array
from beamform import setup_beamforming_directions,DAS_frequency_band, MUSIC_frequency_band,DAMAS_frequency_band,compute_time_delays, interpolate_to_grid
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

def main(file_path, metadata_path, N_MEL_BANDS, band_idx, task, acousticimagingalgo, num_channels):
    # Constants
    c = 343.0  
    fs = 48000  
    npoints = 1024

    if num_channels == 4:
        selected_mics = [5, 9, 25, 21]
    else:
        selected_mics = None

    # Setup microphone array
    positions = setup_microphone_array(selected_mics)
    
    # Setup beamforming directions
    unit_vectors, azimuths, elevations, num_dirs = setup_beamforming_directions(n_points=npoints)
    
    # Compute time delays
    delays = compute_time_delays(positions, unit_vectors, c)
    
    # Load and process audio
    stfts, freqs, num_frames, nperseg, noverlap = load_and_process_audio(file_path, selected_channels=selected_mics)
    
    # Setup frequency bands
    freq_bands, band_names = setup_frequency_bands(fs, 4800, N_MEL_BANDS)
    
    # Compute dynamic ranges
    band_dynamic_ranges = compute_dynamic_ranges(stfts, delays, freqs, freq_bands, num_frames)
    
    # Process frames for each frequency band independently
    max_frames = num_frames  

    parts = file_path.split(os.sep)
    split_name = parts[1]  # 'dev' or 'eval', assuming LOCATA is at parts[0]
    recording_name = [p for p in parts if p.startswith("recording")][0]
    basename = f"{split_name}_{recording_name}"

    band_folder = f"{N_MEL_BANDS}_{band_idx}"
    output_subdir = os.path.join(f"{acousticimagingalgo}-{num_channels}ch-task{task}", basename, band_folder)
    output_dir, band_dirs = setup_output_directories(output_subdir, band_names)

    df_gt = pd.read_csv(metadata_path, header=None)
    df_gt.columns = ['frame_index', 'col1', 'col2', 'azimuth', 'elevation']

    powermax_records = []
    for t in tqdm(range(max_frames), desc="Processing frames"):
        vmin, vmax = band_dynamic_ranges[band_idx]
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

        #  Assert expected shape
        if power_db.shape != (npoints,):
            print(f"⚠️ Skipping frame {t}: unexpected shape {power_db.shape}")
            continue

        # Check for weird values (NaNs, infs, unreasonable negatives)
        if not np.all(np.isfinite(power_db)):
            print(f"⚠️ Skipping frame {t}: power contains NaN or Inf")
            continue

        pred_az_list, pred_el_list = [], []
        # Take the top 10% of points
        n_points = len(power_db)
        top_k = max(3, int(0.1 * n_points))
        idx_sorted = np.argsort(power_db)[-top_k:]  
        idx_top = idx_sorted[np.argsort(-power_db[idx_sorted])] 

        # extract points
        points = np.array([
            [azimuths[idx], elevations[idx], power_db[idx]] for idx in idx_top
        ])

        print(f"\n=== Frame {t+1} ===")
        print("Top points (azimuth, elevation):")
        for p in points:
            print(f"  ({p[0]:.2f}, {p[1]:.2f})")

        kmeans = KMeans(n_clusters=3, random_state=0).fit(points[:, :2])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        print("\nKMeans centroids (azimuth, elevation):")
        for i, c in enumerate(centroids):
            print(f"  Centroid {i+1}: ({c[0]:.2f}, {c[1]:.2f})")

        merged_centroids = merge_clusters(centroids)

        print("\nMerged centroids (azimuth, elevation):")
        for i, c in enumerate(merged_centroids):
            print(f"  Merged centroid {i+1}: ({c[0]:.2f}, {c[1]:.2f})")

        for c in merged_centroids:
            azi_best = c[0]  
            ele_best = c[1]
            pred_az_list.append(azi_best)
            pred_el_list.append(ele_best)
            powermax_records.append([t+1, N_MEL_BANDS, band_idx, azi_best, ele_best])

        # collect GTs for this frame
        df_gt_frame = df_gt[df_gt['frame_index'] == t+1]
        gt_az_list = df_gt_frame['azimuth'].to_list()
        gt_el_list = df_gt_frame['elevation'].to_list()
        # interpolate power to a grid
        power_grid, grid_az_mesh, grid_el_mesh = interpolate_to_grid(
            power_db, azimuths, elevations)

        create_frame_visualization(power_grid, power_db, f_low, f_high, vmin, vmax, stfts,freqs, unit_vectors,
                                        band_idx, band_names,
                                        t, max_frames, nperseg, noverlap, fs,pred_az=pred_az_list, pred_el=pred_el_list, gt_az= gt_az_list, gt_el = gt_el_list)
     
             
    #Initial predictions
    pred_csv_path = os.path.join(output_dir, "predictions.csv")
    with open(pred_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "n_melbands", "band_idx","azimuth", "elevation"])
        writer.writerows(powermax_records)
    
    pred_with_gt_path = pred_csv_path.replace(".csv", "_with_gt.csv")
    add_gt_with_hungarian(
    pred_csv_path,
    metadata_path,
    pred_with_gt_path)

    df = pd.read_csv(pred_with_gt_path)

    df['angular_error'] = angular_error(
        df['azimuth'], df['elevation'],
        df['gt_azimuth'], df['gt_elevation']
    )

    # Save back
    df.to_csv(pred_with_gt_path, index=False)
    print(f"✅ Saved predictions with angular error: {pred_with_gt_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help="Task number (e.g., 2, 4,6)")
    parser.add_argument("--acousticimagingalgo", type=str, default="DAMAS", 
                    choices=["DAMAS", "DAS", "MUSIC"], 
                    help="Acoustic imaging algorithm to use")
    parser.add_argument("--num_channels", type=int, default=32, 
                        choices=[4, 32], 
                        help="Number of microphones to use (4 or 32)")

    args = parser.parse_args()

    task = args.task

    band_config_list = [
        (3, 2)
    ]

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
        for N_MEL_BANDS, band_idx in band_config_list:
            print(f"Running for {file_path} with N_MEL_BANDS={N_MEL_BANDS}, band_idx={band_idx}, "
              f"algo={args.acousticimagingalgo}, channels={args.num_channels}")
            main(file_path, metadata_path, N_MEL_BANDS, band_idx, task, 
             args.acousticimagingalgo, args.num_channels)