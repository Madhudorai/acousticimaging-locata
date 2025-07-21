#SYNTHETIC DATA
import os 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv 
import numpy as np 
import glob 
import pandas as pd 
import h5py 
import argparse

from tools import load_and_process_audio, setup_frequency_bands, compute_dynamic_ranges 
from plot_tools import setup_output_directories, create_frame_visualization
from microphone import load_mic_positions
from beamform import setup_beamforming_directions,DAS_frequency_band,MUSIC_frequency_band, DAMAS_frequency_band, compute_time_delays, interpolate_to_grid
from evaluate import add_gt_to_predictions, angular_error,evaluate_group,bin_and_evaluate

def main(file_path, metadata_path, N_MEL_BANDS, room_name, acousticimagingalgo, num_channels):
    # Constants
    c = 343.0  
    fs = 48000 
    npoints = 1024 

    if num_channels == 4:
        selected_mics = [5, 9, 25, 21]
    else:
        selected_mics = None

    # Setup microphone array
    positions = load_mic_positions(f"dcase_irs_/{room_name}.h5", selected_mics)
    
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
    
    # Setup output directories
    basename_full = os.path.splitext(os.path.basename(file_path))[0]
    basename = basename_full.split('_', 1)[1]
    output_dir, band_dirs = setup_output_directories(f"{acousticimagingalgo}-{num_channels}ch-{room_name}-melbands-{N_MEL_BANDS}/{basename}", band_names)
    
    df_gt = pd.read_csv(metadata_path)
    powermax_records = []
    for t in tqdm(range(max_frames), desc="Processing frames"):
        row_gt = df_gt[(df_gt['frame'] == t+1)]
        if not row_gt.empty:
            gt_az = row_gt.iloc[0]['azimuth']
            gt_el = row_gt.iloc[0]['elevation']
        else:
            gt_az = None
            gt_el = None
        for band_idx in range(N_MEL_BANDS):
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
                                            num_sources=1)
            else:
                raise ValueError(f"Unsupported algorithm: {acousticimagingalgo}")

            
            # Convert to dB with proper floor
            power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))

            idx_max = np.argmax(power_db)
            azi_best = ((azimuths[idx_max] + 180) % 360) - 180  # ensures [-180, 180]
            ele_best = elevations[idx_max]
            powermax_records.append( [t+1, N_MEL_BANDS, band_idx, azi_best, ele_best] )

            # interpolate power to a grid
            power_grid, grid_az_mesh, grid_el_mesh = interpolate_to_grid(
                power_db, azimuths, elevations)

            create_frame_visualization(power_grid, power_db, f_low, f_high, vmin, vmax, stfts,freqs, unit_vectors,
                                            band_idx, band_names,
                                            t, max_frames, nperseg, noverlap, fs,pred_az=azi_best, pred_el=ele_best, gt_az= gt_az, gt_el = gt_el)
        
            
    #Initial predictions
    pred_csv_path = os.path.join(output_dir, "predictions.csv")
    with open(pred_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "n_melbands", "band_idx","azimuth", "elevation"])
        writer.writerows(powermax_records)

    add_gt_to_predictions(
    pred_csv_path,
    metadata_path,
    pred_csv_path.replace(".csv", "_with_gt.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--room', type=str, default='room004', help='Room name, e.g., room004')
    parser.add_argument("--acousticimagingalgo", type=str, default="DAMAS", 
                    choices=["DAMAS", "DAS", "MUSIC"], 
                    help="Acoustic imaging algorithm to use")
    parser.add_argument("--num_channels", type=int, default=32, 
                        choices=[4, 32], 
                        help="Number of microphones to use (4 or 32)")
    args = parser.parse_args()
    room_name = args.room
    acousticimagingalgo = args.acousticimagingalgo
    num_channels = args.num_channels

    wav_dir = f"simulated_data/{room_name}_wavs/"
    metadata_dir = f"simulated_data/{room_name}_metadata/"
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))

    for N_MEL_BANDS in range(3, 13):   # loop over melbands
        for wav_path in wav_files:
            # extract source id from filename
            fname = os.path.basename(wav_path)
            parts = fname.split('_')  
            room, source = parts[0], parts[1].split('.')[0]  

            csv_name = f"{room}_{source}.csv"
            metadata_path = os.path.join(metadata_dir, csv_name)

            if not os.path.exists(metadata_path):
                print(f"⚠️ Warning: metadata not found for {wav_path}")
                continue
            main(wav_path, metadata_path, N_MEL_BANDS, room_name,acousticimagingalgo, num_channels )

        # After all files done for this N_MEL_BANDS → evaluate
        root_dir = f"{acousticimagingalgo}-{num_channels}ch-{room_name}-melbands-{N_MEL_BANDS}"
        csv_paths = glob.glob(os.path.join(root_dir, "**", "*predictions_with_gt.csv"), recursive=True)

        if not csv_paths:
            print(f"❌ No prediction CSVs found for N_MEL_BANDS={N_MEL_BANDS}")
            continue

        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path)
            dfs.append(df)

        all_df = pd.concat(dfs, ignore_index=True)

        # Compute angular error
        all_df['angular_error'] = angular_error(
            all_df['azimuth'], all_df['elevation'],
            all_df['gt_azimuth'], all_df['gt_elevation']
        )

        # Evaluate per band_idx
        for band_idx, df_band in all_df.groupby('band_idx'):
            print(f"\n==========================")
            print(f" Band Index: {band_idx}")
            evaluate_group(df_band)

            bin_edges = [0, 1, 2, 3, 4, 5, 10]  
            bin_and_evaluate(df_band, bin_edges)

        # Save combined results
        out_csv = f"combined_results_nmelbands_{N_MEL_BANDS}.csv"
        all_df.to_csv(out_csv, index=False)
        print(f"\n✅ Combined CSV saved: {out_csv}")