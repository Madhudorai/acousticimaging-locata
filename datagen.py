#SYNTHETIC DATA
import numpy as np
import h5py
import os
import csv
import soundfile as sf
from scipy.signal import fftconvolve
import glob

def cart2sph(xyz):
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z / r))
    return r, azimuth, elevation

def save_metadata_csv(csv_path, source_id, r, azimuth, elevation, duration_sec, frame_rate=10):
    n_frames = int(duration_sec * frame_rate)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'source_id', 'azimuth', 'elevation', 'distance'])
        for frame in range(1, n_frames+1):
            writer.writerow([
                int(frame),
                int(source_id),
                f"{azimuth:.6f}",
                f"{elevation:.6f}",
                f"{r:.6f}"
            ])

def generate_wavs_for_room(h5_path, out_wav_dir, out_csv_dir, duration_sec, sr=48000):
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    roomname = os.path.splitext(os.path.basename(h5_path))[0]

    with h5py.File(h5_path, 'r') as f:
        irs = f['irs'][:]                # shape: (32, 20, rir_len)
        source_pos = f['source_position'][:]  # shape: (20, 3)
        mic_center = f['mic_center_position'][:]

    n_mics, n_sources, rir_len = irs.shape
    n_samples = int(duration_sec * sr) 

    for src_id in range(n_sources):
        signal = np.random.randn(n_samples)*0.01

        # convolve and get actual audio
        audio = np.zeros((n_mics, n_samples + rir_len - 1))
        for mic in range(n_mics):
            rir = irs[mic, src_id]
            audio[mic] = fftconvolve(signal, rir, mode='full')

        # normalize
        audio /= np.max(np.abs(audio))

        # actual duration after convolution
        actual_duration_sec = audio.shape[1] / sr

        wav_path = os.path.join(out_wav_dir, f"{roomname}_source{src_id}.wav")
        sf.write(wav_path, audio.T, sr)

        r, azimuth, elevation = cart2sph(source_pos[src_id] - mic_center)


        csv_path = os.path.join(out_csv_dir, f"{roomname}_source{src_id}.csv")
        save_metadata_csv(
            csv_path, src_id, r, azimuth, elevation,
            duration_sec=actual_duration_sec, frame_rate=10
        )

        print(f"✅ {roomname} | Saved: {wav_path} & {csv_path} ({actual_duration_sec:.2f} sec)")

if __name__ == "__main__":
    # Input and output root
    h5_dir = "dcase_irs_"
    out_root = "simulated_data"

    # Find all .h5 files
    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))

    for h5_path in h5_files:
        roomname = os.path.splitext(os.path.basename(h5_path))[0]

        out_wav_dir = os.path.join(out_root, f"{roomname}_wavs")
        out_csv_dir = os.path.join(out_root, f"{roomname}_metadata")

        generate_wavs_for_room(
            h5_path=h5_path,
            out_wav_dir=out_wav_dir,
            out_csv_dir=out_csv_dir,
            duration_sec=2,
            sr=44100
        )

    print("🎉 All rooms processed!")
