import numpy as np
import soundfile as sf
from scipy.signal import stft
from tqdm import tqdm
from scipy.interpolate import griddata
import librosa
from beamform import beamform_frequency_band

def load_and_process_audio(file_path, fs=48000, selected_channels=None):
    """
    Load and process audio, return STFTs [mics, freqs, frames].
    If selected_channels is given, keep only those channels (0-based).
    """
    import soundfile as sf
    from scipy.signal import stft

    x, fs = sf.read(file_path)  # x.shape: [samples, channels]

    if selected_channels is not None:
        x = x[:, selected_channels]

    # Transpose to [channels, samples]
    x = x.T

    # Compute STFT
    nperseg = 4800
    noverlap = 0 

    stfts = []
    freqs = None
    for ch in x:
        f, _, Zxx = stft(ch, fs, nperseg=nperseg, noverlap=noverlap)
        if freqs is None:
            freqs = f
        stfts.append(Zxx)

    stfts = np.stack(stfts)  # [mics, freqs, frames]
    num_frames = stfts.shape[2]

    return stfts, freqs, num_frames, nperseg, noverlap


def setup_frequency_bands(fs, nfft, N_MEL_BANDS):
    """Setup Mel frequency bands for multi-band analysis"""
    # Generate Mel filterbank and frequency bands
    mel_filters = librosa.filters.mel(sr=fs, n_fft=nfft, n_mels=N_MEL_BANDS,
                                    fmin=100, fmax=8000)

    # Convert mel filter frequencies to actual frequency ranges
    mel_freqs = librosa.mel_frequencies(n_mels=N_MEL_BANDS+2, fmin=100, fmax=8000)

    # Create frequency band ranges from mel frequencies
    freq_bands = []
    band_names = []
    for i in range(N_MEL_BANDS):
        freq_bands.append((mel_freqs[i], mel_freqs[i+2]))  # Overlapping bands
        band_names.append(f"Band_{i+1:02d}_{mel_freqs[i]:.0f}-{mel_freqs[i+2]:.0f}Hz")
    
    return freq_bands, band_names

def compute_dynamic_ranges(stfts, delays, freqs, freq_bands, num_frames):
    """Compute dynamic range for each frequency band"""
    #print("Computing dynamic range for each frequency band...")
    band_dynamic_ranges = []

    for band_idx, (f_low, f_high) in enumerate(freq_bands):

        sample_powers = []
        # Sample a subset of frames to compute dynamic range
        sample_frames = range(0, num_frames, 5)  # Sample every 5th frame, up to 50 frames

        for t in sample_frames:
            power = beamform_frequency_band(stfts, delays, freqs, t,
                                        freq_range=(f_low, f_high),
                                        weighting='triangular')
            # Convert to dB with proper floor
            power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))
            sample_powers.extend(power_db[np.isfinite(power_db)])

        if len(sample_powers) > 0:
            sample_powers = np.array(sample_powers)
            vmin = np.percentile(sample_powers, 1)
            vmax = np.percentile(sample_powers, 99)
            band_dynamic_ranges.append((vmin, vmax))
        else:
            # Fallback range
            band_dynamic_ranges.append((-60, 0))
    
    return band_dynamic_ranges

