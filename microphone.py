import numpy as np 
import h5py 

def load_mic_positions(h5_path, dataset_name="mic_capsules_position", selected_mics=None):
    """
    Load mic positions directly as Cartesian [x, y, z] in meters from HDF5.
    Optionally select only specific mic indices.
    
    Parameters:
        h5_path: path to .h5 file
        dataset_name: dataset inside the file
        selected_mics: list of indices (0-based) to select (default: all)
    """
    with h5py.File(h5_path, 'r') as f:
        positions = f[dataset_name][:]

    if selected_mics is not None:
        positions = positions[selected_mics]

    return positions

def spherical_to_cartesian(colat, azim, r):
    """
    Convert spherical coordinates to Cartesian
    colat: colatitude in degrees (0° = north pole, 90° = equator, 180° = south pole)
    azim: azimuth in degrees (0° = x-axis, 90° = y-axis)
    r: radius
    """
    colat_rad = np.deg2rad(colat)
    azim_rad = np.deg2rad(azim)
    x = r * np.sin(colat_rad) * np.cos(azim_rad)
    y = r * np.sin(colat_rad) * np.sin(azim_rad)
    z = r * np.cos(colat_rad)
    return x, y, z

def setup_microphone_array(selected_mics=None):
    eigenmike_raw = {
        "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042],
        "4": [90, 328, 0.042], "5": [32, 0, 0.042], "6": [55, 45, 0.042],
        "7": [90, 69, 0.042], "8": [125, 45, 0.042], "9": [148, 0, 0.042],
        "10": [125, 315, 0.042], "11": [90, 291, 0.042], "12": [55, 315, 0.042],
        "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042],
        "16": [159, 89, 0.042], "17": [69, 180, 0.042], "18": [90, 212, 0.042],
        "19": [111, 180, 0.042], "20": [90, 148, 0.042], "21": [32, 180, 0.042],
        "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],
        "25": [148, 180, 0.042], "26": [125, 135, 0.042], "27": [90, 111, 0.042],
        "28": [55, 135, 0.042], "29": [21, 269, 0.042], "30": [58, 270, 0.042],
        "31": [122, 270, 0.042], "32": [159, 271, 0.042]
    }

    if selected_mics is None:
        selected_mics = list(range(1, 33))  # all mics

    positions = []
    for idx in selected_mics:
        colat, azim, r = eigenmike_raw[str(idx)]
        positions.append(spherical_to_cartesian(colat, azim, r))
    return np.array(positions)