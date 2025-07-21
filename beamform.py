import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import eigh
def fibonacci_sphere(n_points=1000):
    """
    Generate points on a sphere using Fibonacci tessellation
    This provides a nearly uniform distribution of points
    """
    golden_ratio = (1 + 5**0.5) / 2  # Golden ratio

    indices = np.arange(0, n_points, dtype=float) + 0.5

    # Fibonacci spiral on sphere
    theta = np.arccos(1 - 2 * indices / n_points)  # Colatitude
    phi = np.pi * (1 + golden_ratio) * indices     # Azimuth

    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    unit_vectors = np.column_stack((x, y, z))

    # Convert back to spherical for reference
    elevations = 90 - np.rad2deg(theta)  # Convert colatitude to elevation
    azimuths = np.rad2deg(phi) % 360     # Ensure azimuth is in [0, 360)

    return unit_vectors, azimuths, elevations


def DAS_frequency_band(stfts, delays, freqs, frame_idx, freq_range,
                           weighting='triangular', center_freq=None):
    """
    Compute beamformed power for all directions at a specific time frame
    for a specific frequency band

    Parameters:
    -----------
    stfts : ndarray
        STFT data [mics, freqs, frames]
    delays : ndarray
        Time delays for steering vectors [directions, mics]
    freqs : ndarray
        Frequency bins
    frame_idx : int
        Time frame index
    freq_range : tuple
        Frequency range (min_freq, max_freq) in Hz
    weighting : str
        'uniform' for uniform weighting, 'triangular' for triangular weighting
    center_freq : float or None
        Center frequency for triangular weighting. If None, uses middle of freq_range

    Returns:
    --------
    power : ndarray
        Beamformed power for each direction
    """
    power = np.zeros(delays.shape[0])

    # Find frequency range indices
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    active_freqs = np.where(freq_mask)[0]

    if len(active_freqs) == 0:
        return power

    # Set center frequency for triangular weighting
    if center_freq is None:
        center_freq = (freq_range[0] + freq_range[1]) / 2.0

    for bin_idx in active_freqs:
        f = freqs[bin_idx]

        # Calculate steering vector for this frequency
        phase_shifts = 2 * np.pi * f * delays
        steering_vectors = np.exp(-1j * phase_shifts)

        # Get STFT values for this bin and frame
        X = stfts[:, bin_idx, frame_idx]

        # Compute beamformed output with proper normalization
        beamformed = np.abs(np.sum(steering_vectors * X[np.newaxis, :], axis=1)) ** 2

        # Apply frequency weighting
        if weighting == 'uniform':
            weight = 1.0
        elif weighting == 'triangular':
            # Triangular weighting centered at center_freq
            if f <= center_freq:
                # Rising edge: from freq_range[0] to center_freq
                weight = (f - freq_range[0]) / (center_freq - freq_range[0]) if center_freq > freq_range[0] else 1.0
            else:
                # Falling edge: from center_freq to freq_range[1]
                weight = (freq_range[1] - f) / (freq_range[1] - center_freq) if freq_range[1] > center_freq else 1.0
            # Ensure weight doesn't go below 0
            weight = max(0.0, weight)
        else:
            raise ValueError(f"Unknown weighting type: {weighting}")

        power += beamformed * weight

    return power

def MUSIC_frequency_band(stfts, delays, freqs, frame_idx, freq_range,
                        weighting='triangular', center_freq=None, forward_backward=True,
                        num_sources=1, positions=None):
    """
    Compute MUSIC spectrum for all directions at a specific time frame
    for a specific frequency band
    
    Parameters:
    -----------
    stfts : ndarray
        STFT data [mics, freqs, frames]
    delays : ndarray
        Time delays for steering vectors [directions, mics]
    freqs : ndarray
        Frequency bins
    frame_idx : int
        Time frame index
    freq_range : tuple
        Frequency range (min_freq, max_freq) in Hz
    weighting : str
        'uniform' for uniform weighting, 'triangular' for triangular weighting
    center_freq : float or None
        Center frequency for triangular weighting. If None, uses middle of freq_range
    forward_backward : bool
        Whether to use forward-backward averaging
    num_sources : int
        Number of sources to estimate
    positions : ndarray
        Microphone positions [mics, 3] - only needed if delays is None
        
    Returns:
    --------
    power : ndarray
        MUSIC spectrum for each direction
    """
    c = 343.0  
    # Initialize power array for all directions
    num_dirs = delays.shape[0]
    num_mics = stfts.shape[0]
    music_spectrum = np.zeros(num_dirs)
    
    # Find frequency range indices
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    active_freqs = np.where(freq_mask)[0]
    
    if len(active_freqs) == 0:
        return music_spectrum
    
    # Set center frequency for triangular weighting
    if center_freq is None:
        center_freq = (freq_range[0] + freq_range[1]) / 2.0
    
    # Process each frequency bin and accumulate weighted results
    total_weight = 0.0
    
    for bin_idx in active_freqs:
        f = freqs[bin_idx]
        
        # Get STFT data for this frequency bin and time frame
        X = stfts[:, bin_idx, frame_idx]
        
        # Check signal strength
        signal_power = np.var(X)
        if signal_power < 1e-15:  # Very weak signal
            continue
        
        # Form covariance matrix
        if forward_backward:
            # Forward-backward averaging
            X_fb = np.conj(X[::-1])
            R_forward = np.outer(X, X.conj())
            R_backward = np.outer(X_fb, X_fb.conj())
            R_snapshot = (R_forward + R_backward) / 2
        else:
            R_snapshot = np.outer(X, X.conj())
        
        # Diagonal loading for numerical stability
        R_snapshot += np.eye(num_mics) * 1e-6 * np.trace(R_snapshot) / num_mics
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = eigh(R_snapshot)
        except np.linalg.LinAlgError:
            print(f"Warning: Eigendecomposition failed for frequency {f:.1f} Hz")
            continue
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Check eigenvalue separation (signal detection)
        if len(eigenvalues) > num_sources:
            signal_noise_ratio = eigenvalues[num_sources-1] / eigenvalues[num_sources]
            if signal_noise_ratio < 1.5:  # Poor signal-to-noise separation
                continue
        
        # Noise subspace
        noise_subspace = eigenvectors[:, num_sources:]
        
        # Apply frequency weighting
        if weighting == 'uniform':
            weight = 1.0
        elif weighting == 'triangular':
            # Triangular weighting centered at center_freq
            if f <= center_freq:
                # Rising edge: from freq_range[0] to center_freq
                weight = (f - freq_range[0]) / (center_freq - freq_range[0]) if center_freq > freq_range[0] else 1.0
            else:
                # Falling edge: from center_freq to freq_range[1]
                weight = (freq_range[1] - f) / (freq_range[1] - center_freq) if freq_range[1] > center_freq else 1.0
            # Ensure weight doesn't go below 0
            weight = max(0.0, weight)
        else:
            raise ValueError(f"Unknown weighting type: {weighting}")
        
        # Weight by signal power (this is key!)
        freq_weight = weight * signal_power
        total_weight += freq_weight
        
        # Calculate wavenumber
        k = 2 * np.pi * f / c
        
        # Calculate MUSIC pseudo-spectrum for each direction
        for dir_idx in range(num_dirs):
            phase_shifts = 2 * np.pi * f * delays[dir_idx, :]
            steering_vector = np.exp(1j * phase_shifts)
            
            # Normalize steering vector
            steering_vector = steering_vector / np.linalg.norm(steering_vector)
            
            # MUSIC pseudo-spectrum
            projection = noise_subspace.conj().T @ steering_vector
            denominator = np.real(projection.conj().T @ projection)
            
            # Accumulate weighted MUSIC spectrum
            music_spectrum[dir_idx] += freq_weight / (denominator + 1e-15)
    
    # Normalize by total weight
    if total_weight > 0:
        music_spectrum = music_spectrum / total_weight
    
    return music_spectrum

def DAMAS_frequency_band(stfts, delays, freqs, frame_idx, freq_range,
                        weighting='triangular', center_freq=None,
                        max_iterations=100, tolerance=1e-6,
                        relaxation_factor=0.5, diagonal_removal=True):
    """
    Compute DAMAS deconvolved acoustic source map for all directions at a specific time frame
    for a specific frequency band using iterative deconvolution

    Parameters:
    -----------
    stfts : ndarray
        STFT data [mics, freqs, frames]
    delays : ndarray
        Time delays for steering vectors [directions, mics]
    freqs : ndarray
        Frequency bins
    frame_idx : int
        Time frame index
    freq_range : tuple
        Frequency range (min_freq, max_freq) in Hz
    weighting : str
        'uniform' for uniform weighting, 'triangular' for triangular weighting
    center_freq : float or None
        Center frequency for triangular weighting. If None, uses middle of freq_range
    max_iterations : int
        Maximum number of DAMAS iterations
    tolerance : float
        Convergence tolerance for DAMAS iteration
    relaxation_factor : float
        Relaxation factor for Gauss-Seidel iteration (0 < factor <= 1)
    diagonal_removal : bool
        Whether to remove diagonal elements from PSF matrix

    Returns:
    --------
    source_map : ndarray
        DAMAS deconvolved source strength for each direction
    beamformed_map : ndarray
        Original beamformed map (for comparison)
    """
    # Initialize arrays
    num_dirs = delays.shape[0]
    num_mics = stfts.shape[0]

    # Find frequency range indices
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    active_freqs = np.where(freq_mask)[0]

    if len(active_freqs) == 0:
        return np.zeros(num_dirs), np.zeros(num_dirs)

    # Set center frequency for triangular weighting
    if center_freq is None:
        center_freq = (freq_range[0] + freq_range[1]) / 2.0

    # Compute beamformed power map (dirty map)
    beamformed_map = np.zeros(num_dirs)
    total_weight = 0.0

    # Build Point Spread Function (PSF) matrix
    psf_matrix = np.zeros((num_dirs, num_dirs))

    for bin_idx in active_freqs:
        f = freqs[bin_idx]

        # Get STFT data for this frequency bin and time frame
        X = stfts[:, bin_idx, frame_idx]

        # Check signal strength
        signal_power = np.var(X)
        if signal_power < 1e-15:  # Very weak signal
            continue

        # Calculate steering vectors for this frequency
        phase_shifts = 2 * np.pi * f * delays
        steering_vectors = np.exp(-1j * phase_shifts)  # [directions, mics]

        # Normalize steering vectors
        steering_vectors = steering_vectors / np.linalg.norm(steering_vectors, axis=1, keepdims=True)

        # Compute beamformed output
        beamformed = np.abs(np.sum(steering_vectors * X[np.newaxis, :], axis=1)) ** 2

        # Apply frequency weighting
        if weighting == 'uniform':
            weight = 1.0
        elif weighting == 'triangular':
            # Triangular weighting centered at center_freq
            if f <= center_freq:
                weight = (f - freq_range[0]) / (center_freq - freq_range[0]) if center_freq > freq_range[0] else 1.0
            else:
                weight = (freq_range[1] - f) / (freq_range[1] - center_freq) if freq_range[1] > center_freq else 1.0
            weight = max(0.0, weight)
        else:
            raise ValueError(f"Unknown weighting type: {weighting}")

        # Weight by signal power and frequency weight
        freq_weight = weight * signal_power
        total_weight += freq_weight

        # Accumulate beamformed map
        beamformed_map += beamformed * freq_weight

        # Build PSF matrix for this frequency
        # PSF[i,j] represents the response at location i due to a source at location j
        for i in range(num_dirs):
            for j in range(num_dirs):
                # Cross-correlation between steering vectors
                cross_corr = np.abs(np.vdot(steering_vectors[i], steering_vectors[j])) ** 2
                psf_matrix[i, j] += cross_corr * freq_weight

    # Normalize by total weight
    if total_weight > 0:
        beamformed_map = beamformed_map / total_weight
        psf_matrix = psf_matrix / total_weight
    else:
        return np.zeros(num_dirs), np.zeros(num_dirs)

    # Remove diagonal elements from PSF matrix (optional)
    if diagonal_removal:
        np.fill_diagonal(psf_matrix, 0)

    # DAMAS deconvolution using Gauss-Seidel iteration
    source_map = damas_gauss_seidel(beamformed_map, psf_matrix,
                                   max_iterations, tolerance, relaxation_factor)

    return source_map#, beamformed_map


def damas_gauss_seidel(beamformed_map, psf_matrix, max_iterations=100,
                      tolerance=1e-6, relaxation_factor=0.5):
    """
    Solve DAMAS deconvolution using Gauss-Seidel iteration

    Solves: psf_matrix @ source_map = beamformed_map

    Parameters:
    -----------
    beamformed_map : ndarray
        Measured beamformed power map
    psf_matrix : ndarray
        Point Spread Function matrix
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    relaxation_factor : float
        Relaxation factor (0 < factor <= 1)

    Returns:
    --------
    source_map : ndarray
        Deconvolved source strength map
    """
    n = len(beamformed_map)
    source_map = np.copy(beamformed_map)  # Initialize with beamformed map
    source_map_old = np.copy(source_map)

    # Extract diagonal elements
    diagonal = np.diag(psf_matrix)

    for iteration in range(max_iterations):
        for i in range(n):
            if diagonal[i] == 0:
                continue

            # Compute residual
            residual = beamformed_map[i] - np.dot(psf_matrix[i], source_map)

            # Gauss-Seidel update with relaxation
            correction = residual / diagonal[i]
            source_map[i] += relaxation_factor * correction

            # Enforce non-negativity constraint
            source_map[i] = max(0, source_map[i])

        # Check for convergence
        if iteration > 0:
            relative_change = np.linalg.norm(source_map - source_map_old) / np.linalg.norm(source_map_old)
            if relative_change < tolerance:
                print(f"DAMAS converged after {iteration + 1} iterations")
                break

        source_map_old = np.copy(source_map)

    return source_map

def interpolate_to_grid(power_values, azimuths, elevations, grid_az_res=2, grid_el_res=2):
    """
    Interpolate scattered Fibonacci points to a regular grid for visualization
    """
    # Create regular grid
    grid_az = np.arange(0, 360, grid_az_res)
    grid_el = np.arange(-90, 90 + grid_el_res, grid_el_res)
    grid_az_mesh, grid_el_mesh = np.meshgrid(grid_az, grid_el)

    # Points for interpolation
    points = np.column_stack((azimuths, elevations))
    grid_points = np.column_stack((grid_az_mesh.flatten(), grid_el_mesh.flatten()))

    # Handle azimuth wraparound by creating duplicates
    az_wrapped = azimuths.copy()
    el_wrapped = elevations.copy()
    power_wrapped = power_values.copy()

    # Add points near boundaries to handle wraparound
    mask_near_0 = azimuths < 20
    mask_near_360 = azimuths > 340

    if np.any(mask_near_0):
        az_wrapped = np.concatenate([az_wrapped, azimuths[mask_near_0] + 360])
        el_wrapped = np.concatenate([el_wrapped, elevations[mask_near_0]])
        power_wrapped = np.concatenate([power_wrapped, power_values[mask_near_0]])

    if np.any(mask_near_360):
        az_wrapped = np.concatenate([az_wrapped, azimuths[mask_near_360] - 360])
        el_wrapped = np.concatenate([el_wrapped, elevations[mask_near_360]])
        power_wrapped = np.concatenate([power_wrapped, power_values[mask_near_360]])

    points_wrapped = np.column_stack((az_wrapped, el_wrapped))

    # Interpolate using griddata
    try:
        grid_values = griddata(points_wrapped, power_wrapped, grid_points,
                              method='cubic', fill_value=np.nan)
        grid_values = grid_values.reshape(grid_az_mesh.shape)

        # Fill any remaining NaN values with nearest neighbor
        if np.any(np.isnan(grid_values)):
            grid_values_nn = griddata(points_wrapped, power_wrapped, grid_points,
                                    method='nearest')
            grid_values_nn = grid_values_nn.reshape(grid_az_mesh.shape)
            mask = np.isnan(grid_values)
            grid_values[mask] = grid_values_nn[mask]

    except Exception as e:
        print(f"Cubic interpolation failed: {e}, falling back to linear")
        grid_values = griddata(points_wrapped, power_wrapped, grid_points,
                              method='linear', fill_value=np.nan)
        grid_values = grid_values.reshape(grid_az_mesh.shape)

        # Fill NaN with nearest neighbor
        if np.any(np.isnan(grid_values)):
            grid_values_nn = griddata(points_wrapped, power_wrapped, grid_points,
                                    method='nearest')
            grid_values_nn = grid_values_nn.reshape(grid_az_mesh.shape)
            mask = np.isnan(grid_values)
            grid_values[mask] = grid_values_nn[mask]

    return grid_values, grid_az_mesh, grid_el_mesh

def setup_beamforming_directions(n_points=1024):
    """Setup beamforming directions using Fibonacci tessellation"""
    unit_vectors, azimuths, elevations = fibonacci_sphere(n_points=n_points)
    num_dirs = len(unit_vectors)
    #print(f"Created {num_dirs} directions using Fibonacci tessellation")
    return unit_vectors, azimuths, elevations, num_dirs

def compute_time_delays(positions, unit_vectors, c=343.0):
    """Precompute time delays for all directions and microphones"""
    num_dirs = len(unit_vectors)
    #print(f"Precomputing delays for {num_dirs} directions...")
    delays = np.zeros((num_dirs, positions.shape[0]))
    for d in range(num_dirs):
        for m in range(positions.shape[0]):
            # Time delay = (r_m · d) / c where r_m is mic position and d is direction
            delays[d, m] = np.dot(positions[m], unit_vectors[d]) / c

    #print(f"Delay range: {delays.min():.6f} to {delays.max():.6f} seconds")
    return delays
