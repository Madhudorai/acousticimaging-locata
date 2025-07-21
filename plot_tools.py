import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from sklearn.cluster import KMeans
import os 
import shutil 

def setup_output_directories(output_dir,band_names):
    """Create output directories and video writers"""
    if os.path.exists(output_dir):
       shutil.rmtree(output_dir)
    
    #  Recreate a clean directory
    os.makedirs(output_dir, exist_ok=True)

    band_dirs = []
    #video_writers = []

    for band_name in band_names:
        band_dir = os.path.join(output_dir, band_name)
        frames_dir = os.path.join(band_dir, "frames")
        
        os.makedirs(frames_dir, exist_ok=True)

        band_dirs.append({
            "root": band_dir,
            "frames": frames_dir
            })

        #video_path = os.path.join(band_dir, f'beamforming_360_{band_name}.mp4')
        #writer = imageio.get_writer(video_path, fps=10, quality=8)
        #video_writers.append(writer)

    return output_dir, band_dirs

def create_2d_panoramic_plot(
    ax, power_grid, vmin, vmax, band_name, t, max_frames, nperseg, noverlap, fs,
    pred_az=None, pred_el=None, gt_az=None, gt_el=None):
    """Create 2D equirectangular projection plot"""
    extent = [0, 360, -90, 90]
    im = ax.imshow(power_grid,
                   extent=extent,
                   origin='lower',
                   aspect='auto',
                   cmap='hot',
                   vmin=vmin,
                   vmax=vmax,
                   interpolation='bilinear')

    # Styling for 2D plot
    ax.set_facecolor('black')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Beamformed Power (dB)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Add grid
    ax.grid(True, alpha=0.3, color='white', linestyle='--', linewidth=0.5)

    # Add direction markers
    directions = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
    for label, az in directions.items():
        ax.axvline(x=az, color='cyan', alpha=0.7, linestyle=':', linewidth=1)
        ax.text(az, 95, label, ha='center', va='center',
                color='cyan', fontsize=14, weight='bold')

    # Add elevation markers
    for el in [-60, -30, 0, 30, 60]:
        ax.axhline(y=el, color='white', alpha=0.2, linestyle=':')
        if el == 0:
            ax.text(365, el, 'Horizon', ha='left', va='center',
                    color='white', fontsize=10, weight='bold')
            
    # Helper to wrap azimuth to [0, 360]
    def wrap_az(az):
        return (az + 360) % 360

    # Convert single values to lists if needed
    if pred_az is not None and not np.iterable(pred_az):
        pred_az = [pred_az]
        pred_el = [pred_el]

    if gt_az is not None and not np.iterable(gt_az):
        gt_az = [gt_az]
        gt_el = [gt_el]

    # Plot predictions
    if pred_az is not None:
        for az, el in zip(pred_az, pred_el):
            ax.plot(wrap_az(az), el, 'wo', markersize=8, label='Prediction', markeredgecolor='black')

    # Plot ground truths
    if gt_az is not None:
        for az, el in zip(gt_az, gt_el):
            ax.plot(wrap_az(az), el, 'rx', markersize=10, label='Ground Truth', markeredgewidth=2)   
    
    # Labels and title for 2D plot
    ax.legend(loc='lower center', fontsize=10, facecolor='black', framealpha=0.5, labelcolor='white')
    ax.set_xlabel('Azimuth (degrees)', color='white', fontsize=12)
    ax.set_ylabel('Elevation (degrees)', color='white', fontsize=12)
    ax.set_title(f'{band_name} | Time: {t*(nperseg-noverlap)/fs:.2f}s | Frame {t+1}/{max_frames}',
                 color='white', fontsize=14, weight='bold')

    # Set ticks and limits
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(colors='white')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    return im

def create_3d_spherical_plot(ax, unit_vectors, power_db, vmin, vmax):
    """Create 3D spherical scatter plot"""
    ax.set_facecolor('black')

    # Create 3D scatter plot
    scatter = ax.scatter(unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2],
                         c=power_db, s=8, cmap='hot', alpha=0.8, vmin=vmin, vmax=vmax)

    # Add a wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='white', linewidth=0.5)

    # Styling for 3D plot
    ax.set_xlabel('X', color='white', fontsize=10)
    ax.set_ylabel('Y', color='white', fontsize=10)
    ax.set_zlabel('Z', color='white', fontsize=10)
    ax.set_title('3D View', color='white', fontsize=12, weight='bold')

    # Set equal aspect ratio
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    # Style the 3D axes
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    return scatter

def create_frequency_spectrum_plot(ax, stfts, freqs, f_low, f_high, t):
    """Create frequency spectrum plot for the current band"""
    ax.set_facecolor('black')

    # Show the frequency range for this band
    freq_mask = (freqs >= f_low) & (freqs <= f_high)
    band_freqs = freqs[freq_mask]

    if len(band_freqs) > 0:
        # Average spectrum across all mics for this time frame
        spectrum = np.mean(np.abs(stfts[:, freq_mask, t]), axis=0)
        ax.semilogy(band_freqs, spectrum, 'yellow', linewidth=2, label=f'Band Spectrum')

        # Show triangular weighting
        center_freq = (f_low + f_high) / 2.0
        weights = []
        for f in band_freqs:
            if f <= center_freq:
                weight = (f - f_low) / (center_freq - f_low) if center_freq > f_low else 1.0
            else:
                weight = (f_high - f) / (f_high - center_freq) if f_high > center_freq else 1.0
            weights.append(max(0.0, weight))

        # Normalize weights for visualization
        weights = np.array(weights)
        if np.max(spectrum) > 0:
            scaled_weights = weights * np.max(spectrum) * 0.5
            ax.plot(band_freqs, scaled_weights, 'cyan', linewidth=1, alpha=0.7, label='Triangular Weight')

    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=10)
    ax.set_ylabel('Magnitude', color='white', fontsize=10)
    ax.set_title(f'Band Spectrum: {f_low:.0f}-{f_high:.0f} Hz', color='white', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, color='white')
    ax.tick_params(colors='white')

def create_frame_visualization(power_grid, power_db, f_low, f_high, vmin, vmax, stfts,freqs, unit_vectors,
                                        band_idx, band_names,
                                        t, max_frames, nperseg, noverlap, fs,pred_az=None, pred_el=None, gt_az=None, gt_el=None):
    """Create complete frame visualization with all subplots"""
    # visualizations
    fig = plt.figure(figsize=(16, 10), facecolor='black')

    # Main 2D equirectangular projection
    ax1 = plt.subplot(2, 2, (1, 3))
    create_2d_panoramic_plot(ax1, power_grid, vmin, vmax, band_names[band_idx], 
                            t, max_frames, nperseg, noverlap, fs,pred_az, pred_el, gt_az, gt_el)

    # 3D view of Fibonacci points
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    create_3d_spherical_plot(ax2, unit_vectors, power_db, vmin, vmax)

    # Frequency spectrum for this band
    ax3 = plt.subplot(2, 2, 4)
    create_frequency_spectrum_plot(ax3, stfts, freqs, f_low, f_high, t)
