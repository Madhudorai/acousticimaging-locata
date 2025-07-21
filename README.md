
---

# 📡 AcousticImaging-LOCATA

## 🚀 Setup

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🎧 Synthetic Data

### 📂 Data: `dcase_irs_`

* Contains **precomputed multichannel impulse responses (IRs)** in **Ambisonics A-format**.
* Simulated in **5 indoor environments** (from the Gibson dataset) using **SoundSpaces 2.0**.
* Microphone:

  * 32-channel **EigenMike** placed at the center of the room.
* Sources:

  * 20 uniformly placed static sources with:

    * direct line-of-sight to the array,
    * minimum distances from the array and walls.

---

### 🔨 Generate Synthetic Audio

Run:

```bash
python3 datagen.py
```

This will:

* ✅ Convolve each IR with Gaussian white noise (SD = 0.01)
* ✅ Save multichannel `.wav` files in `simulated_data/`
* ✅ Save metadata:

  ```csv
  frame, source_id, azimuth, elevation, distance
  ```

---

## 📈 Acoustic Imaging

Supported methods: **DAS**, **MUSIC**, **DAMAS**

Run:

```bash
python3 room.py --room room001 --acousticimagingalgo DAS --num_channels 32
```

### Options:

* `--room` — One of the **5 rooms**:
  `room001`, `room002`, …, `room005`
* `--acousticimagingalgo` — Choose algorithm:
  `DAS`, `MUSIC`, or `DAMAS`
* `--num_channels` — Choose number of microphones:
  `4` or `32`

  * (For `4`, the microphones `[6, 10, 26, 22]` form a near-tetrahedral array)

---

### 📋 Outputs

For each audio & each (`n_melbands`, `band_idx`) configuration:

* Framewise predictions saved as:

  ```csv
  frame, n_melbands, band_idx, azimuth, elevation, gt_azimuth, gt_elevation
  ```

  in `predictions_withgt.csv`

* Runs for:

  * `n_melbands` = 3 … 12
  * `band_idx` = 0 … (n\_melbands – 1)

* Automatically computes & reports:

  * Mean Angular Error (MAE) & Standard Deviation (STD) per (`n_melbands`, `band_idx`)
  * Distance vs. error analysis
  * Visualization plots framewise with ground truth, predictions also plotted

---
