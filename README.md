# mmWave Radar Point Cloud Analysis

A Python-based processing pipeline for analysing point-cloud data collected using the **Ai-Thinker RD-60 60 GHz FMCW mmWave Radar Module**.

The system processes recorded or generated CSV data to perform point-cloud clustering, target tracking, movement classification, dwell-time calculation, and 3D visualisation.

> **Current implementation:** This project operates on previously recorded or generated CSV data. It does not currently perform real-time radar processing.

---

## 1. Hardware and Technologies

### Hardware
- Ai-Thinker RD-60 60 GHz FMCW mmWave Radar Module
- Computer for offline data processing

### Software and Libraries
- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

---

## 2. Radar Orientation and Coordinate System

The RD-60 provides point-cloud measurements containing the range, azimuth, and elevation of detected points.

These measurements are represented using Cartesian coordinates (`X`, `Y`, `Z`) for clustering, tracking, and visualisation.

### Original coordinate convention

The spherical-to-Cartesian conversion used to generate XYZ coordinates is:

\[
X=R\cos(E)\cos(A)
\]

\[
Y=R\cos(E)\sin(A)
\]

\[
Z=R\sin(E)
\]

Where:

- `R` = Range in metres
- `A` = Azimuth in degrees
- `E` = Elevation in degrees

The equations assume:
- Positive X is the zero-azimuth direction.
- Positive Y is the direction of increasing azimuth.
- Positive Z is vertically upward.
- Azimuth is measured in the XY plane.
- Elevation is measured relative to the XY plane.

Angles must be converted to radians before applying these equations in Python.

### Physical mounting orientation

For the mounting arrangement discussed for this project, the radar module is rotated 90° clockwise relative to its original reference orientation.

The corresponding coordinate relationship is:

| Original axis | Mounted coordinate |
|---|---|
| Original X | Negative Z |
| Original Y | Positive Y |
| Original Z | Positive X |

Equivalently:

\[
X_{\text{mounted}}=Z_{\text{original}}
\]

\[
Y_{\text{mounted}}=Y_{\text{original}}
\]

\[
Z_{\text{mounted}}=-X_{\text{original}}
\]

This relationship is provided as a reference for understanding the physical mounting orientation.

**Important:** The current Python implementation does not apply this transformation. It uses the `X`, `Y`, and `Z` values in `radar_xyz.csv` directly. Therefore, the displayed axes represent the coordinate system already present in the CSV, not an automatically corrected world-coordinate system.

If the coordinate conversion script has already applied the mounting transformation, do not apply it again.

---

## 3. Input Data Format

The analysis script reads:

`radar_xyz.csv`

The expected columns are:

| Column | Description | Unit |
|---|---|---|
| FrameIndex | Frame number | Integer |
| X | X-coordinate | Metres |
| Y | Y-coordinate | Metres |
| Z | Z-coordinate | Metres |
| Velocity | Radial velocity | m/s |
| SNR | Signal-to-noise ratio | dB |

Example:

```csv
FrameIndex,X,Y,Z,Velocity,SNR
1,0.85,-0.12,0.35,-1.5,29
1,0.92,-0.10,0.38,-1.5,31
1,1.10,-0.15,0.42,-0.75,25
2,0.88,-0.11,0.36,-1.5,28
2,0.95,-0.09,0.39,-0.75,27
```

Each row represents one detected point in a particular radar frame.

The input file should contain valid numerical values. The script removes invalid rows and filters points based on the configured minimum SNR.

---

## 4. Processing Pipeline

The processing pipeline consists of the following stages:

1. **Data loading:** Reads the XYZ point-cloud CSV and removes invalid measurements.
2. **SNR filtering:** Removes points below the configured signal-strength threshold.
3. **DBSCAN clustering:** Groups nearby points into clusters for each frame.
4. **Centroid extraction:** Calculates the centre position and average measurements of each cluster.
5. **Target tracking:** Associates cluster centroids across consecutive frames.
6. **Movement classification:** Classifies tracked targets as stationary, approaching, or moving away.
7. **Dwell-time calculation:** Measures how long tracked targets remain inside a defined Region of Interest (ROI).
8. **Visualisation:** Displays individual radar frames and target trajectories in 3D.

### 4.1 DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups spatially close points into clusters.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand. It can also identify isolated points as noise.

The clustering is performed independently for each frame using the XYZ coordinates.

### 4.2 Centroid Extraction

After clustering, the centroid of each valid cluster is calculated using the mean of its point coordinates:

\[
C_x=\frac{1}{N}\sum_{i=1}^{N}X_i
\]

\[
C_y=\frac{1}{N}\sum_{i=1}^{N}Y_i
\]

\[
C_z=\frac{1}{N}\sum_{i=1}^{N}Z_i
\]

The average velocity, average SNR, and number of points are also recorded for each cluster.

### 4.3 Target Tracking

Cluster centroids are associated across frames to maintain target identities.

Two tracking modes are available:

- **Single-target mode:** Selects the largest cluster in the first frame and subsequently follows the nearest centroid.
- **Multi-target mode:** Uses the Hungarian assignment algorithm to associate multiple centroids with existing target IDs based on spatial distance.

The multi-target implementation is a distance-based tracker and does not currently use motion prediction.

### 4.4 Movement Classification

Movement is classified using the average radial velocity of each tracked target.

The configured threshold separates stationary measurements from moving measurements.

Depending on the radar's velocity sign convention, a moving target is classified as approaching or moving away.

### 4.5 Dwell-Time Calculation

A Region of Interest (ROI) is defined using minimum and maximum limits along all three axes.

A target is considered inside the ROI when:

\[
X_{\min}\leq X\leq X_{\max}
\]

\[
Y_{\min}\leq Y\leq Y_{\max}
\]

\[
Z_{\min}\leq Z\leq Z_{\max}
\]

Dwell time is estimated from the number of frames during which the target is observed inside the ROI:

\[
T_{\text{dwell}}=N_{\text{frames}}\times T_{\text{frame}}
\]

Here, \(T_{\text{frame}}\) is the configured frame interval.

---

## 5. Configuration Parameters

All major parameters can be modified in the configuration section at the beginning of `radar_analysis.py`.

### 5.1 Clustering Parameters

| Parameter | Default | Description |
|---|---:|---|
| `EPS` | `0.35` | Maximum distance in metres between neighbouring points for DBSCAN. |
| `MIN_SAMPLES` | `2` | Minimum number of points required to form a cluster. |
| `MIN_SNR` | `10` | Minimum SNR in dB for a point to be retained. |

**Parameter tuning:**

- Increase `EPS` if points belonging to the same object are being separated into multiple clusters.
- Decrease `EPS` if nearby objects are being merged.
- Increase `MIN_SAMPLES` to reject small groups of detections.
- Increase `MIN_SNR` to remove weaker detections and potentially reduce noise.

The appropriate values depend on the radar's detection density and the environment.

### 5.2 Tracking Parameters

| Parameter | Default | Description |
|---|---:|---|
| `SINGLE_TARGET_MODE` | `True` | Enables single-target tracking when set to `True`. |
| `MAX_TRACK_DISTANCE` | `1.0` | Maximum centroid displacement in metres allowed for a match. |
| `MAX_MISSING_FRAMES` | `5` | Maximum permitted frame gap before a track is considered expired. |

**Tracking mode:**

```python
SINGLE_TARGET_MODE = True
```

Use this for experiments involving one physical target.

```python
SINGLE_TARGET_MODE = False
```

Use this when multiple targets need to be tracked.

For multi-target tracking, `MAX_TRACK_DISTANCE` should be adjusted according to the expected movement between frames. A very small value can cause track fragmentation, while a very large value can cause incorrect associations.

### 5.3 Movement Classification Parameters

| Parameter | Default | Description |
|---|---:|---|
| `VELOCITY_THRESHOLD` | `0.15` | Velocity magnitude below which a target is classified as stationary. |
| `NEGATIVE_IS_APPROACHING` | `True` | Determines whether negative radial velocity represents an approaching target. |

The classification logic is:

| Condition | Classification |
|---|---|
| Absolute velocity below threshold | Stationary |
| Negative velocity, when configured as approaching | Approaching |
| Positive velocity, when configured as approaching | Moving Away |

The RD-60 velocity sign convention should be experimentally verified before interpreting these labels.

### 5.4 Region of Interest Parameters

The default ROI is:

```python
ROI_X = (0.0, 5.0)
ROI_Y = (-2.0, 2.0)
ROI_Z = (-2.0, 2.0)
```

| Parameter | Default | Description |
|---|---|---|
| `ROI_X` | `(0.0, 5.0)` | Minimum and maximum X-coordinate. |
| `ROI_Y` | `(-2.0, 2.0)` | Minimum and maximum Y-coordinate. |
| `ROI_Z` | `(-2.0, 2.0)` | Minimum and maximum Z-coordinate. |

The ROI defines the spatial area in which dwell time is calculated.

These values are example settings and should be adjusted to the actual radar placement and monitored area.

### 5.5 Dwell-Time Parameters

| Parameter | Default | Description |
|---|---:|---|
| `FRAME_PERIOD` | `0.1` seconds | Time interval between consecutive frames. |
| `MAX_MISSING_FRAMES` | `5` | Permitted gap between observations belonging to the same visit. |

The frame interval must be based on the actual data acquisition rate.

For example, if the radar produces 10 frames per second:

\[
T_{\text{frame}}=\frac{1}{10}=0.1\text{ seconds}
\]

The current dwell-time implementation assumes regular frame sampling and uses frame numbers to estimate elapsed time. Missing observations within the allowed gap are included in the estimated visit duration.

---

## 6. Visualisation

The script generates two types of 3D visualisation.

### 6.1 Interactive Frame Viewer

The frame viewer displays one radar frame at a time.

It includes:

- Detected point-cloud measurements.
- DBSCAN clusters in different colours.
- Noise points in light grey.
- Tracked target centroids marked with red X symbols.
- Previous and Next buttons for navigating between frames.

All frames use fixed axis limits to make spatial comparisons easier.

### 6.2 Target Trajectory Plot

A separate 3D trajectory is displayed for each tracked target.

The trajectory includes:

- Green marker: Starting position.
- Red marker: Ending position.
- Dark blue to white colour gradient: Progression over time.
- Direction arrows: Direction of movement between consecutive observations.

The colour gradient represents chronological progression, not target velocity.

Plots are displayed interactively and are not saved as image files.

---

## 7. Generated Output Files

The script generates the following CSV files:

| File | Description |
|---|---|
| `point_cloud_clusters.csv` | Original point measurements with their DBSCAN cluster labels. Noise points have cluster label `-1`. |
| `cluster_centroids.csv` | Centroid coordinates, average velocity, average SNR, and point count for each cluster. |
| `target_tracking.csv` | Tracked target positions, target IDs, velocity, and movement classifications. |
| `dwell_time.csv` | Start frame, end frame, number of frames, and estimated dwell time for each ROI visit. |

These files can be used for further analysis, reporting, and visualisation.

---

## 8. Installation and Execution

### Step 1: Install Python Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

### Step 2: Arrange the Project Files

```text
radar-analysis/
│
├── radar_analysis.py
├── radar_xyz.csv
│
├── point_cloud_clusters.csv       # Generated
├── cluster_centroids.csv          # Generated
├── target_tracking.csv             # Generated
└── dwell_time.csv                  # Generated
```

### Step 3: Configure Parameters

Open `radar_analysis.py` and modify the clustering, tracking, ROI, and frame-period parameters as required.

### Step 4: Run the Script

```bash
python radar_analysis.py
```

The program processes the input CSV, generates the output files, and opens the interactive visualisations.

---

## 9. Limitations and Assumptions

- The current implementation processes stored CSV data rather than live radar streams.
- DBSCAN clustering is performed independently for each frame.
- The single-target tracker assumes that only one target is being monitored.
- Multi-target tracking uses spatial distance and Hungarian assignment without motion prediction.
- Target identities may change when targets cross paths or move beyond the association threshold.
- Movement classification depends on the radar's radial-velocity sign convention.
- Dwell time assumes a known and approximately constant frame interval.
- The coordinate transformation associated with physical radar mounting is not applied by the current analysis script.
- The ROI and clustering thresholds require calibration for the actual installation environment.

---

## 10. Future Improvements

Possible extensions include:

- Real-time processing directly from the radar data stream.
- Improved multi-target tracking using Kalman filters.
- Adaptive DBSCAN parameters for varying point-cloud density.
- More robust handling of missing frames and track fragmentation.
- Calibration of the radar coordinate system against the physical environment.
- A dashboard for displaying target positions, movement, and dwell-time statistics.

---

## 11. References

- Ai-Thinker RD-60 mmWave Radar Module documentation.
- Scikit-learn DBSCAN documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- SciPy Linear Sum Assignment documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- Matplotlib documentation: https://matplotlib.org/stable/