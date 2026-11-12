
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "radar_xyz.csv"

OUTPUT_DIR = "radar_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DBSCAN parameters
EPS = 1.5
MIN_SAMPLES = 4

# Tracking parameters
MAX_TRACK_DISTANCE = 1.5
MAX_MISSING_FRAMES = 5

# Radar frame timing
# Replace this with the actual interval between radar frames.
FRAME_PERIOD = 0.1

# Movement classification
VELOCITY_THRESHOLD = 0.15

# Set this after verifying the RD-60 velocity sign convention.
NEGATIVE_IS_APPROACHING = True

# Region of interest (ROI), in metres
ROI_X = (0.0, 5.0)
ROI_Y = (-2.0, 2.0)
ROI_Z = (-2.0, 2.0)

# Minimum SNR threshold
MIN_SNR = 10


# ============================================================
# 2. LOAD AND PREPROCESS CSV
# ============================================================

print("\nLoading radar data...")

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Remove whitespace from column names
df.columns = df.columns.str.strip()

required_columns = [
    "FrameIndex", "X", "Y", "Z", "Velocity", "SNR"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Convert required fields to numeric
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

# ============================================================
# RD-60 COORDINATE TRANSFORMATION
# Module rotated 90 degrees clockwise
# ============================================================

df["X_old"] = df["X"].copy()
df["Y_old"] = df["Y"].copy()
df["Z_old"] = df["Z"].copy()

df["X"] = df["Z_old"]
df["Y"] = df["Y_old"]
df["Z"] = -df["X_old"]

print("RD-60 coordinate transformation applied.")

# Filter low-SNR points
df = df[df["SNR"] >= MIN_SNR].copy()

df["FrameIndex"] = df["FrameIndex"].astype(int)

frames = sorted(df["FrameIndex"].unique())

if len(frames) == 0:
    raise ValueError("No valid radar frames found.")

print("Total frames:", len(frames))
print("Total valid points:", len(df))


# ============================================================
# 3. DBSCAN CLUSTERING
# ============================================================

print("\nPerforming DBSCAN clustering...")

clustered_frames = []
centroid_records = []

for frame in frames:

    frame_data = df[df["FrameIndex"] == frame].copy()

    if frame_data.empty:
        continue

    points = frame_data[["X", "Y", "Z"]].to_numpy()

    if len(points) < MIN_SAMPLES:
        frame_data["ClusterID"] = -1
        clustered_frames.append(frame_data)
        continue

    clustering = DBSCAN(
        eps=EPS,
        min_samples=MIN_SAMPLES
    )

    labels = clustering.fit_predict(points)

    frame_data["ClusterID"] = labels

    clustered_frames.append(frame_data)

    # Calculate centroid for every non-noise cluster
    for cluster_id in sorted(set(labels)):

        if cluster_id == -1:
            continue

        cluster_points = frame_data[
            frame_data["ClusterID"] == cluster_id
        ]

        centroid_records.append({
            "FrameIndex": frame,
            "ClusterID": cluster_id,

            "X": cluster_points["X"].mean(),
            "Y": cluster_points["Y"].mean(),
            "Z": cluster_points["Z"].mean(),

            "Velocity": cluster_points["Velocity"].mean(),
            "SNR": cluster_points["SNR"].mean(),

            "PointCount": len(cluster_points)
        })


clustered_df = pd.concat(
    clustered_frames,
    ignore_index=True
)

centroids_df = pd.DataFrame(centroid_records)

clustered_df.to_csv(
    os.path.join(OUTPUT_DIR, "point_cloud_clusters.csv"),
    index=False
)

centroids_df.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_centroids.csv"),
    index=False
)

print("DBSCAN completed.")
print("Total clusters:", len(centroids_df))

if centroids_df.empty:
    raise ValueError(
        "DBSCAN did not detect any clusters. "
        "Try reducing MIN_SAMPLES or increasing EPS."
    )


# ============================================================
# 4. TARGET TRACKING
# ============================================================

print("\nTracking targets...")

tracks = {}
tracking_records = []

next_target_id = 1

for frame in frames:

    detections = centroids_df[
        centroids_df["FrameIndex"] == frame
    ].copy()

    if detections.empty:
        continue

    detection_positions = detections[
        ["X", "Y", "Z"]
    ].to_numpy()

    # Find tracks that have not been missing too long
    active_tracks = {}

    for target_id, track in tracks.items():

        missing_frames = frame - track["last_frame"]

        if missing_frames <= MAX_MISSING_FRAMES:
            active_tracks[target_id] = track

    # Build possible track-detection matches
    possible_matches = []

    for target_id, track in active_tracks.items():

        previous_position = track["position"]

        for detection_index, position in enumerate(
            detection_positions
        ):

            distance = np.linalg.norm(
                position - previous_position
            )

            if distance <= MAX_TRACK_DISTANCE:

                possible_matches.append((
                    distance,
                    target_id,
                    detection_index
                ))

    # Greedy nearest-distance assignment
    possible_matches.sort(key=lambda x: x[0])

    matched_tracks = set()
    matched_detections = set()

    assignments = {}

    for distance, target_id, detection_index in possible_matches:

        if target_id in matched_tracks:
            continue

        if detection_index in matched_detections:
            continue

        assignments[detection_index] = target_id

        matched_tracks.add(target_id)
        matched_detections.add(detection_index)

    # Create new target IDs for unmatched detections
    for detection_index in range(len(detections)):

        if detection_index not in assignments:

            assignments[detection_index] = next_target_id
            next_target_id += 1

    # Update tracks and record target positions
    for detection_index, target_id in assignments.items():

        detection = detections.iloc[detection_index]

        position = np.array([
            detection["X"],
            detection["Y"],
            detection["Z"]
        ])

        tracks[target_id] = {
            "position": position,
            "last_frame": frame
        }

        tracking_records.append({
            "FrameIndex": frame,
            "TargetID": target_id,

            "X": detection["X"],
            "Y": detection["Y"],
            "Z": detection["Z"],

            "Velocity": detection["Velocity"],
            "SNR": detection["SNR"],

            "PointCount": detection["PointCount"]
        })


tracking_df = pd.DataFrame(tracking_records)

tracking_df.to_csv(
    os.path.join(OUTPUT_DIR, "target_tracking.csv"),
    index=False
)

print("Tracking completed.")
print("Total target IDs:", tracking_df["TargetID"].nunique())


# ============================================================
# 5. MOVEMENT CLASSIFICATION
# ============================================================

def classify_movement(velocity):

    if abs(velocity) < VELOCITY_THRESHOLD:
        return "Stationary"

    if NEGATIVE_IS_APPROACHING:

        if velocity < 0:
            return "Approaching"

        return "Moving Away"

    else:

        if velocity > 0:
            return "Approaching"

        return "Moving Away"


tracking_df["Movement"] = tracking_df[
    "Velocity"
].apply(classify_movement)

tracking_df.to_csv(
    os.path.join(OUTPUT_DIR, "target_tracking.csv"),
    index=False
)


# ============================================================
# 6. DWELL-TIME CALCULATION
# ============================================================

def inside_roi(row):

    return (
        ROI_X[0] <= row["X"] <= ROI_X[1]
        and ROI_Y[0] <= row["Y"] <= ROI_Y[1]
        and ROI_Z[0] <= row["Z"] <= ROI_Z[1]
    )


tracking_df["InsideROI"] = tracking_df.apply(
    inside_roi,
    axis=1
)

dwell_records = []

for target_id in sorted(tracking_df["TargetID"].unique()):

    target_data = tracking_df[
        tracking_df["TargetID"] == target_id
    ].sort_values("FrameIndex")

    inside_frames = target_data[
        target_data["InsideROI"]
    ]

    if inside_frames.empty:
        continue

    # Divide visits when there is a gap in ROI observations
    visits = []
    current_visit = []

    previous_frame = None

    for _, row in inside_frames.iterrows():

        current_frame = int(row["FrameIndex"])

        if (
            previous_frame is not None
            and current_frame - previous_frame
            > MAX_MISSING_FRAMES
        ):

            visits.append(current_visit)
            current_visit = []

        current_visit.append(row)
        previous_frame = current_frame

    if current_visit:
        visits.append(current_visit)

    # Calculate dwell time for each visit
    for visit_number, visit in enumerate(visits, start=1):

        start_frame = int(visit[0]["FrameIndex"])
        end_frame = int(visit[-1]["FrameIndex"])

        observed_frames = len(visit)

        # Estimate elapsed time from frame indices.
        # This assumes observations are regularly sampled.
        dwell_time = (
            (end_frame - start_frame + 1) * FRAME_PERIOD
        )

        dwell_records.append({

            "TargetID": target_id,
            "Visit": visit_number,

            "StartFrame": start_frame,
            "EndFrame": end_frame,

            "ObservedFrames": observed_frames,
            "DwellTimeSeconds": dwell_time
        })


dwell_df = pd.DataFrame(dwell_records)

dwell_df.to_csv(
    os.path.join(OUTPUT_DIR, "dwell_time.csv"),
    index=False
)

print("Dwell-time calculation completed.")


# ============================================================
# 7. INTERACTIVE FRAME-BY-FRAME VISUALIZATION
# ============================================================

from matplotlib.widgets import Button

print("\nOpening frame viewer...")

frame_list = frames
current_index = 0

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

plt.subplots_adjust(bottom=0.18)


def plot_frame(index):

    ax.clear()

    frame = frame_list[index]

    frame_points = clustered_df[
        clustered_df["FrameIndex"] == frame
    ]

    # Plot DBSCAN clusters
    for cluster_id in sorted(
        frame_points["ClusterID"].unique()
    ):

        cluster_points = frame_points[
            frame_points["ClusterID"] == cluster_id
        ]

        if cluster_id == -1:

            ax.scatter(
                cluster_points["X"],
                cluster_points["Y"],
                cluster_points["Z"],
                color="gray",
                s=10,
                label="Noise"
            )

        else:

            ax.scatter(
                cluster_points["X"],
                cluster_points["Y"],
                cluster_points["Z"],
                s=25,
                label=f"Cluster {cluster_id}"
            )

    # Plot target centroids
    target_points = tracking_df[
        tracking_df["FrameIndex"] == frame
    ]

    for _, target in target_points.iterrows():

        ax.scatter(
            target["X"],
            target["Y"],
            target["Z"],
            color="red",
            marker="X",
            s=100
        )

        ax.text(
            target["X"],
            target["Y"],
            target["Z"],
            f" T{int(target['TargetID'])}"
        )

    ax.set_title(f"Radar Point Cloud - Frame {frame}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.legend(loc="upper right")

    fig.canvas.draw_idle()


def next_frame(event):

    global current_index

    current_index = min(
        current_index + 1,
        len(frame_list) - 1
    )

    plot_frame(current_index)


def previous_frame(event):

    global current_index

    current_index = max(
        current_index - 1,
        0
    )

    plot_frame(current_index)


# Navigation buttons
ax_previous = plt.axes([0.35, 0.05, 0.12, 0.06])
ax_next = plt.axes([0.53, 0.05, 0.12, 0.06])

button_previous = Button(ax_previous, "Previous")
button_next = Button(ax_next, "Next")

button_previous.on_clicked(previous_frame)
button_next.on_clicked(next_frame)

plot_frame(current_index)

plt.show()


# ============================================================
# 9. SEPARATE 3D TRAJECTORY PLOT FOR EACH TARGET
# ============================================================

print("\nGenerating individual target trajectory plots...")

target_ids = sorted(
    tracking_df["TargetID"].unique()
)

for target_id in target_ids:

    target_data = tracking_df[
        tracking_df["TargetID"] == target_id
    ].sort_values("FrameIndex")

    if len(target_data) < 2:
        print(f"Skipping Target {target_id}: insufficient points.")
        continue

    x = target_data["X"].to_numpy()
    y = target_data["Y"].to_numpy()
    z = target_data["Z"].to_numpy()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory
    ax.plot(
        x, y, z,
        marker="o",
        linewidth=2,
        markersize=4,
        label="Trajectory"
    )

    # Starting position
    ax.scatter(
        x[0], y[0], z[0],
        color="green",
        s=100,
        label="Start"
    )

    # Ending position
    ax.scatter(
        x[-1], y[-1], z[-1],
        color="red",
        s=100,
        label="End"
    )

    # Direction arrows
    for i in range(len(x) - 1):

        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        dz = z[i + 1] - z[i]

        ax.quiver(
            x[i], y[i], z[i],
            dx, dy, dz,
            color="blue",
            arrow_length_ratio=0.3,
            linewidth=1.5
        )

    ax.set_title(f"3D Trajectory - Target {target_id}")

    ax.set_xlabel("New X (m)")
    ax.set_ylabel("New Y (m)")
    ax.set_zlabel("New Z (m)")

    ax.legend()
    plt.tight_layout()

    # Display only; do not save
    plt.show()
    plt.close()


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("RADAR PROCESSING COMPLETED")
print("===================================")

print("Frames processed:", len(frames))
print("Valid radar points:", len(df))
print("Clusters detected:", len(centroids_df))
print("Targets identified:", tracking_df["TargetID"].nunique())

print("\nTarget IDs:")

for target_id in target_ids:

    target_data = tracking_df[
        tracking_df["TargetID"] == target_id
    ]

    print(
        f"Target {target_id}: "
        f"{len(target_data)} observations"
    )

print("\nOutput files are saved in:", OUTPUT_DIR)
