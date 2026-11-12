
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from matplotlib.widgets import Button


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "radar_xyz.csv"

OUTPUT_DIR = "radar_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DBSCAN
EPS = 0.35
MIN_SAMPLES = 2
MIN_SNR = 10

# Tracking
SINGLE_TARGET_MODE = True
MAX_TRACK_DISTANCE = 1.0
MAX_MISSING_FRAMES = 5

# Actual time between consecutive radar frames (seconds)
FRAME_PERIOD = 0.1

# Movement classification
VELOCITY_THRESHOLD = 0.15
NEGATIVE_IS_APPROACHING = True

# Region of interest (metres)
ROI_X = (0.0, 5.0)
ROI_Y = (-2.0, 2.0)
ROI_Z = (-2.0, 2.0)

# Fixed plot boundaries for every frame
# Change these limits to suit your radar's measurement area.
X_LIMITS = (-6.0, 6.0)
Y_LIMITS = (-6.0, 6.0)
Z_LIMITS = (-6.0, 6.0)


# ============================================================
# 2. LOAD RADAR DATA
# ============================================================

print("\nLoading radar data...")

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

required_columns = [
    "FrameIndex", "X", "Y", "Z", "Velocity", "SNR"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=required_columns, inplace=True)

df = df[df["SNR"] >= MIN_SNR].copy()
df["FrameIndex"] = df["FrameIndex"].astype(int)

if df.empty:
    raise ValueError("No valid radar points found.")

frames = sorted(df["FrameIndex"].unique())

print("Frames:", len(frames))
print("Valid points:", len(df))


# ============================================================
# 3. COORDINATES
# ============================================================

# Use the original XYZ coordinates without any axis correction.
# No rotation or coordinate transformation is applied.

print("Using original radar XYZ coordinates.")


# ============================================================
# 4. DBSCAN CLUSTERING
# ============================================================

print("\nPerforming DBSCAN clustering...")

clustered_frames = []
centroid_records = []

for frame in frames:

    frame_data = df[
        df["FrameIndex"] == frame
    ].copy()

    points = frame_data[
        ["X", "Y", "Z"]
    ].to_numpy()

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

if centroids_df.empty:
    raise ValueError(
        "No clusters detected. Increase EPS or reduce MIN_SAMPLES."
    )

print("DBSCAN completed.")
print("Total clusters:", len(centroids_df))


# ============================================================
# 5. TARGET TRACKING
# ============================================================

print("\nTracking targets...")

tracking_records = []

if SINGLE_TARGET_MODE:

    # Single physical object experiment.
    # Maintain one persistent target ID.

    TARGET_ID = 1
    previous_position = None

    for frame in frames:

        detections = centroids_df[
            centroids_df["FrameIndex"] == frame
        ].copy()

        if detections.empty:
            continue

        if previous_position is None:

            detection = detections.iloc[0]

        else:

            positions = detections[
                ["X", "Y", "Z"]
            ].to_numpy()

            distances = np.linalg.norm(
                positions - previous_position,
                axis=1
            )

            detection = detections.iloc[
                np.argmin(distances)
            ]

        position = detection[
            ["X", "Y", "Z"]
        ].to_numpy(dtype=float)

        tracking_records.append({

            "FrameIndex": frame,
            "TargetID": TARGET_ID,

            "X": detection["X"],
            "Y": detection["Y"],
            "Z": detection["Z"],

            "Velocity": detection["Velocity"],
            "SNR": detection["SNR"],

            "PointCount": detection["PointCount"]
        })

        previous_position = position

else:

    # Multi-object tracking using Hungarian assignment.

    tracks = {}
    next_target_id = 1

    for frame in frames:

        detections = centroids_df[
            centroids_df["FrameIndex"] == frame
        ].copy().reset_index(drop=True)

        if detections.empty:
            continue

        positions = detections[
            ["X", "Y", "Z"]
        ].to_numpy()

        active_tracks = {
            target_id: track
            for target_id, track in tracks.items()
            if frame - track["last_frame"] <= MAX_MISSING_FRAMES
        }

        track_ids = list(active_tracks.keys())
        assignments = {}

        if track_ids:

            previous_positions = np.array([
                active_tracks[target_id]["position"]
                for target_id in track_ids
            ])

            distances = np.linalg.norm(
                previous_positions[:, None, :] -
                positions[None, :, :],
                axis=2
            )

            row_indices, col_indices = linear_sum_assignment(
                distances
            )

            for row, col in zip(row_indices, col_indices):

                if distances[row, col] <= MAX_TRACK_DISTANCE:

                    assignments[col] = track_ids[row]

        for detection_index in range(len(detections)):

            if detection_index not in assignments:

                assignments[detection_index] = next_target_id
                next_target_id += 1

        for detection_index, target_id in assignments.items():

            detection = detections.iloc[detection_index]

            position = detection[
                ["X", "Y", "Z"]
            ].to_numpy(dtype=float)

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

if tracking_df.empty:
    raise ValueError("No targets could be tracked.")

tracking_df.sort_values(
    ["TargetID", "FrameIndex"],
    inplace=True
)

print("Tracking completed.")
print(
    "Target IDs:",
    sorted(tracking_df["TargetID"].unique())
)


# ============================================================
# 6. MOVEMENT CLASSIFICATION
# ============================================================

def classify_movement(velocity):

    if abs(velocity) < VELOCITY_THRESHOLD:
        return "Stationary"

    if NEGATIVE_IS_APPROACHING:

        return "Approaching" if velocity < 0 else "Moving Away"

    return "Approaching" if velocity > 0 else "Moving Away"


tracking_df["Movement"] = tracking_df[
    "Velocity"
].apply(classify_movement)


# ============================================================
# 7. DWELL-TIME CALCULATION
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

    visits = []
    current_visit = []
    previous_frame = None

    for _, row in inside_frames.iterrows():

        current_frame = int(row["FrameIndex"])

        if (
            previous_frame is not None
            and current_frame - previous_frame > MAX_MISSING_FRAMES
        ):

            visits.append(current_visit)
            current_visit = []

        current_visit.append(row)
        previous_frame = current_frame

    if current_visit:
        visits.append(current_visit)

    for visit_number, visit in enumerate(visits, start=1):

        start_frame = int(visit[0]["FrameIndex"])
        end_frame = int(visit[-1]["FrameIndex"])

        dwell_time = (
            (end_frame - start_frame + 1) * FRAME_PERIOD
        )

        dwell_records.append({

            "TargetID": target_id,
            "Visit": visit_number,

            "StartFrame": start_frame,
            "EndFrame": end_frame,

            "ObservedFrames": len(visit),
            "DwellTimeSeconds": dwell_time
        })


dwell_df = pd.DataFrame(dwell_records)

dwell_df.to_csv(
    os.path.join(OUTPUT_DIR, "dwell_time.csv"),
    index=False
)

tracking_df.to_csv(
    os.path.join(OUTPUT_DIR, "target_tracking.csv"),
    index=False
)

print("Movement and dwell-time calculations completed.")


# ============================================================
# 8. FIXED 3D AXIS CONFIGURATION
# ============================================================

def set_fixed_axes(ax):

    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    ax.set_zlim(Z_LIMITS)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Keep the same physical scale on all axes
    ax.set_box_aspect((
        X_LIMITS[1] - X_LIMITS[0],
        Y_LIMITS[1] - Y_LIMITS[0],
        Z_LIMITS[1] - Z_LIMITS[0]
    ))

    # Show the origin in the same position
    ax.scatter(
        0, 0, 0,
        color="black",
        marker="+",
        s=80,
        label="Origin"
    )


# ============================================================
# 9. INTERACTIVE FRAME-BY-FRAME VIEWER
# ============================================================

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

    set_fixed_axes(ax)

    ax.set_title(f"Radar Point Cloud - Frame {frame}")

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


ax_previous = plt.axes([0.35, 0.05, 0.12, 0.06])
ax_next = plt.axes([0.53, 0.05, 0.12, 0.06])

button_previous = Button(ax_previous, "Previous")
button_next = Button(ax_next, "Next")

button_previous.on_clicked(previous_frame)
button_next.on_clicked(next_frame)

plot_frame(current_index)

plt.show()


# ============================================================
# 10. SEPARATE TRAJECTORY PLOT FOR EACH TARGET
# ============================================================

print("\nDisplaying individual target trajectories...")

target_ids = sorted(
    tracking_df["TargetID"].unique()
)

for target_id in target_ids:

    target_data = tracking_df[
        tracking_df["TargetID"] == target_id
    ].sort_values("FrameIndex")

    if len(target_data) < 2:

        print(
            f"Skipping Target {target_id}: "
            "insufficient trajectory points."
        )

        continue

    x = target_data["X"].to_numpy()
    y = target_data["Y"].to_numpy()
    z = target_data["Z"].to_numpy()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        x, y, z,
        marker="o",
        linewidth=2,
        markersize=4,
        label="Trajectory"
    )

    ax.scatter(
        x[0], y[0], z[0],
        color="green",
        s=100,
        label="Start"
    )

    ax.scatter(
        x[-1], y[-1], z[-1],
        color="red",
        s=100,
        label="End"
    )

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

    set_fixed_axes(ax)

    ax.set_title(f"3D Trajectory - Target {target_id}")
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n====================================")
print("RADAR PROCESSING COMPLETED")
print("====================================")

print("Frames processed:", len(frames))
print("Valid radar points:", len(df))
print("Clusters detected:", len(centroids_df))
print("Targets identified:", tracking_df["TargetID"].nunique())

for target_id in target_ids:

    target_data = tracking_df[
        tracking_df["TargetID"] == target_id
    ]

    print(
        f"Target {target_id}: "
        f"{len(target_data)} observations"
    )

print("\nCSV outputs saved in:", OUTPUT_DIR)
print("No plot images were saved.")
