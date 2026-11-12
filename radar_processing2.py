import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "radar_xyz.csv"

# DBSCAN
EPS = 0.35
MIN_SAMPLES = 2
MIN_SNR = 10

# Tracking
SINGLE_TARGET_MODE = True

MAX_TRACK_DISTANCE = 1.0
MAX_MISSING_FRAMES = 5

# Movement classification
VELOCITY_THRESHOLD = 0.15

# Set this according to the radar's velocity convention.
# Verify whether negative velocity means approaching.
NEGATIVE_IS_APPROACHING = True

# Region of Interest (ROI), in metres
ROI_X = (0.0, 5.0)
ROI_Y = (-2.0, 2.0)
ROI_Z = (-2.0, 2.0)

# Actual time between consecutive frames, in seconds.
# Replace 0.1 with the measured frame interval of your dataset.
FRAME_PERIOD = 0.1

# Plot appearance
FIGURE_SIZE = (14, 11)
PLOT_PADDING = 0.08
MIN_AXIS_RANGE = 1.0

# Trajectory appearance
TRAJECTORY_LINE_WIDTH = 4
ARROW_COUNT = 12


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

required_columns = [
    "FrameIndex", "X", "Y", "Z", "Velocity", "SNR"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in {INPUT_FILE}: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

# Convert required columns to numeric values
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns).copy()

# Keep points above the minimum SNR
df = df[df["SNR"] >= MIN_SNR].copy()

df["FrameIndex"] = df["FrameIndex"].astype(int)

if df.empty:
    raise ValueError(
        "No valid radar points remain after cleaning. "
        "Check the CSV data and MIN_SNR setting."
    )

df = df.sort_values(["FrameIndex"]).reset_index(drop=True)

print(f"Loaded {len(df)} radar points.")
print(f"Frames: {df['FrameIndex'].min()} to {df['FrameIndex'].max()}")


# ============================================================
# 3. CALCULATE FIXED PLOT LIMITS
# ============================================================

def calculate_axis_limits(data):
    limits = {}

    for axis in ["X", "Y", "Z"]:
        minimum = min(data[axis].min(), 0)
        maximum = max(data[axis].max(), 0)

        span = maximum - minimum

        if span < MIN_AXIS_RANGE:
            center = (minimum + maximum) / 2
            minimum = center - MIN_AXIS_RANGE / 2
            maximum = center + MIN_AXIS_RANGE / 2
            span = MIN_AXIS_RANGE

        padding = span * PLOT_PADDING

        limits[axis] = (
            minimum - padding,
            maximum + padding
        )

    return limits


AXIS_LIMITS = calculate_axis_limits(df)


def set_fixed_axes(ax):
    """Apply the same coordinate limits to every 3D plot."""

    ax.set_xlim(AXIS_LIMITS["X"])
    ax.set_ylim(AXIS_LIMITS["Y"])
    ax.set_zlim(AXIS_LIMITS["Z"])

    ax.set_xlabel("X (m)", labelpad=12)
    ax.set_ylabel("Y (m)", labelpad=12)
    ax.set_zlabel("Z (m)", labelpad=12)

    # Preserve the physical proportions of the coordinate ranges.
    ax.set_box_aspect((
        AXIS_LIMITS["X"][1] - AXIS_LIMITS["X"][0],
        AXIS_LIMITS["Y"][1] - AXIS_LIMITS["Y"][0],
        AXIS_LIMITS["Z"][1] - AXIS_LIMITS["Z"][0]
    ))

    ax.scatter(
        0, 0, 0,
        color="black",
        marker="+",
        s=100,
        linewidths=2,
        label="Origin"
    )

    ax.view_init(elev=25, azim=-55)
    ax.tick_params(labelsize=9)


# ============================================================
# 4. DBSCAN CLUSTERING FOR EACH FRAME
# ============================================================

clustered_frames = []
centroid_records = []

frames = sorted(df["FrameIndex"].unique())

for frame_number in frames:

    frame_data = df[
        df["FrameIndex"] == frame_number
    ].copy()

    coordinates = frame_data[["X", "Y", "Z"]].to_numpy()

    if len(frame_data) < MIN_SAMPLES:
        frame_data["Cluster"] = -1
    else:
        clustering = DBSCAN(
            eps=EPS,
            min_samples=MIN_SAMPLES,
            algorithm="auto"
        )

        frame_data["Cluster"] = clustering.fit_predict(
            coordinates
        )

    clustered_frames.append(frame_data)

    # Calculate centroid of every valid cluster
    valid_clusters = sorted(
        frame_data.loc[
            frame_data["Cluster"] != -1,
            "Cluster"
        ].unique()
    )

    for cluster_id in valid_clusters:

        cluster_points = frame_data[
            frame_data["Cluster"] == cluster_id
        ]

        centroid_records.append({
            "FrameIndex": frame_number,
            "Cluster": int(cluster_id),
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

centroids = pd.DataFrame(centroid_records)

clustered_df.to_csv(
    "point_cloud_clusters.csv",
    index=False
)

centroids.to_csv(
    "cluster_centroids.csv",
    index=False
)

print(f"Clustering complete: {len(centroids)} cluster observations.")


# ============================================================
# 5. TARGET TRACKING
# ============================================================

def track_single_target(data):
    """
    Tracks one target.

    The first target is selected using the largest cluster.
    In subsequent frames, the nearest centroid is selected.
    """

    records = []
    previous_position = None

    for frame_number in frames:

        current = data[
            data["FrameIndex"] == frame_number
        ].copy()

        if current.empty:
            continue

        if previous_position is None:
            # Start with the largest cluster
            selected_index = current["PointCount"].idxmax()
        else:
            distances = np.linalg.norm(
                current[["X", "Y", "Z"]].to_numpy()
                - previous_position,
                axis=1
            )

            selected_index = current.index[
                np.argmin(distances)
            ]

            selected_distance = np.min(distances)

            if selected_distance > MAX_TRACK_DISTANCE:
                # Treat this as a new track segment
                previous_position = None
                selected_index = current["PointCount"].idxmax()

        selected = current.loc[selected_index]

        record = selected.to_dict()
        record["TargetID"] = 1

        records.append(record)

        previous_position = selected[
            ["X", "Y", "Z"]
        ].to_numpy(dtype=float)

    return pd.DataFrame(records)


def track_multiple_targets(data):
    """
    Multi-target tracking using Hungarian assignment.

    A centroid is matched to an existing target when its
    distance from the previous observation is within the
    configured maximum distance.
    """

    records = []
    active_tracks = {}
    next_target_id = 1

    for frame_number in frames:

        current = data[
            data["FrameIndex"] == frame_number
        ].copy().reset_index(drop=True)

        # Remove tracks that have been missing too long
        expired_tracks = []

        for target_id, track in active_tracks.items():
            if frame_number - track["last_frame"] > MAX_MISSING_FRAMES:
                expired_tracks.append(target_id)

        for target_id in expired_tracks:
            del active_tracks[target_id]

        if current.empty:
            continue

        current_positions = current[
            ["X", "Y", "Z"]
        ].to_numpy()

        active_ids = list(active_tracks.keys())

        matched_current = set()
        matched_tracks = set()

        if active_ids:

            previous_positions = np.array([
                active_tracks[target_id]["position"]
                for target_id in active_ids
            ])

            distances = np.linalg.norm(
                previous_positions[:, None, :]
                - current_positions[None, :, :],
                axis=2
            )

            row_indices, column_indices = linear_sum_assignment(
                distances
            )

            for row, column in zip(row_indices, column_indices):

                if distances[row, column] <= MAX_TRACK_DISTANCE:

                    target_id = active_ids[row]

                    matched_current.add(column)
                    matched_tracks.add(target_id)

                    active_tracks[target_id] = {
                        "position": current_positions[column],
                        "last_frame": frame_number
                    }

                    record = current.iloc[column].to_dict()
                    record["TargetID"] = target_id
                    records.append(record)

        # Unmatched centroids become new targets
        for index in range(len(current)):

            if index in matched_current:
                continue

            target_id = next_target_id
            next_target_id += 1

            active_tracks[target_id] = {
                "position": current_positions[index],
                "last_frame": frame_number
            }

            record = current.iloc[index].to_dict()
            record["TargetID"] = target_id
            records.append(record)

    return pd.DataFrame(records)


if centroids.empty:
    tracking = pd.DataFrame()
else:
    if SINGLE_TARGET_MODE:
        tracking = track_single_target(centroids)
    else:
        tracking = track_multiple_targets(centroids)


if tracking.empty:
    print("No targets were tracked.")
else:
    tracking = tracking.sort_values(
        ["TargetID", "FrameIndex"]
    ).reset_index(drop=True)

    tracking.to_csv(
        "target_tracking.csv",
        index=False
    )

    print(
        f"Tracking complete: "
        f"{tracking['TargetID'].nunique()} target(s)."
    )


# ============================================================
# 6. MOVEMENT CLASSIFICATION
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


if not tracking.empty:

    tracking["Movement"] = tracking["Velocity"].apply(
        classify_movement
    )

    tracking.to_csv(
        "target_tracking.csv",
        index=False
    )


# ============================================================
# 7. ROI AND DWELL-TIME CALCULATION
# ============================================================

def is_inside_roi(row):

    return (
        ROI_X[0] <= row["X"] <= ROI_X[1]
        and ROI_Y[0] <= row["Y"] <= ROI_Y[1]
        and ROI_Z[0] <= row["Z"] <= ROI_Z[1]
    )


def calculate_dwell_time(data):

    if data.empty:
        return pd.DataFrame()

    data = data.copy()

    data["InsideROI"] = data.apply(
        is_inside_roi,
        axis=1
    )

    inside_data = data[
        data["InsideROI"]
    ].copy()

    if inside_data.empty:
        return pd.DataFrame()

    results = []

    for target_id, target_data in inside_data.groupby("TargetID"):

        target_data = target_data.sort_values(
            "FrameIndex"
        ).reset_index(drop=True)

        visit_start = None
        previous_frame = None

        for _, row in target_data.iterrows():

            current_frame = int(row["FrameIndex"])

            if visit_start is None:
                visit_start = current_frame

            elif current_frame - previous_frame > MAX_MISSING_FRAMES:
                # Close the previous visit
                dwell_frames = previous_frame - visit_start + 1

                results.append({
                    "TargetID": target_id,
                    "StartFrame": visit_start,
                    "EndFrame": previous_frame,
                    "DwellFrames": dwell_frames,
                    "DwellTimeSeconds": dwell_frames * FRAME_PERIOD
                })

                visit_start = current_frame

            previous_frame = current_frame

        # Close the final visit
        if visit_start is not None:

            dwell_frames = previous_frame - visit_start + 1

            results.append({
                "TargetID": target_id,
                "StartFrame": visit_start,
                "EndFrame": previous_frame,
                "DwellFrames": dwell_frames,
                "DwellTimeSeconds": dwell_frames * FRAME_PERIOD
            })

    return pd.DataFrame(results)


dwell_results = calculate_dwell_time(tracking)

dwell_results.to_csv(
    "dwell_time.csv",
    index=False
)

print("\nDwell-time calculation complete.")

if not dwell_results.empty:
    print(dwell_results.to_string(index=False))
else:
    print("No target observations were found inside the ROI.")


# ============================================================
# 8. INTERACTIVE FRAME-BY-FRAME VIEWER
# ============================================================

def show_frame_viewer(data, tracked_data):

    frame_numbers = sorted(
        data["FrameIndex"].unique()
    )

    if not frame_numbers:
        return

    current_index = [0]

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, projection="3d")

    plt.subplots_adjust(
        bottom=0.16,
        top=0.90
    )

    def plot_frame(index):

        ax.clear()

        frame_number = frame_numbers[index]

        frame_data = data[
            data["FrameIndex"] == frame_number
        ]

        noise = frame_data[
            frame_data["Cluster"] == -1
        ]

        clustered = frame_data[
            frame_data["Cluster"] != -1
        ]

        # Noise points
        if not noise.empty:
            ax.scatter(
                noise["X"],
                noise["Y"],
                noise["Z"],
                color="lightgray",
                s=18,
                alpha=0.5,
                label="Noise"
            )

        # Clustered points
        if not clustered.empty:
            ax.scatter(
                clustered["X"],
                clustered["Y"],
                clustered["Z"],
                c=clustered["Cluster"],
                cmap="tab10",
                s=35,
                alpha=0.85,
                label="Clustered Points"
            )

        # Target centroid(s)
        if not tracked_data.empty:

            frame_targets = tracked_data[
                tracked_data["FrameIndex"] == frame_number
            ]

            if not frame_targets.empty:

                ax.scatter(
                    frame_targets["X"],
                    frame_targets["Y"],
                    frame_targets["Z"],
                    color="red",
                    marker="X",
                    s=180,
                    edgecolors="black",
                    linewidths=1.2,
                    label="Target Centroid"
                )

                for _, target in frame_targets.iterrows():
                    ax.text(
                        target["X"],
                        target["Y"],
                        target["Z"],
                        f" T{int(target['TargetID'])}",
                        fontsize=10,
                        color="black"
                    )

        set_fixed_axes(ax)

        ax.set_title(
            f"Radar Point Cloud - Frame {frame_number}",
            fontsize=17,
            pad=25
        )

        ax.legend(loc="upper right", fontsize=10)

        fig.canvas.draw_idle()

    # Previous button
    previous_ax = plt.axes([0.35, 0.04, 0.12, 0.06])
    previous_button = Button(
        previous_ax,
        "Previous"
    )

    # Next button
    next_ax = plt.axes([0.53, 0.04, 0.12, 0.06])
    next_button = Button(
        next_ax,
        "Next"
    )

    def previous_frame(event):

        current_index[0] = (
            current_index[0] - 1
        ) % len(frame_numbers)

        plot_frame(current_index[0])

    def next_frame(event):

        current_index[0] = (
            current_index[0] + 1
        ) % len(frame_numbers)

        plot_frame(current_index[0])

    previous_button.on_clicked(previous_frame)
    next_button.on_clicked(next_frame)

    plot_frame(0)

    plt.show()


# ============================================================
# 9. INDIVIDUAL TARGET TRAJECTORY PLOTS
# ============================================================

def plot_target_trajectories(data):

    if data.empty:
        print("No trajectories to display.")
        return

    for target_id, target_data in data.groupby("TargetID"):

        target_data = target_data.sort_values(
            "FrameIndex"
        ).reset_index(drop=True)

        if len(target_data) < 2:
            print(
                f"Target {target_id}: not enough observations "
                "to draw a trajectory."
            )
            continue

        x = target_data["X"].to_numpy()
        y = target_data["Y"].to_numpy()
        z = target_data["Z"].to_numpy()

        points = np.column_stack((x, y, z))

        # Build consecutive line segments
        segments = np.stack(
            [points[:-1], points[1:]],
            axis=1
        )

        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection="3d")

        # Dark blue at the beginning, fading towards white
        line = Line3DCollection(
            segments,
            cmap="Blues_r",
            linewidth=TRAJECTORY_LINE_WIDTH
        )

        line.set_array(
            np.linspace(0, 1, len(segments))
        )

        ax.add_collection3d(line)

        # Start and end points
        ax.scatter(
            x[0], y[0], z[0],
            color="green",
            marker="o",
            s=140,
            label="Start"
        )

        ax.scatter(
            x[-1], y[-1], z[-1],
            color="red",
            marker="X",
            s=160,
            label="End"
        )

        # Direction arrows along the trajectory
        arrow_spacing = max(
            1,
            len(points) // ARROW_COUNT
        )

        for i in range(
            0,
            len(points) - 1,
            arrow_spacing
        ):

            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            dz = z[i + 1] - z[i]

            progress = i / max(1, len(points) - 2)

            arrow_color = plt.cm.Blues_r(progress)

            ax.quiver(
                x[i],
                y[i],
                z[i],
                dx,
                dy,
                dz,
                color=arrow_color,
                arrow_length_ratio=0.5,
                linewidth=2.5
            )

        set_fixed_axes(ax)

        ax.set_title(
            f"Target {target_id} - 3D Trajectory",
            fontsize=17,
            pad=25
        )

        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.show()
        plt.close(fig)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\nGenerated CSV files:")
print("1. point_cloud_clusters.csv")
print("2. cluster_centroids.csv")
print("3. target_tracking.csv")
print("4. dwell_time.csv")

# Interactive frame viewer
show_frame_viewer(
    clustered_df,
    tracking
)

# Separate trajectory plot for each target
plot_target_trajectories(tracking)

print("\nRadar analysis completed.")