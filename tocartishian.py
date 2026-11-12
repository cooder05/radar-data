
import csv
import math

INPUT_FILE = "y-move\k_point_cloud.csv"
OUTPUT_FILE = "radar_xyz.csv"

with open(INPUT_FILE, "r", newline="", encoding="utf-8-sig") as infile:

    # Automatically detect delimiter
    sample = infile.read(2048)
    infile.seek(0)

    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    reader = csv.DictReader(infile, dialect=dialect)

    # Remove spaces from column names
    reader.fieldnames = [
        name.strip() for name in reader.fieldnames
    ]

    print("Detected columns:", reader.fieldnames)

    with open(OUTPUT_FILE, "w", newline="") as outfile:

        fields = [
            "FrameIndex", "PointIndex",
            "X", "Y", "Z",
            "Velocity", "SNR", "track_uid"
        ]

        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()

        for row in reader:

            # Clean column names and values
            row = {
                key.strip(): value.strip()
                for key, value in row.items()
                if key is not None
            }

            r = float(row["Range"])

            azimuth = math.radians(float(row["Azimuth"]))
            elevation = math.radians(float(row["Elevation"]))

            x = r * math.cos(elevation) * math.cos(azimuth)
            y = r * math.cos(elevation) * math.sin(azimuth)
            z = r * math.sin(elevation)

            writer.writerow({
                "FrameIndex": row["FrameIndex"],
                "PointIndex": row["PointIndex"],
                "X": round(x, 4),
                "Y": round(y, 4),
                "Z": round(z, 4),
                "Velocity": row["Velocity"],
                "SNR": row["SNR"],
                "track_uid": row["track_uid"]
            })

print("Conversion completed:", OUTPUT_FILE)
