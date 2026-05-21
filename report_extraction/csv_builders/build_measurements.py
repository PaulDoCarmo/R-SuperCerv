#!/usr/bin/env python3
"""Build a CSV with per-measurement rows from report JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def derive_id(file_path: Path) -> str:
    name = file_path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if name.endswith(".legacy"):
        name = name[: -len(".legacy")]
    if name.endswith("_report"):
        name = name[: -len("_report")]
    return name


def load_json_paths(json_folder: Path) -> Iterable[Path]:
    return sorted(json_folder.glob("*.json"))


def extract_measurement_value(measurements: object) -> tuple[float | None, str | None]:
    if not isinstance(measurements, dict) or not measurements:
        return None, None
    first_key = next(iter(measurements))
    entry = measurements.get(first_key)
    if not isinstance(entry, dict):
        return None, None
    value = entry.get("value")
    unit = entry.get("unit")
    return (float(value) if isinstance(value, (int, float)) else None, unit if isinstance(unit, str) else None)


def extract_control_point_positions(control_points: object) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    if not isinstance(control_points, list) or len(control_points) < 2:
        return None, None, None, None, None, None

    def get_position(point: object) -> tuple[float | None, float | None, float | None]:
        if not isinstance(point, dict):
            return None, None, None
        position = point.get("position_ras")
        if not isinstance(position, list) or len(position) < 3:
            return None, None, None
        x, y, z = position[0], position[1], position[2]
        return (
            float(x) if isinstance(x, (int, float)) else None,
            float(y) if isinstance(y, (int, float)) else None,
            float(z) if isinstance(z, (int, float)) else None,
        )

    x1, y1, z1 = get_position(control_points[0])
    x2, y2, z2 = get_position(control_points[1])
    return x1, y1, z1, x2, y2, z2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a CSV with per-measurement rows from report JSON files."
    )
    parser.add_argument("json_folder", help="Folder containing JSON report files")
    parser.add_argument("output_folder", help="Folder to write the CSV")
    parser.add_argument(
        "--output-name",
        default="measurements.csv",
        help="CSV filename to create in output_folder (default: measurements.csv)",
    )
    args = parser.parse_args()

    json_folder = Path(args.json_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / args.output_name

    fieldnames = [
        "ID",
        "measurement_index",
        "x1",
        "y1",
        "z1",
        "x2",
        "y2",
        "z2",
        "measurement",
        "unit_measure",
    ]

    rows = []
    for json_path in load_json_paths(json_folder):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        dictation = data.get("dictation", "")
        if not isinstance(dictation, str) or dictation == "":
            continue

        measurements = data.get("measurements")
        if not isinstance(measurements, list):
            continue

        measurement_index = 0
        has_valid_measurements = False
        for measurement in measurements:
            if not isinstance(measurement, dict):
                continue
            value, unit = extract_measurement_value(measurement.get("measurements"))
            if value is None or value == 0:
                continue

            measurement_index += 1
            has_valid_measurements = True

            x1, y1, z1, x2, y2, z2 = extract_control_point_positions(
                measurement.get("control_points")
            )

            rows.append(
                {
                    "ID": derive_id(json_path),
                    "measurement_index": measurement_index,
                    "x1": x1,
                    "y1": y1,
                    "z1": z1,
                    "x2": x2,
                    "y2": y2,
                    "z2": z2,
                    "measurement": value,
                    "unit_measure": unit,
                }
            )

        if not has_valid_measurements:
            rows.append(
                {
                    "ID": derive_id(json_path),
                    "measurement_index": "no_measure",
                    "x1": None,
                    "y1": None,
                    "z1": None,
                    "x2": None,
                    "y2": None,
                    "z2": None,
                    "measurement": None,
                    "unit_measure": None,
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
