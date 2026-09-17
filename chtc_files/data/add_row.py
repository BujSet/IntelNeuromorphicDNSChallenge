#!/usr/bin/env python3
"""Append a row to data.csv computed from a result directory's score CSVs.

Usage: python3 add_row.py <directory_name>

The directory (relative to this script, or an absolute path) must contain:
  si_snr_scores.csv              -> Validation SI-SNR avg
  musdb_snn_test_sisnr_scores.csv -> Test SI-SNR avg/median
  musdb_snn_test_sdr_scores.csv   -> Test SDR avg/median
Each CSV has a header row and a "vocals" score in the 3rd column.
"""
import csv
import statistics
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
DATA_CSV = DATA_DIR / "data.csv"


def load_scores(path):
    vals = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                vals.append(float(row[2]))
    return vals


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <directory_name>")

    dir_arg = sys.argv[1]
    folder = Path(dir_arg)
    if not folder.is_absolute():
        folder = DATA_DIR / dir_arg
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    name = folder.name

    with open(DATA_CSV) as f:
        lines = f.readlines()

    existing_names = {line.split(",")[0] for line in lines}
    if name in existing_names:
        sys.exit(f"Row for '{name}' already exists in {DATA_CSV}")

    # The file has a second, unrelated table below the main one, separated
    # by a blank line (first column empty). Insert the new row just before
    # that separator so it lands in the main table, not at the file's end.
    insert_at = len(lines)
    for i, line in enumerate(lines):
        if i > 0 and line.split(",")[0].strip() == "":
            insert_at = i
            break

    val = load_scores(folder / "si_snr_scores.csv")
    test_sisnr = load_scores(folder / "musdb_snn_test_sisnr_scores.csv")
    test_sdr = load_scores(folder / "musdb_snn_test_sdr_scores.csv")

    row = [
        name,
        f"{statistics.mean(val):.4f}",
        f"{statistics.mean(test_sisnr):.4f}",
        f"{statistics.mean(test_sdr):.4f}",
        f"{statistics.median(test_sisnr):.4f}",
        f"{statistics.median(test_sdr):.4f}",
    ]
    row_line = ",".join(row) + "\n"

    lines.insert(insert_at, row_line)
    with open(DATA_CSV, "w") as f:
        f.writelines(lines)

    print("Added row:", ",".join(row))


if __name__ == "__main__":
    main()
