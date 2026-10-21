from __future__ import annotations

import argparse
import csv
from pathlib import Path
from aerohash.paths import PROJECT_ROOT


KEEP_COLUMNS = [
    "station",
    "valid",
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "alti",
    "vsby",
    "wxcodes",
    "metar",
]


def clean_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        missing_columns = [column for column in KEEP_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=KEEP_COLUMNS)
            writer.writeheader()
            for row in reader:
                writer.writerow({column: row.get(column, "") for column in KEEP_COLUMNS})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep selected columns from ZBAA.csv")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data_processed/ZBAA.csv"), help="Input CSV file")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data_processed/ZBAA_clean.csv"), help="Output CSV file")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    clean_csv(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
