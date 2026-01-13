#!/usr/bin/env python3
"""
Remove duplicate labels from YOLO format annotation files.
Scans all .txt files in the dataset and removes duplicate bounding box annotations.
"""

from pathlib import Path
from tqdm import tqdm


def remove_duplicates_from_file(label_file: Path, dry_run=False):
    """
    Remove duplicate labels from a single file.

    Args:
        label_file: Path to the label file
        dry_run: If True, don't actually modify files

    Returns:
        Number of duplicates removed
    """
    if not label_file.exists():
        return 0

    with open(label_file, 'r') as f:
        lines = f.readlines()

    # Parse lines and track unique annotations
    seen = set()
    unique_lines = []
    duplicates = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Use the line itself as the key to detect exact duplicates
        # Format: class_id x_center y_center width height
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
        else:
            duplicates += 1

    if duplicates > 0 and not dry_run:
        with open(label_file, 'w') as f:
            for line in unique_lines:
                f.write(line + '\n')

    return duplicates


def clean_dataset_labels(data_root: Path, dry_run=False):
    """
    Clean all label files in the dataset.

    Args:
        data_root: Root path of the dataset (e.g., /home/hetao/graduate/data/MDMT-yolo)
        dry_run: If True, only report what would be changed
    """
    data_root = Path(data_root)

    # Find all label directories
    label_dirs = []
    for camera_dir in ['camera1', 'camera2']:
        for split in ['train', 'val', 'test']:
            label_path = data_root / camera_dir / 'labels' / split
            if label_path.exists():
                label_dirs.append(label_path)

    print(f"Found {len(label_dirs)} label directories:")
    for ld in label_dirs:
        print(f"  - {ld}")

    # Process all label files
    total_files = 0
    total_duplicates = 0
    files_with_duplicates = 0

    for label_dir in label_dirs:
        print(f"\nProcessing {label_dir}...")
        label_files = list(label_dir.glob('*.txt'))

        for label_file in tqdm(label_files):
            duplicates = remove_duplicates_from_file(label_file, dry_run)
            total_files += 1
            if duplicates > 0:
                files_with_duplicates += 1
                total_duplicates += duplicates
                if not dry_run:
                    tqdm.write(f"  {label_file.name}: {duplicates} duplicate(s) removed")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files with duplicates: {files_with_duplicates}")
    print(f"  Total duplicates removed: {total_duplicates}")
    if dry_run:
        print(f"\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove duplicate labels from YOLO dataset")
    parser.add_argument("--data-root", type=str, default="/home/hetao/graduate/data/MDMT-yolo",
                        help="Root path of the dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be changed, don't modify files")

    args = parser.parse_args()

    print("Cleaning duplicate labels from YOLO dataset...")
    print(f"Data root: {args.data_root}")
    if args.dry_run:
        print("DRY RUN mode - no files will be modified\n")

    clean_dataset_labels(args.data_root, args.dry_run)
