"""Convert raw PLC-CECT data layout into the folder structure expected by TransLiver.

The script expects the raw dataset to look like the structure shown in the README screenshots:
- ``ct_files``: four NIfTI volumes per patient (C1, C2, C3, P phases)
- ``mask_files``: lesion masks that align with the CT volumes
- ``liver_mask_files``: liver masks (optional; copied if present)
- ``patient_data.csv``: metadata describing each scan/mask pair

It creates phase-specific folders named ``artery``, ``venous``, ``delayed``, and ``plain``
plus their ``*_label`` and ``*_liver`` counterparts so that the existing registration and
classification preprocessing scripts can run directly.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import Dict, Iterable, Mapping, MutableMapping

import nibabel as nib
import numpy as np

PHASE_MAP_DEFAULT = {"C1": "artery", "C2": "venous", "C3": "delayed", "P": "plain"}
CANCER_CLASS_MAP = {
    "HCC": 1,
    "ICC": 2,
    "cHCC-CCA": 3,
    "NON-LIVER CANCER": 4,
    "NON-LIVER": 4,
    "NONLIVER": 4,
    "OTHER": 4,
}


def parse_phase_map(phase_map: str | None) -> Dict[str, str]:
    """Parse a CLI phase mapping string.

    The string should look like ``"C1:artery,C2:venous"``. Keys are compared case-insensitively.
    """

    if not phase_map:
        return PHASE_MAP_DEFAULT.copy()
    mapping: Dict[str, str] = {}
    for item in phase_map.split(","):
        raw = item.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Invalid phase mapping segment '{raw}', expected KEY:VALUE format")
        src, dst = raw.split(":", 1)
        mapping[src.strip().upper()] = dst.strip()
    return mapping


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def load_metadata(csv_path: str) -> Iterable[Mapping[str, str]]:
    with open(csv_path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"patient", "phase", "cancer_type", "ct_path", "mask_path"}
        missing = required - set(col.lower() for col in reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"patient_data.csv is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            normalized = {k.lower(): v.strip() for k, v in row.items() if k}
            yield normalized


def copy_volume(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def collect_lesion_ids(mask_path: str) -> Iterable[int]:
    data = nib.load(mask_path).get_fdata()
    unique_vals = np.unique(data.astype(int))
    return [val for val in unique_vals if val > 0]


def update_lesion_classes(
    lesion_classes: MutableMapping[str, Dict[str, Dict[str, Dict[str, int]]]],
    phase: str,
    patient: str,
    lesion_ids: Iterable[int],
    class_id: int,
) -> None:
    phase_dict = lesion_classes.setdefault(phase, {})
    patient_dict = phase_dict.setdefault(patient, {})
    for lid in lesion_ids:
        patient_dict[str(lid)] = {"class": class_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Path to the raw PLC-CECT folder containing ct_files, mask_files, and patient_data.csv",
    )
    parser.add_argument(
        "--dest-root",
        required=True,
        help="Output directory for the phase-organized data",
    )
    parser.add_argument(
        "--phase-map",
        help=(
            "Mapping from CSV phase codes to TransLiver names (e.g., 'C1:artery,C2:venous,C3:delayed,P:plain'). "
            "Defaults to the PLC-CECT convention."
        ),
    )
    parser.add_argument(
        "--save-lesion-classes",
        action="store_true",
        help="Also generate lesion_classes.npy under dest-root using cancer_type labels",
    )
    args = parser.parse_args()

    source_root = os.path.abspath(args.source_root)
    dest_root = os.path.abspath(args.dest_root)
    phase_map = parse_phase_map(args.phase_map)

    metadata_path = os.path.join(source_root, "patient_data.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find patient_data.csv at {metadata_path}")

    lesion_classes: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    for entry in load_metadata(metadata_path):
        patient = entry["patient"]
        phase_code = entry["phase"].upper()
        if phase_code not in phase_map:
            raise KeyError(f"Unexpected phase code '{phase_code}'. Please extend --phase-map accordingly.")
        phase_name = phase_map[phase_code]

        ct_src = resolve_path(source_root, entry["ct_path"])
        mask_src = resolve_path(source_root, entry["mask_path"])
        liver_mask_src = entry.get("liver_mask_path")
        if liver_mask_src:
            liver_mask_src = resolve_path(source_root, liver_mask_src)

        ct_dst = os.path.join(dest_root, phase_name, f"{patient}.nii.gz")
        mask_dst = os.path.join(dest_root, f"{phase_name}_label", f"{patient}.nii.gz")
        liver_dst = os.path.join(dest_root, f"{phase_name}_liver", f"{patient}.nii.gz")

        copy_volume(ct_src, ct_dst)
        copy_volume(mask_src, mask_dst)
        if liver_mask_src and os.path.exists(liver_mask_src):
            copy_volume(liver_mask_src, liver_dst)

        if args.save_lesion_classes:
            cancer_type = entry["cancer_type"].upper()
            class_id = CANCER_CLASS_MAP.get(cancer_type)
            if class_id is None:
                raise KeyError(
                    f"Unmapped cancer_type '{cancer_type}'. Please update CANCER_CLASS_MAP for your dataset."
                )
            lesion_ids = collect_lesion_ids(mask_src)
            update_lesion_classes(lesion_classes, phase_name, patient, lesion_ids, class_id)

    if args.save_lesion_classes:
        target = os.path.join(dest_root, "lesion_classes.npy")
        np.save(target, lesion_classes)
        print(f"Saved lesion class mapping to {target}")

    print("Data reorganized successfully. Output root:", dest_root)


if __name__ == "__main__":
    main()
