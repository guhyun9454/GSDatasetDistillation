#!/usr/bin/env python3
"""Extract the 6 ImageNet benchmark subsets into GSDD's clean 10-class layout.

GSDD's loaders (`DC/utils.py`, `DM/utils.py`) hardcode `num_classes=10` and
`ImageFolder(data_path/{train,val})`, expecting EXACTLY the 10 subset classes.
The DISC/DDiF tarballs in `/nas2/data/guhyun9454/ImageNet/*.tar.gz` instead store
the data in DISC "full-train" layout (the full sorted ImageNet folder set, with
the subset selected later by an index filter). This script bridges the two: for
each subset it resolves the 10 benchmark synsets and links/extracts only those
into `store/dataset/<subset>/{train,val}`.

Subset membership is defined the SAME way as GSDD's `utils/generate_subset.py`:
each subset is a list of INDICES into the sorted list of all ImageNet class
folders (NOT WNIDs). So the resolved synset for index `i` is
`sorted(os.listdir(<imagenet>/train))[i]`. This mirrors `create_symlinks()`
exactly, so the resulting folders match what GSDD expects.

PLAN-ONLY NOTE (2026-05-23): authored during the Seraph ceph migration. Do not
run until the user confirms the final `--imagenet` path on ceph. The default
paths below are placeholders.

Class-membership gate
---------------------
The risk (flagged in the spec) is that a tarball does NOT carry the full sorted
ImageNet folder set, in which case index `i` resolves to the wrong synset. To
guard against silent mis-selection this script:
  1. requires the source `train/` to expose enough folders that every index is
     in range, and (by default) warns unless exactly 1000 folders are present;
  2. optionally cross-checks the resolved WNIDs against a trusted reference list
     (`--wnid-list`, one WNID per line in canonical sorted order) and hard-fails
     on any mismatch;
  3. verifies each resolved class folder is non-empty in both train and val;
  4. prints the resolved WNID + image counts per class so the user can eyeball
     them against the DD-benchmark definition before trusting the run.

Usage (on the cluster, after ceph paths are final):
    python scripts/seraph/prepare_subsets.py \
        --imagenet /ceph_data/guhyun9454/ImageNet/full \
        --out      store/dataset \
        --mode     symlink \
        --subsets  all
"""
import argparse
import os
import sys

# Reuse the canonical index lists from GSDD's own generator so the two never drift.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.generate_subset import Config  # noqa: E402

SUBSETS = list(Config.dict.keys())  # imagenette, imagewoof, imagefruit, imageyellow, imagemeow, imagesquawk


def list_sorted_classes(train_dir):
    if not os.path.isdir(train_dir):
        sys.exit(f"ERROR: train dir not found: {train_dir}")
    classes = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    return classes


def count_images(d):
    if not os.path.isdir(d):
        return 0
    return sum(
        1
        for f in os.listdir(d)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )


def load_reference_wnids(path):
    with open(path) as f:
        wnids = [ln.strip() for ln in f if ln.strip()]
    return wnids


def resolve_subset(subset, all_classes, ref_wnids):
    indices = Config.dict[subset]
    if max(indices) >= len(all_classes):
        sys.exit(
            f"ERROR [{subset}]: index {max(indices)} out of range for "
            f"{len(all_classes)} source classes. The source `train/` likely does "
            f"NOT carry the full sorted ImageNet folder set, so index->synset "
            f"resolution is invalid. Stage the full-class layout first."
        )
    resolved = [all_classes[i] for i in indices]
    if len(set(resolved)) != 10:
        sys.exit(f"ERROR [{subset}]: resolved {len(set(resolved))} unique classes, expected 10.")

    if ref_wnids is not None:
        expected = [ref_wnids[i] for i in indices]
        mismatch = [(i, r, e) for i, r, e in zip(indices, resolved, expected) if r != e]
        if mismatch:
            lines = "\n".join(f"    idx {i}: resolved {r} != reference {e}" for i, r, e in mismatch)
            sys.exit(f"ERROR [{subset}]: WNID mismatch vs --wnid-list:\n{lines}")
    return list(zip(indices, resolved))


def link_or_copy(src, dst, mode):
    if os.path.lexists(dst):
        return
    if mode == "symlink":
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    elif mode == "copy":
        import shutil

        shutil.copytree(src, dst)
    else:
        sys.exit(f"ERROR: unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagenet", required=True, help="Full-class ImageNet root containing train/ and val/.")
    ap.add_argument("--out", default="store/dataset", help="Output root for store/dataset/<subset>/{train,val}.")
    ap.add_argument("--subsets", default="all", help="'all' or comma-separated subset names.")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--wnid-list", default=os.path.join(os.path.dirname(__file__), "imagenet1k_sorted_wnids.txt"),
                    help="Trusted sorted WNID list for the membership gate (defaults to the bundled "
                         "imagenet1k_sorted_wnids.txt). Pass '' to disable the WNID cross-check.")
    ap.add_argument("--expect-classes", type=int, default=1000, help="Warn unless the source has this many classes.")
    args = ap.parse_args()

    targets = SUBSETS if args.subsets == "all" else [s.strip() for s in args.subsets.split(",")]
    unknown = [s for s in targets if s not in Config.dict]
    if unknown:
        sys.exit(f"ERROR: unknown subset(s) {unknown}. Valid: {SUBSETS}")

    train_dir = os.path.join(args.imagenet, "train")
    val_dir = os.path.join(args.imagenet, "val")
    all_classes = list_sorted_classes(train_dir)
    print(f"Source: {args.imagenet}  ({len(all_classes)} sorted train classes)")
    if len(all_classes) != args.expect_classes:
        print(
            f"WARNING: expected {args.expect_classes} source classes, found {len(all_classes)}. "
            f"Index->synset resolution is only valid if this is the FULL sorted ImageNet "
            f"folder set (the DISC full-train layout). Verify the resolved WNIDs below."
        )

    ref_wnids = load_reference_wnids(args.wnid_list) if args.wnid_list else None
    if ref_wnids is None:
        print("NOTE: no --wnid-list given; running STRUCTURAL gate only (10 non-empty dirs). "
              "Eyeball the resolved WNIDs against the DD-benchmark definition before trusting results.")

    all_ok = True
    for subset in targets:
        print(f"\n=== {subset} ===")
        resolved = resolve_subset(subset, all_classes, ref_wnids)
        out_train = os.path.join(args.out, subset, "train")
        out_val = os.path.join(args.out, subset, "val")
        os.makedirs(out_train, exist_ok=True)
        os.makedirs(out_val, exist_ok=True)

        for idx, wnid in resolved:
            src_t = os.path.join(train_dir, wnid)
            src_v = os.path.join(val_dir, wnid)
            n_train = count_images(src_t)
            n_val = count_images(src_v)
            status = "OK"
            if n_train == 0:
                status, all_ok = "EMPTY-TRAIN", False
            elif not os.path.isdir(src_v) or n_val == 0:
                status, all_ok = "EMPTY-VAL", False
            print(f"  idx {idx:4d} -> {wnid}  train={n_train:5d} val={n_val:4d}  [{status}]")
            if status == "OK":
                link_or_copy(src_t, os.path.join(out_train, wnid), args.mode)
                link_or_copy(src_v, os.path.join(out_val, wnid), args.mode)

        n_out = len([d for d in os.listdir(out_train) if os.path.isdir(os.path.join(out_train, d))])
        if n_out != 10:
            print(f"  GATE FAIL: {subset} has {n_out} class dirs in train (expected 10).")
            all_ok = False
        else:
            print(f"  GATE PASS: {subset} -> 10 classes at {os.path.join(args.out, subset)}")

    if not all_ok:
        sys.exit("\nONE OR MORE SUBSETS FAILED THE CLASS-MEMBERSHIP GATE. See above.")
    print("\nAll requested subsets passed the class-membership gate.")


if __name__ == "__main__":
    main()
