import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_generalized_dice, compute_surface_dice
from tqdm import tqdm


def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max() + 1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(
            float(
                compute_generalized_dice(
                    torch.tensor(seg_i).unsqueeze(0).unsqueeze(0),
                    torch.tensor(gt_i).unsqueeze(0).unsqueeze(0),
                )[0]
            )
        )
    return np.mean(dsc)


def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max() + 1):
        gt_i = torch.tensor(gt == i)
        seg_i = torch.tensor(seg == i)
        nsd.append(
            float(
                compute_surface_dice(
                    seg_i.unsqueeze(0).unsqueeze(0),
                    gt_i.unsqueeze(0).unsqueeze(0),
                    class_thresholds=[tolerance],
                    spacing=spacing,
                )
            )
        )
    return np.mean(nsd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segs",
        required=True,
        type=str,
        help="directory of segmentation output",
    )
    parser.add_argument(
        "--gts",
        required=True,
        type=str,
        help="directory of ground truth",
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="directory of original images",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="seg_metrics.csv",
        help="output csv file",
    )
    args = parser.parse_args()

    seg_metrics = OrderedDict()
    seg_metrics["case"] = []
    seg_metrics["dsc"] = []
    seg_metrics["nsd"] = []

    segs = sorted(Path(args.segs).glob("*.npz"))

    for seg_file in tqdm(segs):
        gt_file = Path(args.gts) / seg_file.name
        img_file = Path(args.imgs) / seg_file.name
        if not gt_file.exists() or not img_file.exists():
            continue

        npz_seg = np.load(seg_file, "r")
        npz_gt = np.load(gt_file, "r")

        seg = npz_seg["segs"]
        gt = npz_gt["gts"] if "gts" in npz_gt else npz_gt["segs"]
        dsc = compute_multi_class_dsc(gt, seg)

        if seg_file.name.startswith("3D"):
            npz_img = np.load(img_file, "r")
            spacing = npz_img["spacing"]
            nsd = compute_multi_class_nsd(gt, seg, spacing)
        else:
            spacing = [1.0, 1.0, 1.0]
            nsd = compute_multi_class_nsd(
                np.expand_dims(gt, -1), np.expand_dims(seg, -1), spacing
            )

        seg_metrics["case"].append(seg_file.name)
        seg_metrics["dsc"].append(np.round(dsc, 4))
        seg_metrics["nsd"].append(np.round(nsd, 4))

    dsc_np = np.array(seg_metrics["dsc"])
    nsd_np = np.array(seg_metrics["nsd"])
    avg_dsc = np.mean(dsc_np[~np.isnan(dsc_np)])
    avg_nsd = np.mean(nsd_np[~np.isnan(nsd_np)])
    seg_metrics["case"].append("average")
    seg_metrics["dsc"].append(avg_dsc)
    seg_metrics["nsd"].append(avg_nsd)

    df = pd.DataFrame(seg_metrics)
    df.to_csv(args.output_csv, index=False, na_rep="NaN")

    print("Average DSC: {:.4f}".format(avg_dsc))
    print("Average NSD: {:.4f}".format(avg_nsd))


if __name__ == "__main__":
    main()
