import argparse
import sys
from datetime import datetime
from glob import glob
from os.path import join, basename
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))


class ResizeLongestSide(torch.nn.Module):
    def __init__(
        self,
        long_side_length: int,
        interpolation: str,
    ) -> None:
        super().__init__()
        self.long_side_length = long_side_length
        self.interpolation = interpolation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        oldh, oldw = image.shape[-2:]
        if max(oldh, oldw) == self.long_side_length:
            return image
        newh, neww = self.get_preprocess_shape(oldh, oldw, self.long_side_length)
        return F.interpolate(
            image, (newh, neww), mode=self.interpolation, align_corners=False
        )

    @staticmethod
    def get_preprocess_shape(
        oldh: int,
        oldw: int,
        long_side_length: int,
    ) -> Tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class MinMaxScale(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) >= 3 and image.shape[-3] == 3
        min_val = image.amin((-3, -2, -1), keepdim=True)
        max_val = image.amax((-3, -2, -1), keepdim=True)
        return (image - min_val) / torch.clip(max_val - min_val, min=1e-8, max=None)


class PadToSquare(torch.nn.Module):
    def __init__(self, target_size: int) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        return F.pad(image, (0, self.target_size - w, 0, self.target_size - h), value=0)


def get_bbox(mask: np.ndarray) -> np.ndarray:
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes


def resize_box(
    box: np.ndarray,
    original_size: Tuple[int, int],
    prompt_encoder_input_size: int,
) -> np.ndarray:
    new_box = np.zeros_like(box)
    ratio = prompt_encoder_input_size / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


def get_image_transform(
    long_side_length: int,
    min_max_scale: bool = True,
    normalize: bool = False,
    pixel_mean: Optional[List[float]] = None,
    pixel_std: Optional[List[float]] = None,
    interpolation: str = "bilinear",
) -> transforms.Transform:
    tsfm = [
        ResizeLongestSide(long_side_length, interpolation),
        transforms.ToDtype(dtype=torch.float32, scale=False),
    ]
    if min_max_scale:
        tsfm.append(MinMaxScale())
    if normalize:
        tsfm.append(transforms.Normalize(pixel_mean, pixel_std))
    tsfm.append(PadToSquare(long_side_length))
    return transforms.Compose(tsfm)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    default="/workspace/inputs/",
    help="root directory of the data",
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="/workspace/outputs/",
    help="directory to save the prediction",
)

parser.add_argument(
    "--model",
    type=str,
    default="model.pth",
    help="path to model checkpoint",
)

parser.add_argument(
    "--save_overlay",
    default=False,
    action="store_true",
    help="whether to save the overlay image",
)

parser.add_argument(
    "-png_save_dir",
    type=str,
    default="./overlay",
    help="directory to save the overlay image",
)

args = parser.parse_args()

data_root = args.input_dir
save_overlay = args.save_overlay
output_dir = Path(args.output_dir)
png_save_dir = Path(args.png_save_dir)
output_dir.mkdir(parents=True, exist_ok=True)
png_save_dir.mkdir(parents=True, exist_ok=True)


if save_overlay:
    from src.utils.visualize import visualize_output


transform_image = get_image_transform(long_side_length=512)


def infer_2D(img_npz_file, model):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, "r")

    img = npz_data["imgs"]
    if len(img.shape) < 3:
        img = np.repeat(img[..., None], 3, axis=-1)  # (H, W, 3)

    original_size = img.shape[:2]
    new_size = ResizeLongestSide.get_preprocess_shape(
        original_size[0], original_size[1], 512
    )
    tsfm_img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.uint8)
    tsfm_img = transform_image(tsfm_img.unsqueeze(0))

    boxes = npz_data["boxes"]
    tsfm_boxes = []
    for box in boxes:
        box = resize_box(
            box,
            original_size=original_size,
            prompt_encoder_input_size=512,
        )
        tsfm_boxes.append(box)
    tsfm_boxes = torch.tensor(np.array(tsfm_boxes), dtype=torch.float32)

    image_embedding = model.image_encoder(tsfm_img)

    segs = np.zeros(original_size, dtype=np.uint16)
    for idx, box in enumerate(tsfm_boxes, start=1):
        mask, _ = model.prompt_and_decoder(image_embedding, box.unsqueeze(0))
        mask = model.postprocess_masks(mask, new_size, original_size)
        mask = mask.squeeze((0, 1)).cpu().numpy()
        segs[mask > 0] = idx

    np.savez_compressed(output_dir / npz_name, segs=segs)

    if save_overlay:
        visualize_output(
            img=npz_data["imgs"],
            boxes=npz_data["boxes"],
            segs=segs,
            save_file=(png_save_dir / npz_name).with_suffix(".png"),
        )


def infer_3D(img_npz_file: str, model):
    prompt_encoder_input_size = model.prompt_encoder.input_image_size[0]

    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, "r")
    img_3D = npz_data["imgs"]
    boxes = npz_data["boxes"]

    if len(img_3D.shape) == 3:
        # gray: (D, H, W) -> (D, 3, H, W)
        tsfm_img_3D = np.repeat(img_3D[:, None, ...], 3, axis=1)
    else:
        # rbg: (D, H, W, 3) -> (D, 3, H, W)
        tsfm_img_3D = np.transpose(img_3D, (0, 3, 1, 2))

    original_size = img_3D.shape[-2:]
    new_size = ResizeLongestSide.get_preprocess_shape(
        original_size[0], original_size[1], 512
    )
    tsfm_img_3D = transform_image(torch.tensor(tsfm_img_3D, dtype=torch.uint8))

    tsfm_boxes = []
    for box3D in boxes:
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        box2D = np.array([x_min, y_min, x_max, y_max])
        box2D = resize_box(
            box2D,
            original_size=original_size,
            prompt_encoder_input_size=prompt_encoder_input_size,
        )
        box3D = np.array([box2D[0], box2D[1], z_min, box2D[2], box2D[3], z_max])
        tsfm_boxes.append(box3D)
    tsfm_boxes = torch.tensor(np.array(tsfm_boxes), dtype=torch.float32)

    segs = np.zeros_like(img_3D, dtype=np.uint8)

    for idx, box3D in enumerate(tsfm_boxes, start=1):
        segs_i = np.zeros_like(segs, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        box_default = np.array([x_min, y_min, x_max, y_max])
        z_middle = (z_max + z_min) // 2

        # infer from middle slice to the z_max
        box_2D = box_default
        for z in range(int(z_middle), int(z_max)):
            img_2d = tsfm_img_3D[z, :, :, :].unsqueeze(0)  # (1, 3, H, W)
            image_embedding = model.image_encoder(img_2d)  # (1, 256, 64, 64)

            box_torch = torch.as_tensor(box_2D[None, ...], dtype=torch.float)  # (B, 4)
            mask, _ = model.prompt_and_decoder(image_embedding, box_torch)
            mask = model.postprocess_masks(mask, new_size, original_size)
            mask = mask.squeeze().cpu().numpy()
            if np.max(mask) > 0:
                box_2D = get_bbox(mask)
                box_2D = resize_box(
                    box=box_2D,
                    original_size=original_size,
                    prompt_encoder_input_size=prompt_encoder_input_size,
                )
                segs_i[z, mask > 0] = 1
            else:
                box_2D = box_default

        # infer from middle slice to the z_min
        if np.max(segs_i[int(z_middle), :, :]) == 0:
            box_2D = box_default
        else:
            box_2D = get_bbox(segs_i[int(z_middle), :, :])
            box_2D = resize_box(
                box=box_2D,
                original_size=original_size,
                prompt_encoder_input_size=prompt_encoder_input_size,
            )

        for z in range(int(z_middle - 1), int(z_min - 1), -1):
            img_2d = tsfm_img_3D[z, :, :, :].unsqueeze(0)  # (1, 3, H, W)
            image_embedding = model.image_encoder(img_2d)  # (1, 256, 64, 64)

            box_torch = torch.as_tensor(box_2D[None, ...], dtype=torch.float)  # (B, 4)
            mask, _ = model.prompt_and_decoder(image_embedding, box_torch)
            mask = model.postprocess_masks(mask, new_size, original_size)
            mask = mask.squeeze().cpu().numpy()
            if np.max(mask) > 0:
                box_2D = get_bbox(mask)
                box_2D = resize_box(
                    box=box_2D,
                    original_size=original_size,
                    prompt_encoder_input_size=prompt_encoder_input_size,
                )
                segs_i[z, mask > 0] = 1
            else:
                box_2D = box_default

        segs[segs_i > 0] = idx

    np.savez_compressed(output_dir / npz_name, segs=segs)

    # visualize image, mask and bounding box
    if save_overlay:
        z = segs.shape[0] // 2
        visualize_output(
            img=npz_data["imgs"][z],
            boxes=npz_data["boxes"][:, [0, 1, 3, 4]],
            segs=segs[z],
            save_file=(png_save_dir / npz_name).with_suffix(".png"),
        )


if __name__ == "__main__":
    model = torch.load(args.model).to("cpu")

    img_npz_files = sorted(glob(join(data_root, "*.npz"), recursive=True))
    efficiency = OrderedDict()
    efficiency["case"] = []
    efficiency["time"] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith("2D"):
            infer_2D(img_npz_file, model)
        else:
            infer_3D(img_npz_file, model)
        end_time = time()
        efficiency["case"].append(basename(img_npz_file))
        efficiency["time"].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            current_time,
            "file name:",
            basename(img_npz_file),
            "time cost:",
            np.round(end_time - start_time, 4),
        )
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(output_dir / "efficiency.csv", index=False)
