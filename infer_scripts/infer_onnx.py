import argparse
from datetime import datetime
from glob import glob
from os.path import join, basename
from pathlib import Path
from time import time
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from lru import LRU
from tqdm import tqdm
from collections import OrderedDict

IMAGE_ENCODER_INPUT_SIZE = 512
EMBEDDINGS_CACHE_SIZE = 512


class SamEncoder:
    def __init__(self, model_path: str):
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(
            model_path, opt, providers=["OpenVINOExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image: np.ndarray):
        return self.session.run(
            None,
            {
                "image": image,
                "original_size": np.asarray(image.shape[:2], dtype=np.int16),
            },
        )[0]


class SamDecoder:
    def __init__(self, model_path: str):
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(
            model_path, opt, providers=["OpenVINOExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(
        self,
        image_embeddings: np.ndarray,
        boxes: np.ndarray,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ):
        mask = self.session.run(
            None, {"image_embeddings": image_embeddings, "boxes": boxes}
        )[0].squeeze((0, 1))
        mask = mask[: input_size[0], : input_size[1]]
        mask = cv2.resize(
            mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return mask


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


def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def get_bbox(mask: np.ndarray) -> np.ndarray:
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return np.array([x_min, y_min, x_max, y_max])


def infer_2D(img_npz_file: str, encoder: SamEncoder, decoder: SamDecoder):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, "r")

    img = npz_data["imgs"]
    if len(img.shape) < 3:
        img = np.repeat(img[..., None], 3, axis=-1)

    original_size = img.shape[:2]
    input_size = get_preprocess_shape(*original_size, IMAGE_ENCODER_INPUT_SIZE)

    boxes = npz_data["boxes"].astype(np.float32) / max(original_size)

    embedding = encoder(img)

    segs = np.zeros(original_size, dtype=np.uint16)
    for idx, box in enumerate(boxes, start=1):
        mask = decoder(embedding, box, input_size, original_size)
        segs[mask > 0] = idx

    np.savez_compressed(output_dir / npz_name, segs=segs)

    if save_overlay:
        visualize_output(
            img=npz_data["imgs"],
            boxes=npz_data["boxes"],
            segs=segs,
            save_file=(png_save_dir / npz_name).with_suffix(".png"),
        )


def infer_3D(img_npz_file: str, encoder: SamEncoder, decoder: SamDecoder):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, "r")
    img_3D = npz_data["imgs"]
    boxes_3D = npz_data["boxes"]

    original_size = img_3D.shape[1:3]
    input_size = get_preprocess_shape(*original_size, IMAGE_ENCODER_INPUT_SIZE)

    segs = np.zeros_like(img_3D, dtype=np.uint8)

    cached_embeddings = LRU(EMBEDDINGS_CACHE_SIZE)

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_i = np.zeros_like(segs, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        box_default = np.array([x_min, y_min, x_max, y_max]).astype(np.float32) / max(
            original_size
        )
        z_middle = (z_max + z_min) // 2

        # infer from middle slice to the z_max
        box_2D = box_default
        for z in range(int(z_middle), int(z_max)):
            if z in cached_embeddings:
                embedding = cached_embeddings[z]
            else:
                img_2d = img_3D[z, :, :]
                if len(img_2d.shape) == 2:
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
                else:
                    img_3c = img_2d
                embedding = encoder(img_3c)
                cached_embeddings[z] = embedding

            mask = decoder(embedding, box_2D, input_size, original_size)

            if np.max(mask) > 0:
                box_2D = get_bbox(mask).astype(np.float32) / max(original_size)
                segs_i[z, mask > 0] = 1
            else:
                box_2D = box_default

        # infer from middle slice to the z_min
        if np.max(segs_i[int(z_middle), :, :]) == 0:
            box_2D = box_default
        else:
            box_2D = get_bbox(segs_i[int(z_middle), :, :]).astype(np.float32) / max(
                original_size
            )

        for z in range(int(z_middle - 1), int(z_min - 1), -1):
            if z in cached_embeddings:
                embedding = cached_embeddings[z]
            else:
                img_2d = img_3D[z, :, :]
                if len(img_2d.shape) == 2:
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
                else:
                    img_3c = img_2d
                embedding = encoder(img_3c)
                cached_embeddings[z] = embedding

            mask = decoder(embedding, box_2D, input_size, original_size)

            if np.max(mask) > 0:
                box_2D = get_bbox(mask).astype(np.float32) / max(original_size)
                segs_i[z, mask > 0] = 1
            else:
                box_2D = box_default

        segs[segs_i > 0] = idx

    del cached_embeddings

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


def main():
    encoder = SamEncoder(
        "logs/export_onnx/runs/2024-03-28_23-14-59/onnx/encoder.optimized.onnx"
    )
    decoder = SamDecoder(
        "logs/export_onnx/runs/2024-03-28_23-14-59/onnx/decoder.optimized.onnx"
    )
    img_npz_files = sorted(glob(join(data_root, "*.npz"), recursive=True))
    efficiency = OrderedDict()
    efficiency["case"] = []
    efficiency["time"] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith("2D"):
            infer_2D(img_npz_file, encoder, decoder)
        else:
            infer_3D(img_npz_file, encoder, decoder)
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


if __name__ == "__main__":
    main()
