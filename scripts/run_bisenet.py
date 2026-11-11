
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# Make the src/ directory available for Python imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvp.models.bisenet import (
    BiSeNetFaceParser,
    CLASS_NAMES,
)


# A fixed RGB colour for each of the 19 segmentation classes.
# The same colours are used for original and Mooney images.
CLASS_PALETTE = np.array(
    [
        (0, 0, 0),          # 0  Background
        (255, 191, 128),    # 1  Skin
        (255, 0, 0),        # 2  Left eyebrow
        (255, 80, 80),      # 3  Right eyebrow
        (0, 0, 255),        # 4  Left eye
        (80, 80, 255),      # 5  Right eye
        (0, 255, 255),      # 6  Eyeglasses
        (0, 128, 255),      # 7  Left ear
        (80, 180, 255),     # 8  Right ear
        (255, 255, 0),      # 9  Earring
        (255, 128, 0),      # 10 Nose
        (128, 0, 128),      # 11 Mouth
        (255, 0, 255),      # 12 Upper lip
        (192, 0, 192),      # 13 Lower lip
        (0, 128, 0),        # 14 Neck
        (0, 255, 0),        # 15 Necklace
        (128, 128, 128),    # 16 Cloth
        (128, 64, 0),       # 17 Hair
        (255, 255, 255),    # 18 Hat
    ],
    dtype=np.uint8,
)


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def index_images(folder: Path) -> dict:
    """Index images by filename stem."""

    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    images = {}

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError(f"Duplicate image stem: {path.stem}")

            images[path.stem] = path

    if not images:
        raise ValueError(f"No images found in {folder}")

    return images


def get_image_pairs() -> list:
    """Match original and Mooney images using their filenames."""

    original = index_images(PROJECT_ROOT / "Original")
    mooney = index_images(PROJECT_ROOT / "Mooney")

    original_names = set(original)
    mooney_names = set(mooney)

    original_only = sorted(original_names - mooney_names)
    mooney_only = sorted(mooney_names - original_names)

    if original_only or mooney_only:
        raise ValueError(
            "Original/Mooney filename mismatch.\n"
            f"Only in Original: {original_only[:10]}\n"
            f"Only in Mooney: {mooney_only[:10]}"
        )

    return [
        (stem, original[stem], mooney[stem])
        for stem in sorted(original_names)
    ]


def save_segmentation_outputs(
    image_path: Path,
    mask: np.ndarray,
    output_directory: Path,
) -> dict:
    """Save raw mask, class-ID PNG, colour mask and image overlay."""

    output_directory.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    # Check that the predicted mask matches the input dimensions.
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"image shape {(image.height, image.width)}"
        )

    if mask.min() < 0 or mask.max() >= len(CLASS_NAMES):
        raise ValueError("Predicted mask contains an invalid class ID.")

    # 1. Save raw integer class IDs.
    mask_directory = output_directory / "masks"
    mask_directory.mkdir(parents=True, exist_ok=True)

    np.save(
        mask_directory / f"{image_path.stem}.npy",
        mask.astype(np.uint8),
    )

    # 2. Save a grayscale PNG containing class IDs 0-18.
    class_mask_directory = output_directory / "class_masks"
    class_mask_directory.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask.astype(np.uint8)).save(
        class_mask_directory / f"{image_path.stem}.png"
    )

    # 3. Convert class IDs to consistent RGB colours.
    color_mask = CLASS_PALETTE[mask]

    color_directory = output_directory / "color_masks"
    color_directory.mkdir(parents=True, exist_ok=True)

    Image.fromarray(color_mask).save(
        color_directory / f"{image_path.stem}.png"
    )

    # 4. Overlay predicted facial regions on the input image.
    overlay_directory = output_directory / "overlays"
    overlay_directory.mkdir(parents=True, exist_ok=True)

    # Keep background pixels transparent in the overlay.
    alpha = np.where(mask == 0, 0, 115).astype(np.uint8)

    rgba_mask = np.dstack((color_mask, alpha))

    overlay = Image.alpha_composite(
        image.convert("RGBA"),
        Image.fromarray(rgba_mask),
    ).convert("RGB")

    overlay.save(
        overlay_directory / f"{image_path.stem}.png"
    )

    # 5. Calculate the percentage of pixels belonging to each class.
    class_counts = np.bincount(
        mask.ravel(),
        minlength=len(CLASS_NAMES),
    )

    total_pixels = mask.size

    class_percentages = {
        class_name: round(
            float(class_counts[class_id] / total_pixels * 100),
            4,
        )
        for class_id, class_name in enumerate(CLASS_NAMES)
    }

    return class_percentages


def calculate_mask_agreement(
    original_mask: np.ndarray,
    mooney_mask: np.ndarray,
) -> dict:
    """Compare corresponding predicted masks."""

    if original_mask.shape != mooney_mask.shape:
        raise ValueError(
            "Original and Mooney masks have different dimensions."
        )

    # Percentage of pixels receiving exactly the same predicted class.
    pixel_agreement = float(
        np.mean(original_mask == mooney_mask) * 100
    )

    # Compare predicted face-region coverage, excluding background.
    original_foreground = original_mask != 0
    mooney_foreground = mooney_mask != 0

    intersection = np.logical_and(
        original_foreground,
        mooney_foreground,
    ).sum()

    union = np.logical_or(
        original_foreground,
        mooney_foreground,
    ).sum()

    foreground_iou = (
        float(intersection / union)
        if union > 0
        else 1.0
    )

    return {
        "pixel_label_agreement_pct": round(pixel_agreement, 4),
        "foreground_mask_iou": round(foreground_iou, 6),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate BiSeNet facial segmentation masks "
            "for original and Mooney images."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of image pairs to process. Use 100 for all pairs.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    pairs = get_image_pairs()

    print(f"\nMatched image pairs found: {len(pairs)}")

    selected_pairs = pairs[:args.limit]

    print(f"Image pairs selected: {len(selected_pairs)}")

    # Load the model once for the entire dataset.
    parser_model = BiSeNetFaceParser()

    result_root = PROJECT_ROOT / "Result_4"

    original_output = result_root / "original"
    mooney_output = result_root / "mooney"

    original_output.mkdir(parents=True, exist_ok=True)
    mooney_output.mkdir(parents=True, exist_ok=True)

    comparison_results = []

    for index, (stem, original_path, mooney_path) in enumerate(
        selected_pairs,
        start=1,
    ):
        print(f"\n[{index}/{len(selected_pairs)}] Processing {stem}")

        # Predict masks for both conditions.
        original_mask = parser_model.predict_mask(original_path)
        mooney_mask = parser_model.predict_mask(mooney_path)

        # Save all visual and numerical outputs.
        original_percentages = save_segmentation_outputs(
            original_path,
            original_mask,
            original_output,
        )

        mooney_percentages = save_segmentation_outputs(
            mooney_path,
            mooney_mask,
            mooney_output,
        )

        # Compare the two predicted masks.
        agreement = calculate_mask_agreement(
            original_mask,
            mooney_mask,
        )

        height, width = original_mask.shape

        row = {
            "image_stem": stem,
            "original_file": original_path.name,
            "mooney_file": mooney_path.name,
            "model_id": parser_model.model_id,
            "checkpoint": parser_model.checkpoint_path.name,
            "mask_dimensions": f"{height}x{width}",
            "device": str(parser_model.device),
            **agreement,
        }

        # Include per-class pixel coverage for both conditions.
        for class_name in CLASS_NAMES:
            row[f"original_{class_name}_pct"] = (
                original_percentages[class_name]
            )

            row[f"mooney_{class_name}_pct"] = (
                mooney_percentages[class_name]
            )

        comparison_results.append(row)

        print(
            "  Original classes present:",
            int(np.unique(original_mask).size),
        )

        print(
            "  Mooney classes present:",
            int(np.unique(mooney_mask).size),
        )

        print(
            f"  Pixel label agreement: "
            f"{agreement['pixel_label_agreement_pct']:.2f}%"
        )

        print("  Pair completed.")

    # Save paired comparison results.
    csv_path = result_root / "bisenet_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(comparison_results)

    print("\nBiSeNet processing completed.")
    print(f"Model: {parser_model.model_id}")
    print(f"Pairs processed: {len(comparison_results)}")
    print(f"Original outputs: {original_output}")
    print(f"Mooney outputs: {mooney_output}")
    print(f"Comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
