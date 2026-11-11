
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# Make the src/ directory available for Python imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvp.models.depth_anything import DepthAnythingV2Encoder


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


def save_depth_outputs(
    depth_map: np.ndarray,
    output_stem: Path,
) -> None:
    """
    Save the numerical depth prediction and a visual PNG.

    The PNG is independently min-max normalised for visualisation.
    The NPY file retains the original floating-point predictions.
    """

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    # Save raw numerical prediction.
    np.save(
        output_stem.with_suffix(".npy"),
        depth_map.astype(np.float32),
    )

    # Create an 8-bit grayscale version for visual inspection.
    finite_values = depth_map[np.isfinite(depth_map)]

    if finite_values.size == 0:
        raise ValueError("Depth map contains no finite values.")

    minimum = finite_values.min()
    maximum = finite_values.max()

    if maximum > minimum:
        normalised = (depth_map - minimum) / (maximum - minimum)
    else:
        normalised = np.zeros_like(depth_map)

    normalised = np.nan_to_num(
        normalised,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    visual_map = (
        np.clip(normalised, 0.0, 1.0) * 255
    ).round().astype(np.uint8)

    Image.fromarray(visual_map, mode="L").save(
        output_stem.with_suffix(".png")
    )


def depth_statistics(depth_map: np.ndarray) -> dict:
    """Calculate descriptive statistics for a depth prediction."""

    finite_values = depth_map[np.isfinite(depth_map)]

    if finite_values.size == 0:
        raise ValueError("Depth map contains no finite values.")

    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Depth Anything V2 predictions for "
            "original and Mooney image pairs."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of image pairs to process. Use 100 for all pairs.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional Hugging Face model ID override.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    pairs = get_image_pairs()

    print(f"\nMatched image pairs found: {len(pairs)}")

    selected_pairs = pairs[:args.limit]

    print(f"Image pairs selected: {len(selected_pairs)}")

    # Load the model once for the entire dataset.
    if args.model_id:
        encoder = DepthAnythingV2Encoder(model_id=args.model_id)
    else:
        encoder = DepthAnythingV2Encoder()

    result_root = PROJECT_ROOT / "Result_3"

    original_output = result_root / "original"
    mooney_output = result_root / "mooney"

    original_output.mkdir(parents=True, exist_ok=True)
    mooney_output.mkdir(parents=True, exist_ok=True)

    comparison_results = []

    for index, (stem, original_path, mooney_path) in enumerate(
        selected_pairs, start=1
    ):
        print(f"\n[{index}/{len(selected_pairs)}] Processing {stem}")

        # Predict depth for the original image.
        original_depth = encoder.predict_depth(original_path)

        # Predict depth for the corresponding Mooney image.
        mooney_depth = encoder.predict_depth(mooney_path)

        # Save both numerical and visual outputs.
        save_depth_outputs(
            original_depth,
            original_output / stem,
        )

        save_depth_outputs(
            mooney_depth,
            mooney_output / stem,
        )

        # Calculate descriptive statistics.
        original_stats = depth_statistics(original_depth)
        mooney_stats = depth_statistics(mooney_depth)

        height, width = original_depth.shape

        comparison_results.append({
            "image_stem": stem,
            "original_file": original_path.name,
            "mooney_file": mooney_path.name,
            "model_id": encoder.model_id,
            "embedding_or_depth_dimensions": f"{height}x{width}",
            "original_depth_min": original_stats["min"],
            "original_depth_max": original_stats["max"],
            "original_depth_mean": original_stats["mean"],
            "original_depth_std": original_stats["std"],
            "mooney_depth_min": mooney_stats["min"],
            "mooney_depth_max": mooney_stats["max"],
            "mooney_depth_mean": mooney_stats["mean"],
            "mooney_depth_std": mooney_stats["std"],
            "device": str(encoder.device),
        })

        print(
            "  Original depth shape:",
            original_depth.shape,
        )
        print(
            "  Mooney depth shape:",
            mooney_depth.shape,
        )
        print("  Pair completed.")

    # Save paired comparison metadata.
    csv_path = result_root / "depth_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(comparison_results)

    print("\nDepth Anything V2 processing completed.")
    print(f"Model: {encoder.model_id}")
    print(f"Pairs processed: {len(comparison_results)}")
    print(f"Original outputs: {original_output}")
    print(f"Mooney outputs: {mooney_output}")
    print(f"Comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
