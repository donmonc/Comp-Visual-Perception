
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F


# Locate the project root and make src/ importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvp.models.dinov2 import Dinov2Encoder


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


def get_image_pairs(project_root: Path) -> list:
    """Match original and Mooney images using their filenames."""

    original = index_images(project_root / "Original")
    mooney = index_images(project_root / "Mooney")

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

    pairs = [
        (name, original[name], mooney[name])
        for name in sorted(original_names)
    ]

    return pairs


def save_embedding(path: Path, embedding):
    """Save an embedding as a NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(
        path,
        embedding.numpy().astype(np.float32)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings from original and Mooney images."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of image pairs to process. Use 100 for all images."
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    pairs = get_image_pairs(PROJECT_ROOT)

    print(f"\nOriginal/Mooney pairs found: {len(pairs)}")

    selected_pairs = pairs[:args.limit]

    print(f"Pairs selected for processing: {len(selected_pairs)}")

    # Initialise model once, not once per image.
    encoder = Dinov2Encoder()

    result_root = PROJECT_ROOT / "Result_1"

    original_output = result_root / "original"
    mooney_output = result_root / "mooney"

    original_output.mkdir(parents=True, exist_ok=True)
    mooney_output.mkdir(parents=True, exist_ok=True)

    comparison_results = []

    for index, (stem, original_path, mooney_path) in enumerate(
        selected_pairs, start=1
    ):

        print(f"\n[{index}/{len(selected_pairs)}] Processing {stem}")

        # Extract embeddings.
        original_embedding = encoder.encode_image(original_path)
        mooney_embedding = encoder.encode_image(mooney_path)

        # Compare the two image representations.
        similarity = F.cosine_similarity(
            original_embedding.unsqueeze(0),
            mooney_embedding.unsqueeze(0),
            dim=1
        ).item()

        # Save embeddings separately.
        save_embedding(
            original_output / f"{stem}.npy",
            original_embedding
        )

        save_embedding(
            mooney_output / f"{stem}.npy",
            mooney_embedding
        )

        comparison_results.append({
            "image_stem": stem,
            "original_file": original_path.name,
            "mooney_file": mooney_path.name,
            "cosine_similarity": round(similarity, 6),
            "embedding_dimension": encoder.embedding_dim,
            "model": encoder.model_id,
            "device": str(encoder.device)
        })

        print(f"Cosine similarity: {similarity:.6f}")

    # Write comparison results to CSV.
    csv_path = result_root / "dinov2_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(comparison_results)

    print("\nDINOv2 processing completed.")
    print(f"Original embeddings: {original_output}")
    print(f"Mooney embeddings: {mooney_output}")
    print(f"Comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
