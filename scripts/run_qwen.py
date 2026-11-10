
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


# Make the src/ directory available for Python imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cvp.models.qwen_v1 import Qwen25VLEncoder


# Use exactly the same prompt for original and Mooney images.
PROMPT = """
Describe only what is visually observable in this image.

Focus on:
1. Whether a human face is visible.
2. Which facial regions are visually distinguishable:
   face outline, eyes, eyebrows, nose, mouth and ears.
3. The relative positions and visible shapes of these regions.
4. Whether the boundaries between facial regions are clear or ambiguous.
5. Any visible non-face or background content.

Do not infer identity, age, gender, ethnicity, personality or other
personal characteristics.

Do not assume that a missing detail is absent. Distinguish between
a feature that is not visible and one that cannot be clearly discerned.

Be concise. Explicitly mention uncertainty where appropriate.
""".strip()


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def index_images(folder: Path) -> dict:
    """Index image files using their filename stems."""

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


def reusable_output(
    output_path: Path,
    image_path: Path,
    model_id: str,
) -> dict | None:
    """Reuse an existing response if its configuration still matches."""

    if not output_path.is_file():
        return None

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))

        if (
            data.get("model_id") == model_id
            and data.get("image_file") == image_path.name
            and data.get("prompt") == PROMPT
        ):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return None


def generate_and_save(
    encoder: Qwen25VLEncoder,
    image_path: Path,
    output_path: Path,
    condition: str,
    max_new_tokens: int,
) -> dict:
    """Generate a response and save its metadata as JSON."""

    print(f"  Generating {condition} response: {image_path.name}")

    response = encoder.describe_image(
        image_path=image_path,
        prompt=PROMPT,
        max_new_tokens=max_new_tokens,
    )

    result = {
        "image_stem": image_path.stem,
        "image_file": image_path.name,
        "condition": condition,
        "model_id": encoder.model_id,
        "prompt": PROMPT,
        "response": response,
        "response_word_count": len(response.split()),
        "max_new_tokens": max_new_tokens,
        "device": str(encoder.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare Qwen2.5-VL responses for original and Mooney images."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of image pairs to process. Use 100 for the full dataset.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens generated per image.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional Hugging Face model ID override.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing outputs instead of reusing them.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    # Prevent accidental model downloads and inference on local CPU.
    if not torch.cuda.is_available():
        parser.error(
            "CUDA is not available. Run Qwen inference on a CUDA-enabled "
            "remote GPU environment."
        )

    pairs = get_image_pairs()

    print(f"\nMatched image pairs found: {len(pairs)}")

    selected_pairs = pairs[:args.limit]

    print(f"Image pairs selected: {len(selected_pairs)}")

    # Load the model only once for the entire dataset.
    if args.model_id:
        encoder = Qwen25VLEncoder(model_id=args.model_id)
    else:
        encoder = Qwen25VLEncoder()

    result_root = PROJECT_ROOT / "Result_2"

    original_output = result_root / "original"
    mooney_output = result_root / "mooney"

    original_output.mkdir(parents=True, exist_ok=True)
    mooney_output.mkdir(parents=True, exist_ok=True)

    comparison_results = []

    for index, (stem, original_path, mooney_path) in enumerate(
        selected_pairs, start=1
    ):
        print(f"\n[{index}/{len(selected_pairs)}] Processing {stem}")

        original_json = original_output / f"{stem}.json"
        mooney_json = mooney_output / f"{stem}.json"

        # Check whether previous outputs can safely be reused.
        original_result = None
        mooney_result = None

        if not args.overwrite:
            original_result = reusable_output(
                original_json, original_path, encoder.model_id
            )
            mooney_result = reusable_output(
                mooney_json, mooney_path, encoder.model_id
            )

        # Generate only outputs that are missing or outdated.
        if original_result is None:
            original_result = generate_and_save(
                encoder,
                original_path,
                original_json,
                "original",
                args.max_new_tokens,
            )
        else:
            print("  Reusing existing original response.")

        if mooney_result is None:
            mooney_result = generate_and_save(
                encoder,
                mooney_path,
                mooney_json,
                "mooney",
                args.max_new_tokens,
            )
        else:
            print("  Reusing existing Mooney response.")

        # One comparison row per matched image pair.
        comparison_results.append({
            "image_stem": stem,
            "original_file": original_path.name,
            "mooney_file": mooney_path.name,
            "model_id": encoder.model_id,
            "prompt": PROMPT,
            "original_response": original_result["response"],
            "mooney_response": mooney_result["response"],
            "original_word_count": original_result["response_word_count"],
            "mooney_word_count": mooney_result["response_word_count"],
            "original_json": str(original_json.relative_to(PROJECT_ROOT)),
            "mooney_json": str(mooney_json.relative_to(PROJECT_ROOT)),
            "device": str(encoder.device),
        })

        print("  Pair completed.")

    # Save the paired comparison results.
    csv_path = result_root / "qwen_comparison.csv"

    fieldnames = list(comparison_results[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_results)

    print("\nQwen2.5-VL processing completed.")
    print(f"Model: {encoder.model_id}")
    print(f"Pairs processed: {len(comparison_results)}")
    print(f"Original responses: {original_output}")
    print(f"Mooney responses: {mooney_output}")
    print(f"Comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
