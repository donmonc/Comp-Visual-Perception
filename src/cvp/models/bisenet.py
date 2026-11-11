
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from cvp.models._bisenet_impl.bisenet import BiSeNet


# Model identification and configuration.
MODEL_ID = "yakhyo/face-parsing-resnet18"
NUM_CLASSES = 19
INPUT_SIZE = (512, 512)

CLASS_NAMES = [
    "background",
    "skin",
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "eyeglasses",
    "left_ear",
    "right_ear",
    "earring",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
    "neck",
    "necklace",
    "clothing",
    "hair",
    "hat",
]


class BiSeNetFaceParser:
    """Load pretrained BiSeNet and predict facial segmentation masks."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
    ):
        self.model_id = MODEL_ID
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Resolve the checkpoint relative to the project root.
        project_root = Path(__file__).resolve().parents[3]

        if checkpoint_path is None:
            checkpoint_path = project_root / "weights" / "resnet18.pt"

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"BiSeNet checkpoint not found: {self.checkpoint_path}"
            )

        print(f"Loading model: {self.model_id}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")

        # Initialise architecture without downloading ImageNet weights.
        self.model = BiSeNet(
            num_classes=NUM_CLASSES,
            backbone_name="resnet18",
        )

        # Load the pretrained face-parsing checkpoint.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # Strict loading checks that the checkpoint matches the architecture.
        self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # Use the same preprocessing as the upstream implementation.
        self.transform = transforms.Compose([
            transforms.Resize(
                INPUT_SIZE,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        print("BiSeNet loaded successfully.")
        print(f"Number of classes: {NUM_CLASSES}")

    @torch.inference_mode()
    def predict_mask(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Predict a 19-class facial segmentation mask.

        Returns:
            A uint8 NumPy array with shape (original_height, original_width).
            Each pixel contains its predicted class ID (0-18).
        """

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = image.size

            # Resize and normalise the input.
            image_tensor = self.transform(image).unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)

        # BiSeNet returns three outputs:
        # main segmentation output, auxiliary output 1, auxiliary output 2.
        outputs = self.model(image_tensor)

        # Use the main segmentation output.
        logits = outputs[0]

        # Select the most likely class for each pixel.
        predicted_mask = logits.argmax(dim=1).squeeze(0)

        # Move predictions to CPU as an integer NumPy array.
        predicted_mask = predicted_mask.cpu().numpy().astype(np.uint8)

        # Restore the original image dimensions using nearest-neighbour
        # interpolation so that class IDs are not blended.
        mask_image = Image.fromarray(predicted_mask)

        mask_image = mask_image.resize(
            original_size,
            resample=Image.Resampling.NEAREST,
        )

        return np.array(mask_image, dtype=np.uint8)
