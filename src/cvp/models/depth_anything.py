
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2Encoder:
    """Load Depth Anything V2 and predict relative depth maps."""

    def __init__(self, model_id: str = MODEL_ID):
        # Store the model ID as the single source of truth.
        self.model_id = model_id

        # Use GPU when available; otherwise use CPU.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading model: {self.model_id}")
        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)

        self.model.to(self.device)
        self.model.eval()

        print("Depth Anything V2 loaded successfully.")

    @torch.inference_mode()
    def predict_depth(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Predict a relative depth map for an image.

        Returns:
            A float32 NumPy array with the same height and width
            as the original image.
        """

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Open the image and convert it to RGB.
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

            # Preprocess the image for Depth Anything V2.
            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

        # Move input tensors to the model's device.
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Run depth estimation.
        outputs = self.model(**inputs)

        # Resize the predicted depth to the original image dimensions.
        predicted_depth = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        )

        depth_map = predicted_depth.squeeze().float().cpu().numpy()

        return depth_map.astype(np.float32)
