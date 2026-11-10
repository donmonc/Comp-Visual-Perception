from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov2-base" #"facebook/dinov2-small" <--- I also tried this but base was slightly better than small.


class Dinov2Encoder:
    """Load DINOv2 and extract global image representations."""

    def __init__(self, model_id: str = MODEL_ID):
        
        self.model_id = model_id
        
        # Automatically select GPU if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_id}")
        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size

        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> torch.Tensor:
        """
        Convert an image into a DINOv2 CLS-token embedding.

        Returns:
            A 1D CPU tensor representing the entire image.
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image and convert grayscale/RGBA images to RGB.
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        # Resize, normalize and convert image to PyTorch tensors.
        inputs = self.processor(images=image, return_tensors="pt")

        # Move input tensors to the same device as the model.
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Forward pass through DINOv2.
        outputs = self.model(**inputs)

        # Token 0 is the CLS token: our global image representation.
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

        return embedding.detach().cpu()
