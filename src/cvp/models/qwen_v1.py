
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class Qwen25VLEncoder:
    """Load Qwen2.5-VL and generate responses about images."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id

        # Prevent accidental loading of the 3B model on local CPU.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen2.5-VL inference requires a CUDA-enabled GPU "
                "in this implementation. Use a remote GPU environment."
            )

        self.device = torch.device("cuda")

        print(f"Loading model: {self.model_id}")
        print(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        self.model.eval()

        print("Qwen2.5-VL loaded successfully.")

    @torch.inference_mode()
    def describe_image(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a response for a single image."""

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # Convert the conversation into the model's expected format.
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Extract and preprocess image inputs.
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        # Generate the model response.
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Remove the input prompt tokens from the generated sequence.
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()
