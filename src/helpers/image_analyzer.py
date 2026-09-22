"""Image analysis back-ends that describe images for markdown output."""

import base64
import io
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent / "prompt" / "visual_descriptor.md"
_PROMPT_IMAGE_ANALYSIS = _PROMPT_PATH.read_text()


class BaseImageAnalyzer(ABC):
    """Base class shared by all image analyzer back-ends."""

    def __init__(self, prompt: str = _PROMPT_IMAGE_ANALYSIS) -> None:
        self.prompt = prompt

    def image_filter(
        self,
        img: str | bytes,
        max_size: int = 1024,
        sharpness_factor: float = 1.5,
        contrast_factor: float = 1.3,
    ) -> str:
        """Pre-process an image and return it as a base64-encoded PNG.

        The image is sharpened, contrast-enhanced, and downscaled if its
        longest side exceeds ``max_size`` (aspect ratio preserved).

        Args:
            img: Image bytes, or a path to an image file.
            max_size: Maximum length of the longest side; larger images are
                scaled down proportionally.
            sharpness_factor: Sharpness enhancement factor (1.0 = original).
            contrast_factor: Contrast enhancement factor (1.0 = original).

        Returns:
            Base64-encoded PNG string.
        """
        try:
            from PIL import Image, ImageEnhance
        except ImportError as exc:
            raise ImportError(
                "`pillow` package not found. Please install it with `pip install pillow`"
            ) from exc

        raw = img if isinstance(img, bytes) else Path(img).read_bytes()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        w, h = image.size
        if max(w, h) > max_size:
            if w > h:
                image = image.resize((max_size, int(h * (max_size / w))), Image.LANCZOS)
            else:
                image = image.resize((int(w * (max_size / h)), max_size), Image.LANCZOS)

        image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        base64_image = base64.b64encode(png_bytes).decode('utf-8')
        return base64_image
    
    @abstractmethod
    def analyze_image(self, img: str | bytes) -> str:
        """Analyze an image using the provided language model.

        Args:
            img: The image to be analyzed.

        Returns:
            The extracted textual content.
        """
        raise NotImplementedError


class HuggingFaceImageAnalyzer(BaseImageAnalyzer):
    """Analyze images using a Hugging Face pipeline (deprecated)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        device_map: str = "auto",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ) -> None:
        super().__init__()
        warnings.warn(
            "HuggingFaceImageAnalyzer is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._model_name = model_name
        self.device_map = device_map
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Silence the Hugging Face transformers logging.
        from transformers import logging

        logging.set_verbosity_error()

    @lru_cache(maxsize=None)
    def _load_model(self):
        """Load the pipeline once and keep it cached on the instance."""
        try:
            from transformers import pipeline
            import torch
        except ImportError as exc:
            raise ImportError(
                "`transformers` and `torch` packages not found. Please install them "
                "with `pip install transformers torch`"
            ) from exc

        return pipeline(
            "image-text-to-text", model=self._model_name, device_map=self.device_map
        )

    def analyze_image(self, img: str | bytes) -> str:
        img_base64 = self.image_filter(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image",
                        "image": img_base64,
                        "mime_type": "image/png",
                    },
                ],
            },
        ]

        pipe = self._load_model()
        output = pipe(
            text=messages,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
        )
        return output[0]["generated_text"][-1]["content"]


class OpenAIImageAnalyzer(BaseImageAnalyzer):
    """Analyze images using an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        reasoning_effort: str = "none",
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self._model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def analyze_image(self, img: str | bytes) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "`openai` package not found. Please install it with `pip install openai`"
            ) from exc

        img_base64 = self.image_filter(img)

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                            },
                        },
                    ],
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        return (response.choices[0].message.content or "").strip()