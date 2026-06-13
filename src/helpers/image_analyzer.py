import base64
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from functools import lru_cache

prompt_path = Path(__file__).resolve().parent / "prompt" / "visual_descriptor.md"

_PROMPT_IMAGE_ANALYSIS = prompt_path.read_text()

class BaseImageAnalyzer(ABC):
    def __init__(
        self,
        model: str,
        prompt: str = _PROMPT_IMAGE_ANALYSIS,
        max_output_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the ImageAnalyzer.

        Args:
            image: The image to analyze.
            inference: The inference engine to use.
            model: The model to use.
            prompt: The prompt to use.
            mime_type: The MIME type of the image.
            max_output_tokens: The maximum number of output tokens.
            temperature: The temperature for the inference engine.
        """
        self.model = model
        self.prompt = prompt
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    
    def image_filter(
        self,
        img: str | bytes,
        max_size: int = 1024,
        sharpness_factor: float = 1.5,
        contrast_factor: float = 1.3,
    ) -> str:
        """Pre-process image by applying sharpening, contrast enhancement, and resizing,
        then encode to base64.

        Args:
            img: Image bytes or file path.
            max_size: max_size: Maximum dimension for the longest side; image is scaled down proportionally if larger or vice versa.
            sharpness_factor: Sharpness enhancement factor (1.0 = original).
            contrast_factor: Contrast enhancement factor (1.0 = original).

        Returns:
            Base64-encoded PNG string.
        """
        try:
            from PIL import Image, ImageEnhance
            import io
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
    """
    Analyze images using Hugging Face pipeline.
    """
    def __init__(
            self, 
            model_name: str = "Qwen/Qwen3.5-0.8B", 
            device_map: str = "auto",
            temperature: float = 0.7,
            max_output_tokens: int = 2048
        ):
        super().__init__(model=model_name)
        warnings.warn(
            "HuggingFaceImageAnalyzer is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        self._model_name = model_name
        self.device_map = device_map
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Suppress all warnings
        from transformers import logging
        logging.set_verbosity_error()

    @lru_cache(maxsize=None)
    def _load_model(self):
        try:
            from transformers import pipeline
            import torch
        except ImportError as exc:
            raise ImportError(
                "`transformers` and `torch` packages not found. please install them with "
                "`pip install transformers torch`"
            ) from exc

        pipe = pipeline("image-text-to-text", model=self._model_name, device_map=self.device_map)
        return pipe

    def analyze_image(self, img: str | bytes) -> str:
        img_base64 = self.image_filter(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT_IMAGE_ANALYSIS},
                    {
                        "type": "image",
                        "image": img_base64,
                        "mime_type": "image/png",
                    },
                ],
            },
        ]

        generate_kwargs = {
            "do_sample": True,
            "temperature": self.temperature,
            "max_new_tokens": self.max_output_tokens,
        }

        pipe = self._load_model()
        output = pipe(text=messages, **generate_kwargs)
        return output[0]['generated_text'][-1]['content']


class GroqImageAnalyzer(BaseImageAnalyzer):
    """
    Analyze images using Groq models.
    """
    def __init__(
            self, 
            api_key: str,
            model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: float = 0.7,
            max_output_tokens: int = 2048
        ):
        super().__init__(model=model_name)
        self._model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


    def analyze_image(self, img: str | bytes) -> str:
        try:
            import groq
        except ImportError:
            raise ImportError(
                "`groq` package not found. please install it with "
                "`pip install groq`"
            )

        img_base64 = self.image_filter(img)

        client = groq.Client()
        response = client.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT_IMAGE_ANALYSIS},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            },
                        },
                    ],
                },
            ],
            model = self._model_name,
            max_tokens = self.max_output_tokens,
            temperature = self.temperature,
        )

        return (response.choices[0].message.content or "").strip()


class OpenAIImageAnalyzer(BaseImageAnalyzer):
    """
    Analyze images using OpenAI models.
    """
    def __init__(
            self, 
            api_key: str,
            base_url: str,
            model_name: str,
            temperature: float = 0.7,
            max_output_tokens: int = 2048
        ):
        super().__init__(model=model_name)
        self.api_key = api_key
        self.base_url = base_url
        self._model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


    def analyze_image(self, img: str | bytes) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "`openai` package not found. please install it with "
                "`pip install openai`"
            )

        img_base64 = self.image_filter(img)

        client = openai.OpenAI(
            api_key = self.api_key,
            base_url = self.base_url
        )
        response = client.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT_IMAGE_ANALYSIS},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            },
                        },
                    ],
                },
            ],
            model = self._model_name,
            max_tokens = self.max_output_tokens,
            temperature = self.temperature,
        )

        return (response.choices[0].message.content or "").strip()