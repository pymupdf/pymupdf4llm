## Image Analyzer (New!)

PyMuPDF4LLM now includes a powerful **Image Analyzer** feature designed to bridge the gap between visual content and structured text. This feature uses Vision Language Models (VLMs) to extract detailed information from images—such as logos, complex tables, and charts—and convert them into clean, LLM-ready Markdown.

### How it Works
The Image Analyzer is designed to handle the nuance of visual data that standard OCR often misses:
- **Hierarchical Table Parsing:** Specifically optimized to detect and reconstruct complex, multi-level X-axis structures (e.g., grouping data by "Model" then "Configuration").
- **Brand & Logo Recognition:** Identifies and transcribes visible text from logos and brand marks.
- **Chart & Graph Interpretation:** Converts bar charts, line graphs, and pie charts into structured Markdown tables, capturing data points and trend notes.
- **Smart OCR:** Uses a hybrid approach to only apply OCR where necessary (e.g., on scanned regions or illegible text), preserving the quality of native digital text.

### Key Features
- **Model Agnostic:** Compatible with multiple inference backends, including Hugging Face, Groq, and OpenAI.
- **Layout-Aware:** Maintains the natural reading order and structural context of the document.
- **Markdown Optimized:** Outputs are formatted specifically for RAG pipelines, ensuring that visual data is indexed as meaningfully as the surrounding text.
- **Base-Class Architecture:** Built on an abstract `BaseImageAnalyzer` interface that enforces a unified API across all supported backends.
- **Pre-Processing Pipeline:** Images are automatically pre-processed (sharpening, contrast enhancement, resizing) before inference to ensure consistent quality.
- **Prompt-Driven Classification:** Uses a sophisticated system prompt (`visual_descriptor.md`) that classifies images into three types (equations, charts, general) and applies type-specific formatting rules.
- **Flexible Integration:** Seamlessly integrates into the document layout pipeline by accepting an `analyze_image` parameter in `parse_document()`.
- **Multiple Backends Supported:**
  - **OpenAIImageAnalyzer:** Uses the OpenAI vision API for high-accuracy analysis.
  - **GroqImageAnalyzer:** Leverages Groq's inference engine with the `meta-llama/llama-4-scout-17b-16e-instruct` model for fast, cost-effective analysis.
  - **HuggingFaceImageAnalyzer:** Uses the Hugging Face `image-text-to-text` pipeline (default model: `Qwen/Qwen3.5-0.8B`). *Note: Deprecated and scheduled for removal in future versions.*

### Benchmark Category Averages

Current benchmark category averages:

| Category | Score |
| --- | ---: |
| Education | 0.7058 |
| Economics | 0.4403 |
| Government | 0.6529 |
| Finance | 0.3877 |
| Healthcare | 0.6390 |

### Usage
You can use the Image Analyzer by initializing an `ImageAnalyzer` subclass and passing it to the `parse_document()` function.

```python
pip install pymupdf4llm-tsr
```

```python
import pymupdf4llm
from pymupdf4llm.helpers.image_analyzer import OpenAIImageAnalyzer

# Initialize the analyzer
analyzer = OpenAIImageAnalyzer(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL"), 
    model_name=os.getenv("OPENAI_MODEL_NAME")
)

# Initialize to_markdown to get parsed pdf document
pymupdf4llm.to_markdown(
                document,
                analyze_image=analyzer,
            )
```

### Technical Details
- **Prompt Engineering:** Uses a sophisticated system prompt (`visual_descriptor.md`) that enforces strict structural rules to prevent "hallucinated" layout collapses.
- **Performance:** Optimized to be significantly faster and cheaper than standard vision-based LLM extraction by using efficient inference.
- **Customizable:** Easily configure `model`, `max_output_tokens`, and `temperature` to suit your specific data requirements.
- **Integration Point:** The analyzer is invoked automatically during document processing for all `picture` and `formula` boundary boxes.
- **Optional Feature:** The `analyze_image` parameter is optional, allowing PDF processing without image analysis.