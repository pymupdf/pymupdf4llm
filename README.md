## Image Analyzer (New!)

PyMuPDF4LLM now includes a powerful **Image Analyzer** feature designed to bridge the gap between visual content and structured text. This feature uses Vision Language Models (VLMs) to extract detailed information from images—such as logos, complex tables, and charts—and convert them into clean, LLM-ready Markdown.

### How it Works
The Image Analyzer is designed to handle the nuance of visual data that standard OCR often misses:
- **Hierarchical Table Parsing:** Specifically optimized to detect and reconstruct complex, multi-level X-axis structures (e.g., grouping data by "Model" then "Configuration").
- **Brand & Logo Recognition:** Identifies and transcribes visible text from logos and brand marks.
- **Chart & Graph Interpretation:** Converts bar charts, line graphs, and pie charts into structured Markdown tables, capturing data points and trend notes.
- **Smart OCR:** Uses a hybrid approach to only apply OCR where necessary (e.g., on scanned regions or illegible text), preserving the quality of native digital text.

### Key Features
- **Model Agnostic:** Compatible with multiple inference backends, including Hugging Face, Groq, OpenAI, and llama-cpp-python.
- **Layout-Aware:** Maintains the natural reading order and structural context of the document.
- **Markdown Optimized:** Outputs are formatted specifically for RAG pipelines, ensuring that visual data is indexed as meaningfully as the surrounding text.

### Usage
You can use the Image Analyzer by calling the `analyze_image` method from the `ImageAnalyzer` classes.

```python
import pymupdf4llm
from pymupdf4llm.helpers.image_analyzer import HuggingFaceImageAnalyzer

# Initialize the analyzer with your preferred model
analyzer = HuggingFaceImageAnalyzer(model_name="Qwen/Qwen3.5-0.8B")

# Analyze an image and get the structured Markdown
markdown_output = analyzer.analyze_image("path/to/chart.png")
print(markdown_output)
```

### Technical Details
- **Prompt Engineering:** Uses a sophisticated system prompt (`visual_descriptor.md`) that enforces strict structural rules to prevent "hallucinated" layout collapses.
- **Performance:** Optimized to be significantly faster and cheaper than standard vision-based LLM extraction by using targeted OCR and efficient inference.
- **Customizable:** Easily configure `max_output_tokens`, `temperature`, and `ocr_dpi` to suit your specific data requirements.

---

*Note: Ensure you have the required dependencies installed for your chosen backend (e.g., `pip install transformers torch` for Hugging Face).*
