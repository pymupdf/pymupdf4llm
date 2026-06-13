import os
import dotenv
from pymupdf4llm.helpers.image_analyzer import OpenAIImageAnalyzer
import pymupdf
import pymupdf4llm

dotenv.load_dotenv()

analyzer = OpenAIImageAnalyzer(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL"), 
    model_name=os.getenv("OPENAI_MODEL_NAME"),
    temperature=0.7,
    max_output_tokens=2048,
)

def test_image_analyzer():
    
    doc_path = os.path.normpath(f'{__file__}/../../tests/test_image_analyzer.pdf')
    path_export = os.path.normpath(f'{__file__}/../../tests/test_image_analyzer2.md')
    
    with pymupdf.open(doc_path) as document:
        actual = pymupdf4llm.to_markdown(
                document,
                header=False,  # include/omit page headers
                footer=False,  # include/omit page footers
                force_text=False,
                dpi=300,
                analyze_image=analyzer,
            )

    with open(path_export, 'w', encoding='utf8') as f:
        f.write(actual)

if __name__ == "__main__":
    test_image_analyzer()