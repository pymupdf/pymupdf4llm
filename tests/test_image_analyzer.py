import os
import dotenv
from src.helpers.image_analyzer import OpenAIImageAnalyzer
import pymupdf
import src

dotenv.load_dotenv()

model = OpenAIImageAnalyzer(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL"), 
    model_name=os.getenv("OPENAI_MODEL_NAME")
)

def test_image_analyzer():
    
    path = os.path.normpath(f'{__file__}/../../tests/test_image_analyzer.pdf')
    path_export = os.path.normpath(f'{__file__}/../../tests/test_image_analyzer2.md')
    
    with pymupdf.open(path) as document:
        actual = src.to_markdown(
                document,
                header=False,  # include/omit page headers
                footer=False,  # include/omit page footers
                show_progress=False,
                force_text=False,
                page_separators=False,
                # analyze_image=model,
            )

    with open(path_export, 'w', encoding='utf8') as f:
        f.write(actual)

if __name__ == "__main__":
    test_image_analyzer()