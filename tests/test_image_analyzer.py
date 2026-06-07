import difflib
import os
import platform
import dotenv

dotenv.load_dotenv()

import pymupdf4llm
# import pymupdf4llm
import pymupdf
from pymupdf4llm.helpers.image_analyzer import LlamaCppImageAnalyzer
from llama_cpp.llama_chat_format import Gemma4ChatHandler


MODEL_REPO = "unsloth/gemma-4-E4B-it-GGUF"
MMPROJ_FILE = "mmproj-F16.gguf"
MODEL_FILE = "gemma-4-E4B-it-UD-Q4_K_XL.gguf"

chat_handler = Gemma4ChatHandler.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MMPROJ_FILE,
    verbose=False,
)

def test_image_analyzer():
    
    path = os.path.normpath(f'{__file__}/../../tests/test_image_analyzer.pdf')
    path_export = os.path.normpath(f'{__file__}/../../tests/test_image_analyzer.md')
    image_path = os.path.normpath(f'{__file__}/../../tests/images')
    
    with pymupdf.open(path) as document:
        actual = pymupdf4llm.to_markdown(
                document,
                write_images=False,  # do not write image files
                embed_images=False,  # embed images as base64 strings
                image_format="png",  # image format (embedded or written)
                dpi=300,  # image resolution in dots per inch
                image_path=image_path,
                header=False,  # include/omit page headers
                footer=False,  # include/omit page footers
                show_progress=False,
                force_text=False,
                page_separators=False,
                describe_image=LlamaCppImageAnalyzer(model_name=MODEL_REPO, filename=MODEL_FILE, chat_handler=chat_handler, device="cuda"),
                use_ocr=False,
            )

    with open(path_export, 'w', encoding='utf8') as f:
        f.write(actual)