import os

import pytest
from llama_index.core.schema import Document as LlamaIndexDocument

try:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document as LlamaIndexDocument

    print("All imports are successful.")
except ImportError:
    raise NotImplementedError("Please install 'llama_index' is needed.")


from pymupdf4llm.pymupdf4llm.llama_index.pdf_markdown_reader import PDFMarkdownReader

PDF = "input.pdf"


def _get_test_file_path(file_name: str, __file__: str = __file__) -> str:
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        ".." "helpers",
        file_name,
    )
    file_path = os.path.normpath(file_path)
    return file_path


def test_load_data():
    # Arrange
    # ---
    pdf_reader = PDFMarkdownReader()
    path = _get_test_file_path(PDF, __file__)
    extra_info = {"test_key": "test_value"}

    # Act
    # ---
    documents = pdf_reader.load_data(path, extra_info)

    # Assert
    # ---
    assert isinstance(documents, list)
    for doc in documents:
        assert isinstance(doc, LlamaIndexDocument)


def test_load_data_with_invalid_file_path():
    # Arrange
    # ---
    pdf_reader = PDFMarkdownReader()
    extra_info = {"test_key": "test_value"}
    path = "fake/path"

    # Act & Assert
    # ---
    with pytest.raises(Exception):
        pdf_reader.load_data(path, extra_info)


def test_load_data_with_invalid_extra_info():
    # Arrange
    # ---
    pdf_reader = PDFMarkdownReader()
    extra_info = "invalid_extra_info"
    path = _get_test_file_path(PDF, __file__)

    # Act & Assert
    # ---
    with pytest.raises(TypeError):
        pdf_reader.load_data(path, extra_info)


@pytest.mark.asyncio
async def test_aload_data_with_invalid_file_path():
    # Arrange
    # ---
    pdf_reader = PDFMarkdownReader()
    extra_info = {"test_key": "test_value"}

    # Act
    # ---
    path = "Fake/path"

    # Assert
    # ---
    with pytest.raises(Exception):
        await pdf_reader.aload_data(path, extra_info)


@pytest.mark.asyncio
async def test_aload_data_with_invalid_extra_info():
    # Arrange
    # ---
    pdf_reader = PDFMarkdownReader()
    extra_info = "invalid_extra_info"

    # Act
    # ---
    path = _get_test_file_path(PDF, __file__)

    # Assert
    # ---
    with pytest.raises(TypeError):
        await pdf_reader.aload_data(path, extra_info)
