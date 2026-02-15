import sys
import pytest
import json
from unittest.mock import patch, MagicMock
from pymupdf4llm.__main__ import main

TEST_PDF = "examples/country-capitals/national-capitals.pdf"


@pytest.fixture
def mock_pymupdf():
    with patch("pymupdf.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.metadata = {"title": "Test PDF"}
        mock_doc.is_encrypted = False
        mock_doc.is_pdf = True
        mock_open.return_value = mock_doc
        yield mock_open


@pytest.fixture
def mock_to_markdown():
    with patch("pymupdf4llm.to_markdown") as mock:
        mock.return_value = "# Test Markdown"
        yield mock


def test_cli_basic_markdown(mock_pymupdf, mock_to_markdown):
    test_args = ["pymupdf4llm", TEST_PDF]
    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout.write") as mock_write:
            main()
            mock_to_markdown.assert_called_once()
            args, kwargs = mock_to_markdown.call_args
            assert args[0] == mock_pymupdf.return_value
            written = "".join(call[0][0] for call in mock_write.call_args_list)
            assert "# Test Markdown" in written


def test_cli_output_file(mock_pymupdf, mock_to_markdown, tmp_path):
    output_file = tmp_path / "output.md"
    test_args = ["pymupdf4llm", TEST_PDF, "-o", str(output_file)]
    with patch.object(sys, "argv", test_args):
        main()
        assert output_file.read_text() == "# Test Markdown"


def test_cli_pages_parsing(mock_pymupdf, mock_to_markdown):
    test_args = ["pymupdf4llm", TEST_PDF, "-p", "1,3,5-7"]
    with patch.object(sys, "argv", test_args):
        main()
        args, kwargs = mock_to_markdown.call_args
        assert kwargs["pages"] == [0, 2, 4, 5, 6]


def test_cli_pages_n_resolution(mock_pymupdf, mock_to_markdown):
    test_args = ["pymupdf4llm", TEST_PDF, "-p", "5-N"]
    with patch.object(sys, "argv", test_args):
        main()
        args, kwargs = mock_to_markdown.call_args
        assert kwargs["pages"] == [4, 5, 6, 7, 8, 9]


def test_cli_json_format(mock_pymupdf):
    with patch("pymupdf4llm.to_json") as mock_to_json:
        mock_to_json.return_value = {"key": "value"}
        test_args = ["pymupdf4llm", TEST_PDF, "--json"]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout.write") as mock_write:
                main()
                mock_to_json.assert_called_once()
                written = "".join(call[0][0] for call in mock_write.call_args_list)
                assert json.loads(written) == {"key": "value"}


def test_cli_text_format(mock_pymupdf):
    with patch("pymupdf4llm.to_text") as mock_to_text:
        mock_to_text.return_value = "Plain text content"
        test_args = ["pymupdf4llm", TEST_PDF, "--text"]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout.write") as mock_write:
                main()
                mock_to_text.assert_called_once()
                written = "".join(call[0][0] for call in mock_write.call_args_list)
                assert "Plain text content" in written


def test_cli_missing_file():
    test_args = ["pymupdf4llm", "non_existent.pdf"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_cli_version():
    test_args = ["pymupdf4llm", "--version"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_cli_page_chunks(mock_pymupdf, mock_to_markdown):
    mock_to_markdown.return_value = [{"text": "page 1"}]
    test_args = ["pymupdf4llm", TEST_PDF, "--page-chunks"]
    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout.write") as mock_write:
            main()
            args, kwargs = mock_to_markdown.call_args
            assert kwargs["page_chunks"] is True
            written = "".join(call[0][0] for call in mock_write.call_args_list)
            assert json.loads(written) == [{"text": "page 1"}]


def test_cli_page_separators(mock_pymupdf, mock_to_markdown):
    test_args = ["pymupdf4llm", TEST_PDF, "--page-separators"]
    with patch.object(sys, "argv", test_args):
        main()
        args, kwargs = mock_to_markdown.call_args
        assert kwargs["page_separators"] is True


def test_cli_info(mock_pymupdf):
    test_args = ["pymupdf4llm", TEST_PDF, "--info"]
    with patch.object(sys, "argv", test_args):
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
            mock_print.assert_called_once()
            printed_text = mock_print.call_args[0][0]
            info = json.loads(printed_text)
            assert "page_count" in info
            assert info["page_count"] == 10


def test_cli_bash_completion(mock_pymupdf):
    test_args = ["pymupdf4llm", "--bash-completion"]
    with patch.object(sys, "argv", test_args):
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
            mock_print.assert_called_once()
            assert "complete -F _pymupdf4llm_completion" in mock_print.call_args[0][0]
