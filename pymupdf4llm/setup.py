import os
import setuptools
from pathlib import Path

readme = Path("README.md").read_bytes().decode()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Utilities",
]

VERSION = "1.27.2.1"
VERSION_TUPLE = tuple(int(x) for x in VERSION.split("."))

PYMUPDF_SETUP_VERSION = os.environ.get('PYMUPDF_SETUP_VERSION')
if PYMUPDF_SETUP_VERSION:
    # Allow testing with non-matching pymupdf/layout versions.
    requires = ["tabulate"]
else:
    requires = [
            f"pymupdf==1.27.2",
            f"pymupdf_layout==1.27.2",
            "tabulate",
            ]

text = f"# Generated file - do not edit.\n{VERSION=}\n{VERSION_TUPLE=}\n"
Path("pymupdf4llm/versions_file.py").write_text(text)

setuptools.setup(
    name="pymupdf4llm",
    version=VERSION,
    author="Artifex",
    author_email="support@artifex.com",
    description="PyMuPDF Utilities for LLM/RAG",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requires,
    python_requires=">=3.10",
    license="Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License",
    url="https://github.com/pymupdf/pymupdf4llm",
    classifiers=classifiers,
    package_data={
        "pymupdf4llm": ["helpers/*.py", "llama/*.py", "ocr/*.py"],
    },
    project_urls={
        "Documentation": "https://pymupdf.readthedocs.io/",
        "Source": "https://github.com/pymupdf/pymupdf4llm/tree/main/pymupdf4llm",
        "Tracker": "https://github.com/pymupdf/pymupdf4llm/issues",
        "Changelog": "https://github.com/pymupdf/pymupdf4llm/blob/main/CHANGES.md",
        "License": "https://github.com/pymupdf/pymupdf4llm/blob/main/LICENSE",
    },
)
