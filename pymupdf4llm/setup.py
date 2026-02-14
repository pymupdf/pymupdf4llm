import setuptools
from pathlib import Path

readme = Path("README.md").read_bytes().decode()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Topic :: Utilities",
]

version = "0.3.4"
pymupdf_version = "1.27.1"
pymupdf_version_tuple = tuple(int(x) for x in pymupdf_version.split("."))
requires = [f"pymupdf>={pymupdf_version}", "tabulate"]
extras_require = {
    "layout": [f"pymupdf-layout>={pymupdf_version}"],
}

text = f"# Generated file - do not edit.\nMINIMUM_PYMUPDF_VERSION = {pymupdf_version_tuple}\nVERSION = '{version}'\n"
Path("pymupdf4llm/versions_file.py").write_text(text)

setuptools.setup(
    name="pymupdf4llm",
    version=version,
    author="Artifex",
    author_email="support@artifex.com",
    description="PyMuPDF Utilities for LLM/RAG",
    packages=setuptools.find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requires,
    extras_require=extras_require,
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
