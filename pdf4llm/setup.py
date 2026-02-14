import os

import setuptools

setup_py_cwd = os.path.dirname(__file__)
with open(os.path.join(setup_py_cwd, "README.md"), encoding="utf-8") as f:
    readme = f.read()

version = "0.3.4"  # must always equal the pymupdf4llm version

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Topic :: Utilities",
]

requires = [f"pymupdf4llm=={version}"]

setuptools.setup(
    name="pdf4llm",
    version=version,
    author="Artifex",
    author_email="support@artifex.com",
    description="PyMuPDF Utilities for LLM/RAG",
    packages=setuptools.find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requires,
    python_requires=">=3.10",
    license="Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License",
    url="https://github.com/pymupdf/RAG",
    classifiers=classifiers,
    package_data={},
    project_urls={
        "Documentation": "https://pymupdf.readthedocs.io/",
        "Source": "https://github.com/pymupdf/RAG/tree/main/pdf4llm/pdf4llm",
        "Tracker": "https://github.com/pymupdf/RAG/issues",
        "Changelog": "https://github.com/pymupdf/RAG/blob/main/CHANGES.md",
        "License": "https://github.com/pymupdf/RAG/blob/main/LICENSE",
    },
)
