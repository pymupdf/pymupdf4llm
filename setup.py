import os
import sys
import textwrap

import pipcl

VERSION = "0.2.0"
VERSION_TUPLE = tuple(int(x) for x in VERSION.split("."))

pymupdf_version = "1.27.2"

pymupdf_layout_version = "1.27.2"


PYMUPDF_SETUP_VERSION = os.environ.get("PYMUPDF_SETUP_VERSION")
if PYMUPDF_SETUP_VERSION:
    # Allow testing with non-matching pymupdf/layout versions.
    requires_dist = ["tabulate"]
else:
    requires_dist = [
        f"pymupdf>={pymupdf_version}",
        f"pymupdf_layout>={pymupdf_layout_version}",
        "tabulate",
    ]


def build():
    ret = list()

    version_info = textwrap.dedent(f"""
            # Generated file - do not edit.
            {VERSION=}
            {VERSION_TUPLE=}
            """)
    ret.append((version_info.encode("utf-8"), "pymupdf4llm/versions_file.py"))

    for p in pipcl.git_items("src"):
        ret.append((f"src/{p}", f"pymupdf4llm/{p}"))

    print(f"ret:")
    for i in ret:
        print(f"    {i}")
    return ret


def sdist():
    return pipcl.git_items(".")


p = pipcl.Package(
    "pymupdf4llm-tsr",
    VERSION,
    requires_dist=requires_dist,
    requires_python=">=3.10",
    pure=True,
    author="TSR",
    author_email="tusharsoni.info@gmail.com",
    summary="PyMuPDF Utilities for LLM/RAG with Visual Analyzer",
    description="README.md",
    description_content_type="text/markdown",
    classifier=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
    ],
    license="MIT",
    project_url=[
        "Source, https://github.com/iam-tsr/pymupdf4llm",
    ],
    fn_build=build,
    fn_sdist=sdist,
)

build_wheel = p.build_wheel

if __name__ == "__main__":
    p.handle_argv(sys.argv)
