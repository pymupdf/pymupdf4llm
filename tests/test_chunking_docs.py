"""Executes every ```python block in CHUNKING.md, in order, in one namespace.

The cookbook's ``# ->`` lines are captured output; this test guarantees the
code that produced them still runs. Blocks are executed from the
repository root so the relative PDF paths in the document resolve. Blocks
that need an optional dependency (langchain-core, llama-index-core) are
skipped when it is not installed; any other exception fails the test with
the block index and its first line.
"""

import os
import re
import traceback

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "CHUNKING.md")

# Top-level module names whose absence only skips the block using them.
OPTIONAL_MODULES = ("langchain_core", "llama_index")

_PYTHON_FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```", re.S | re.M)


def _python_blocks():
    with open(DOC, encoding="utf-8") as f:
        return _PYTHON_FENCE.findall(f.read())


def test_chunking_md_python_blocks_run(monkeypatch):
    monkeypatch.chdir(ROOT)
    blocks = _python_blocks()
    assert blocks, "no ```python blocks found in CHUNKING.md"

    namespace = {"__name__": "chunking_md"}
    skipped = []
    for index, source in enumerate(blocks):
        first_line = source.strip().splitlines()[0]
        try:
            exec(compile(source, f"CHUNKING.md[block {index}]", "exec"), namespace)
        except ModuleNotFoundError as exc:
            top = (exc.name or "").split(".")[0]
            if top in OPTIONAL_MODULES:
                skipped.append((index, first_line, top))
                continue
            pytest.fail(
                f"CHUNKING.md block {index} ({first_line!r}) needs module "
                f"{exc.name!r}:\n{traceback.format_exc()}"
            )
        except Exception:
            pytest.fail(
                f"CHUNKING.md block {index} ({first_line!r}) raised:\n"
                f"{traceback.format_exc()}"
            )

    for index, first_line, module in skipped:
        print(f"skipped block {index} ({first_line!r}): optional module "
              f"{module!r} not installed")
