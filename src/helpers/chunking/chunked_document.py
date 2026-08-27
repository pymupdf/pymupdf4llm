"""ChunkedDocument: the return type of to_chunks.

A Sequence[Chunk] carrying the element registry, the table/figure/section
views (bidirectionally linked to chunks), a section hierarchy tree,
ingestion diagnostics, and cheap re-chunking from retained units.
"""

import json
from collections.abc import Sequence
from dataclasses import fields
from types import MappingProxyType

from .models import SectionNode

# Assembly-tier parameters reassemble_chunks() can honor from retained units (D11).
# Everything else (sentence_splitter, header_footer_mode, tokenizer,
# weights, parse options) changes the substrate and requires a new
# to_chunks() call.
REASSEMBLY_PARAMS = frozenset({
    "max_tokens",
    "min_tokens",
    "breakpoint_threshold",
    "table_mode",
    "merge_small_chunks",
    "respect_section_starts",
})

_MISSING = object()


class ChunkedDocument(Sequence):
    """Retrieval-ready chunks plus document-level structure.

    - Sequence protocol: ``len(cd)``, ``cd[i]``, iteration, slicing.
    - ``elements``: every layout box the parser saw (header/footer included),
      addressable as ``p{page}.b{box}``.
    - ``tables`` / ``figures`` / ``sections``: views linked to chunks both
      ways (view.chunk_id(s) <-> chunk.metadata.table_ids/figure_ids/
      section_id).
    - ``hierarchy``: sections as a tree of SectionNode (root level 0).
    - ``diagnostics``: structured input for an ingestion gate.
    """

    def __init__(self, chunks, *, elements=(), tables=(), figures=(),
                 sections=(), params=None, doc=None, all_units=(),
                 header_footer_excluded=0):
        self._chunks = tuple(chunks)
        self.elements = list(elements)
        self.tables = list(tables)
        self.figures = list(figures)
        self.sections = list(sections)
        self.params = MappingProxyType(dict(params or {}))
        self._doc = doc
        self._all_units = list(all_units)
        self._header_footer_excluded = header_footer_excluded
        self._text = None
        self._by_id = None
        self._hierarchy = None

        # Bind section element spans to the registry so SectionChunk.text
        # can assemble lazily.
        for s in self.sections:
            lo, hi = s.element_span
            s._elements = tuple(self.elements[lo:hi])

    # ── Sequence protocol ───────────────────────────────────────────

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self._chunks[index])
        return self._chunks[index]

    def __repr__(self):
        return (f"ChunkedDocument(chunks={len(self._chunks)}, "
                f"tables={len(self.tables)}, figures={len(self.figures)}, "
                f"sections={len(self.sections)})")

    @property
    def chunks(self) -> tuple:
        """The chunks as a tuple (== tuple(self)); explicit-access form."""
        return self._chunks

    # ── Lazy document text ──────────────────────────────────────────

    @property
    def text(self) -> str:
        """All chunk text joined in reading order (lazy, cached)."""
        if self._text is None:
            self._text = "\n\n".join(c.text for c in self._chunks)
        return self._text

    # ── Section hierarchy tree ──────────────────────────────────────

    @property
    def hierarchy(self) -> SectionNode:
        """Sections nested as a tree; the root node has level 0."""
        if self._hierarchy is None:
            root = SectionNode()
            stack = [root]
            for s in self.sections:
                node = SectionNode(title=s.title, level=s.level,
                                   section_id=s.id)
                while len(stack) > 1 and stack[-1].level >= s.level:
                    stack.pop()
                stack[-1].children.append(node)
                stack.append(node)
            self._hierarchy = root
        return self._hierarchy

    # ── Addressing ──────────────────────────────────────────────────

    def get(self, id: str, default=_MISSING):
        """Look up any public id: c{n}, t{n}, f{n}, s{n}, p{page}.b{box}.

        Raises KeyError for an unknown id unless *default* is given.
        """
        if self._by_id is None:
            index = {}
            for c in self._chunks:
                index[c.id] = c
            for group in (self.tables, self.figures, self.sections):
                for item in group:
                    index[item.id] = item
            for el in self.elements:
                index[el.id] = el
            self._by_id = index

        found = self._by_id.get(id, _MISSING) if id else _MISSING
        if found is _MISSING:
            if default is _MISSING:
                raise KeyError(id)
            return default
        return found

    # ── Serialization ───────────────────────────────────────────────

    def to_dicts(self, *, include_tagged: bool = True) -> list:
        """Chunks as flat, json-safe, payload-ready dicts (schema §4.5)."""
        return [_chunk_to_dict(c, include_tagged) for c in self._chunks]

    def to_json(self, *, include_tagged: bool = True, **json_kwargs) -> str:
        json_kwargs.setdefault("default", _json_default)
        return json.dumps(
            self.to_dicts(include_tagged=include_tagged),
            **json_kwargs,
        )

    # ── Re-assembly (assembly tier, from retained units) ────────────

    def reassemble_chunks(self, **params):
        """Re-run assembly from retained units with new budget params.

        Only assembly-tier parameters are accepted (REASSEMBLY_PARAMS);
        substrate parameters (sentence_splitter, header_footer_mode,
        tokenizer, weights) and parse options raise ValueError — call
        to_chunks() on a new parse for those.
        """
        unknown = set(params) - REASSEMBLY_PARAMS
        if unknown:
            raise ValueError(
                f"reassemble_chunks only accepts {sorted(REASSEMBLY_PARAMS)}; "
                f"{sorted(unknown)} require a new to_chunks() call"
            )
        from . import _chunk_units

        merged = {**self.params, **params}
        return _chunk_units(self._doc, self._all_units, merged)

    # ── Diagnostics (D16) ───────────────────────────────────────────

    @property
    def diagnostics(self) -> dict:
        """Structured ingestion-gate input, derived from the registry.

        Consumers decide PROCEED/HOLD/REJECT; this only reports facts:
        counts, zero-chunk causes, text-less figures (OCR candidates),
        degenerate tables, and header/footer exclusions.
        """
        pages = [p.page_number for p in self._doc.pages] if self._doc else []
        pages_with_chunks = set()
        for c in self._chunks:
            pages_with_chunks.update(
                range(c.metadata.page_start, c.metadata.page_end + 1))

        zero_chunk_causes = []
        if not self._chunks:
            if not self._all_units:
                zero_chunk_causes.append("no_text_extracted")
            elif all(u.is_header_footer for u in self._all_units):
                zero_chunk_causes.append("all_units_header_footer")
            else:
                zero_chunk_causes.append("all_units_filtered")

        return {
            "chunk_count": len(self._chunks),
            "element_count": len(self.elements),
            "table_count": len(self.tables),
            "figure_count": len(self.figures),
            "section_count": len(self.sections),
            "page_count": len(pages),
            "pages_without_chunks": sorted(set(pages) - pages_with_chunks),
            "zero_chunk_causes": zero_chunk_causes,
            "figures_without_text": [f.id for f in self.figures
                                     if not f.has_text],
            "degenerate_tables": [t.id for t in self.tables
                                  if not t.text.strip()],
            "header_footer_excluded": self._header_footer_excluded,
        }

    # ── Framework exports (stretch, import-guarded) ─────────────────

    def to_langchain_documents(self):
        """Chunks as langchain Documents (requires langchain-core)."""
        from langchain_core.documents import Document

        return [
            Document(page_content=c.text,
                     metadata=_chunk_to_dict(c, False)["metadata"])
            for c in self._chunks
        ]

    def to_llama_nodes(self):
        """Chunks as llama-index TextNodes (requires llama-index-core)."""
        from llama_index.core.schema import TextNode

        return [
            TextNode(id_=c.id, text=c.text,
                     metadata=_chunk_to_dict(c, False)["metadata"])
            for c in self._chunks
        ]


def _chunk_to_dict(chunk, include_tagged: bool) -> dict:
    d = {
        "id": chunk.id,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "metadata": {
            f.name: getattr(chunk.metadata, f.name)
            for f in fields(chunk.metadata)
        },
    }
    if include_tagged:
        d["tagged_content"] = chunk.tagged_content
    return d


def _json_default(o):
    if isinstance(o, bytes):
        return None  # images are not serialized into JSON
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)!r}")
