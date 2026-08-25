# Layout-Aware Chunking — internals

A retrieval-friendly chunker that splits a PDF into coherent pieces using
**PDF-native layout signals** — layout boxes, box classes, font sizes,
vertical and horizontal gaps, and page breaks — instead of slicing rendered
Markdown. It runs without semantic embeddings and with no required
dependency beyond PyMuPDF itself (`tiktoken` is optional).

User-facing API documentation lives in the repository root `CHUNKING.md`.
This file maps the pipeline for contributors.

## Pipeline

```
parse_document(...)                       (document_layout.py — upstream)
   Step A  LayoutBox → SentenceUnit           (sentence_builder.py)
           text via text_source.box_to_markdown — the ONLY renderer callsite
   Step B  boundary scores (layout-only)      (boundary_scorer.py)
   Step C  assemble by score/structure/budget (chunk_assembler.py)
   Step D  refine: split / semantic / budget  (chunk_assembler.py)
   Step E  Chunk + t/f/s views + metadata     (serializer.py)
        →  ChunkedDocument                    (chunked_document.py)
```

## Module contracts

- **text_source.py** — single dispatch from a LayoutBox to its canonical
  markdown, mirroring `ParsedDocument.to_markdown`. Upstream renderer
  signature drift is absorbed here and nowhere else. `table_content()` is
  the only reader of the `box.table` dict schema: it routes markdown|html
  with `is not None` (a degenerate table carries `markdown == ""` and must
  never route as html). Renderers that mutate textlines in place are fed a
  defensive copy, so chunking never changes a later `to_markdown`.
  Documented deviations: figures get `[Figure f{n}: WxH]` placeholders;
  formula text is kept (to_markdown drops it); images stay in metadata.

- **sentence_builder.py** — units carry precomputed structure hints,
  geometry, and (for figures) the document-wide figure number that becomes
  the `f{n}` view id.

- **boundary_scorer.py** — layout signals only. No embedder, no network,
  no model.

- **chunk_assembler.py** — never joins or tokenizes chunk text; budgets
  sum per-unit token counts precomputed by the pipeline (documented error
  contract: sum vs joined-text tokenization differs by at most the number
  of joins). ProtoChunks track a contiguous `unit_range` over the
  renumbered working unit list.

- **serializer.py** — materializes chunk text once, attaches citation
  metadata (`section_path` = the innermost owning section's path, derived
  from layout-detected headings — the PDF TOC is never consulted, so the
  path stays populated and consistent with the sections view on documents
  without bookmarks; D18), and builds the tables/figures/sections views
  with bidirectional chunk links.
  `TableChunk.headers` comes exclusively from `<th>` in the engine's HTML
  rendering (stdlib HTMLParser) — provably `[]` on markdown-mode parses,
  no local header heuristics.

- **chunked_document.py** — the `Sequence[Chunk]` return type: element
  registry (every box preserved, header/footer included), `get()` over all
  public ids, `diagnostics`, assembly-tier `reassemble_chunks()` from retained units,
  JSON-safe exports with `content_hash`.

## Seam rules (rebase safety vs upstream)

All chunking code lives in this package. `document_layout.py` is touched
only inside the `ParsedDocument.to_chunks` method hunk; renderers, table
code, and `to_markdown` are never edited. The engine-internal parse flag
`render_html_tables` is rejected at the public boundary; the HTML table
opt-in is exposed as `table_output="html"` (like `to_markdown`), and the
public `edge_threshold` parse option passes through to `parse_document`.
On a PyMuPDF without the layout-union table model, `table_output="html"`
degrades to markdown tables with a parser warning. Element ids are scoped
to the parse options: html mode re-splits text boxes around tables, so
`p{page}.b{box}` addresses are not comparable across md/html parses.

## Testing

`tests/test_chunking_text_source.py` — per-box equivalence with
`to_markdown` and the side-effect-free guarantee.
`tests/test_chunking_api.py` — public surface, kwargs router, containment,
partial-parse addressing.
`tests/test_chunking_v2.py` — id contracts, view round-trips, reassemble_chunks
tiers, diagnostics, content_hash, canned html-mode adapter tests.
`tests/test_chunking_html_tables.py` — live html-mode integration
(D13 `<th>` header extraction, markdown|html exclusivity, chunk-text
containment in `to_markdown(table_output="html")`); skipped unless the
installed PyMuPDF carries the layout-union table model.
