import concurrent.futures
import copy
import inspect
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pymupdf4llm
from pymupdf4llm import pymupdf
from pymupdf4llm.ocr import OCRMode
from pymupdf4llm.worker_sizing import auto_workers

# Disable MuPDF warnings/errors globally
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def _derive_stem(input):
    if isinstance(input, (str, Path)):
        return Path(input).stem
    return "document"


def _prepare_doc_dir(input, out_dir):
    stem = _derive_stem(input)
    doc_dir = Path(out_dir) / stem if out_dir else None
    if doc_dir:
        doc_dir.mkdir(parents=True, exist_ok=True)
    return stem, doc_dir


def _prepare_local_options(base_options, doc_dir, backend):
    local_options = copy.deepcopy(base_options)
    local_options["doc_dir"] = str(doc_dir) if doc_dir else ""

    if local_options.get("embed_images"):
        if backend == "txt":
            print("embed_images=True ignored for backend 'txt'")
    elif local_options.get("write_images") and doc_dir:
        image_dir = doc_dir / "images"
        image_dir.mkdir(exist_ok=True)
        local_options["image_path"] = "./images"

    return local_options


@contextmanager
def _conversion_log_context(doc_dir):
    if not doc_dir:
        yield
        return

    log_path = Path(doc_dir) / "run-log.txt"
    log_file = log_path.open("w", encoding="utf-8")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    pymupdf.set_messages(stream=sys.stdout)

    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()
        pymupdf.set_messages(stream=None)


def _describe_ocr_mode(options):
    ocr_mode = options.get("use_ocr")
    if isinstance(ocr_mode, OCRMode):
        return ocr_mode.description  # pylint: disable=no-value-for-parameter
    return OCRMode(ocr_mode).description  # pylint: disable=no-value-for-parameter


def _log_run_start(options):
    start = datetime.now()
    print(f"Start: {start.isoformat()}")
    print(f"OCR mode: {_describe_ocr_mode(options)}")
    print(f"Layout mode: {options.get('layout_enabled')}")


def _log_run_end(success, error=None):
    end = datetime.now()
    print(f"\nEnd: {end.isoformat()}")
    print("Status: success" if success else "Status: error")
    if error is not None:
        print("Exception:")
        print(error)


def _apply_consumer(output, consumer, consumer_kwargs, *, input, backend, options):
    if consumer is None:
        return output

    if not callable(consumer):
        raise TypeError("consumer must be callable")

    kwargs = dict(consumer_kwargs or {})
    signature = inspect.signature(consumer)
    params = signature.parameters

    context_values = {
        "input": input,
        "backend": backend,
        "options": options,
    }
    for name, value in context_values.items():
        if name in params and name not in kwargs:
            kwargs[name] = value

    consumed = consumer(output, **kwargs)
    return output if consumed is None else consumed


def _write_output(output, input, out_dir, backend):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _derive_stem(input)
    doc_dir = out_dir / stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "md": _write_markdown,
        "json": _write_json,
        "txt": _write_txt,
    }
    writer = writers.get(backend)
    if writer is None:
        raise ValueError(f"Unsupported backend: {backend}")
    writer(output, doc_dir, stem)


def _write_markdown(md_text, out_dir, stem):
    if isinstance(md_text, (list, dict)):
        path = out_dir / f"{stem}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(md_text, f, ensure_ascii=False, indent=1)
    elif isinstance(md_text, str):
        path = out_dir / f"{stem}.md"
        path.write_text(md_text, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported markdown output type: {type(md_text)}")


def _write_json(obj, out_dir, stem):
    path = out_dir / f"{stem}.json"
    path.write_bytes(obj.encode())


def _write_txt(output, out_dir, stem):
    if isinstance(output, (list, dict)):
        path = out_dir / f"{stem}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=1)
    elif isinstance(output, str):
        path = out_dir / f"{stem}.txt"
        path.write_text(output, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported markdown output type: {type(output)}")


def _worker_convert_one(input, backend, options, consumer, consumer_kwargs, doc_dir):
    doc_dir = Path(doc_dir)

    with _conversion_log_context(doc_dir):
        _log_run_start(options)

        try:
            doc = pymupdf.open(input)
            print(f"Pages: {doc.page_count}")
            doc.close()

            out = convert_document(
                input,
                backend=backend,
                options=options,
                consumer=consumer,
                consumer_kwargs=consumer_kwargs,
            )
            _log_run_end(True)
            return BatchResult(input, True, None, out)

        except Exception as e:
            _log_run_end(False, e)
            return BatchResult(input, False, e, None)


# ---------------------------------------------------------------------------
# Internal single-document API
# ---------------------------------------------------------------------------


def convert_document(input, *, backend, options, consumer, consumer_kwargs):
    """
    Convert a single document using the selected backend and options.
    This function is used by all strategies (sequential, process pool, persistent).
    """

    layout_enabled = pymupdf4llm._use_layout
    conversion_options = copy.deepcopy(options)
    conversion_options.pop("layout_enabled", None)
    print(f"Processing document: {input}")

    if layout_enabled:
        converters = {
            "md": pymupdf4llm.to_markdown,
            "json": pymupdf4llm.to_json,
            "txt": pymupdf4llm.to_text,
        }
        converter = converters.get(backend)
        if converter is None:
            raise ValueError(f"Unsupported backend: {backend}")
        output = converter(input, **conversion_options)
        return _apply_consumer(
            output,
            consumer,
            consumer_kwargs,
            input=input,
            backend=backend,
            options=conversion_options,
        )

    converters = {"md": pymupdf4llm.to_markdown}
    converter = converters.get(backend)
    if converter is None:
        raise ValueError(f"Unsupported backend in non-layout mode: {backend}")
    output = converter(input, **conversion_options)
    return _apply_consumer(
        output,
        consumer,
        consumer_kwargs,
        input=input,
        backend=backend,
        options=conversion_options,
    )


# ---------------------------------------------------------------------------
# Base Strategy + BatchResult
# ---------------------------------------------------------------------------


class _BaseStrategy:
    def run(
        self,
        inputs,
        backend,
        options,
        consumer,
        consumer_kwargs,
        out_dir,
        progress_callback,
    ):
        raise NotImplementedError


class BatchResult:
    """
    Represents the result of processing a single document.
    """

    def __init__(self, input, success, error, output):
        self.input = input
        self.success = success
        self.error = error
        self.output = output

    def __repr__(self):
        return f"BatchResult(input={self.input!r}, success={self.success})"


# ---------------------------------------------------------------------------
# Sequential Strategy
# ---------------------------------------------------------------------------


class _SequentialStrategy(_BaseStrategy):
    def run(
        self,
        inputs,
        backend,
        options,
        consumer,
        consumer_kwargs,
        out_dir,
        progress_callback,
    ):
        total = len(inputs)
        done = 0

        for inp in inputs:
            _, doc_dir = _prepare_doc_dir(inp, out_dir)
            local_options = _prepare_local_options(options, doc_dir, backend)

            with _conversion_log_context(doc_dir):
                _log_run_start(local_options)
                try:
                    out = convert_document(
                        inp,
                        backend=backend,
                        options=local_options,
                        consumer=consumer,
                        consumer_kwargs=consumer_kwargs,
                    )

                    if out_dir is not None:
                        _write_output(out, inp, out_dir, backend)

                    result = BatchResult(inp, True, None, out)

                except Exception as e:
                    result = BatchResult(inp, False, e, None)

                _log_run_end(
                    result.success, result.error if not result.success else None
                )

            done += 1
            if progress_callback:
                progress_callback(done, total)

            yield result


# ---------------------------------------------------------------------------
# Process Pool Strategy
# ---------------------------------------------------------------------------


class _ProcessPoolStrategy(_BaseStrategy):
    def __init__(self, workers):
        self.workers = workers

    def run(
        self,
        inputs,
        backend,
        options,
        consumer,
        consumer_kwargs,
        out_dir,
        progress_callback,
    ):
        total = len(inputs)
        done = 0

        task_args = []
        for inp in inputs:
            _, doc_dir = _prepare_doc_dir(inp, out_dir)
            local_options = _prepare_local_options(options, doc_dir, backend)

            task_args.append(
                (inp, backend, local_options, consumer, consumer_kwargs, str(doc_dir))
            )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers
        ) as executor:

            futures = [
                executor.submit(_worker_convert_one, *args) for args in task_args
            ]

            for fut, inp in zip(futures, inputs):
                try:
                    result = fut.result()

                    if result.success and out_dir is not None:
                        _write_output(result.output, inp, out_dir, backend)

                except Exception as e:
                    result = BatchResult(inp, False, e, None)

                done += 1
                if progress_callback:
                    progress_callback(done, total)

                yield result


# ---------------------------------------------------------------------------
# Streaming Process Pool Strategy
# ---------------------------------------------------------------------------


class _StreamingProcessPoolStrategy(_BaseStrategy):
    def __init__(self, workers):
        self.workers = workers

    def run(
        self,
        inputs,
        backend,
        options,
        consumer,
        consumer_kwargs,
        out_dir,
        progress_callback,
    ):
        total = len(inputs)
        done = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers
        ) as executor:

            future_to_input = {}

            for inp in inputs:
                _, doc_dir = _prepare_doc_dir(inp, out_dir)
                local_options = _prepare_local_options(options, doc_dir, backend)

                fut = executor.submit(
                    _worker_convert_one,
                    inp,
                    backend,
                    local_options,
                    consumer,
                    consumer_kwargs,
                    str(doc_dir),
                )
                future_to_input[fut] = inp

            for fut in concurrent.futures.as_completed(future_to_input):
                inp = future_to_input[fut]

                try:
                    result = fut.result()

                    if result.success and out_dir is not None:
                        _write_output(result.output, inp, out_dir, backend)

                except Exception as e:
                    result = BatchResult(inp, False, e, None)

                done += 1
                if progress_callback:
                    progress_callback(done, total)

                yield result


# ---------------------------------------------------------------------------
# Persistent worker pool
# ---------------------------------------------------------------------------

_worker_state = {}


def _worker_init(backend, options, consumer, consumer_kwargs):
    """
    Initializer for persistent worker processes.
    Runs once per worker and stores all expensive state.
    """
    _worker_state["backend"] = backend
    _worker_state["options"] = options
    _worker_state["consumer"] = consumer
    _worker_state["consumer_kwargs"] = consumer_kwargs


def _worker_convert_one_persistent(input, doc_dir):
    """
    Worker function for persistent pool.
    Reuses OCR/layout state stored in _worker_state.
    """
    doc_dir = Path(doc_dir)

    with _conversion_log_context(doc_dir):
        _log_run_start(_worker_state["options"])

        try:
            # Page count logging
            doc = pymupdf.open(input)
            print(f"Pages: {doc.page_count}")
            doc.close()

            # Per-document options
            local_options = _prepare_local_options(
                _worker_state["options"],
                doc_dir,
                _worker_state["backend"],
            )

            # Actual conversion
            print(f"Processing document: {input}")
            out = convert_document(
                input,
                backend=_worker_state["backend"],
                options=local_options,
                consumer=_worker_state["consumer"],
                consumer_kwargs=_worker_state["consumer_kwargs"],
            )

            print(f"Output type: {type(out)}")
            if isinstance(out, str):
                print(f"Output length: {len(out)}")
            else:
                print(f"Output repr: {repr(out)}")

            _log_run_end(True)
            return BatchResult(input, True, None, out)

        except Exception as e:
            _log_run_end(False, e)
            return BatchResult(input, False, e, None)


# ---------------------------------------------------------------------------
# Persistent Pool Strategy
# ---------------------------------------------------------------------------


class _PersistentPoolStrategy(_BaseStrategy):
    def __init__(self, workers):
        self.workers = workers

    def run(
        self,
        inputs,
        backend,
        options,
        consumer,
        consumer_kwargs,
        out_dir,
        progress_callback,
    ):
        total = len(inputs)
        done = 0

        # Persistent workers keep OCR/layout state alive
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_worker_init,
            initargs=(backend, copy.deepcopy(options), consumer, consumer_kwargs),
        ) as executor:

            futures = []
            for inp in inputs:
                _, doc_dir = _prepare_doc_dir(inp, out_dir)

                futures.append(
                    executor.submit(_worker_convert_one_persistent, inp, str(doc_dir))
                )

            # Collect results
            for fut, inp in zip(futures, inputs):
                try:
                    result = fut.result()
                    if result.success and out_dir is not None:
                        _write_output(result.output, inp, out_dir, backend)

                except Exception as e:
                    result = BatchResult(inp, False, e, None)

                done += 1
                if progress_callback:
                    progress_callback(done, total)

                yield result


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------


def _select_strategy(workers, streaming, persistent=False):
    """
    Select the appropriate batch processing strategy.
    """

    # Single-threaded
    if workers in (None, 0, 1):
        return _SequentialStrategy()

    # Persistent pool (OCR/layout warm)
    if persistent:
        return _PersistentPoolStrategy(workers)

    # Streaming pool (unordered results)
    if streaming:
        return _StreamingProcessPoolStrategy(workers)

    # Normal process pool
    return _ProcessPoolStrategy(workers)


# ---------------------------------------------------------------------------
# Default options per backend
# ---------------------------------------------------------------------------


def _default_options(backend):
    """
    Default conversion options for each backend.
    """

    if backend == "md":
        return dict(
            image_dpi=150,
            embed_images=False,
            filename="",
            footer=False,
            force_ocr=False,
            force_text=True,
            header=False,
            ignore_code=False,
            image_format="png",
            image_path="",
            ocr_dpi=150,
            ocr_function=None,
            ocr_language="eng",
            page_chunks=False,
            page_height=None,
            page_separators=False,
            pages=None,
            page_width=612,
            show_progress=False,
            use_ocr=True,
            write_images=False,
        )

    elif backend == "json":
        return dict(
            image_dpi=150,
            image_format="png",
            image_path="",
            pages=None,
            ocr_dpi=150,
            write_images=False,
            embed_images=False,
            show_progress=False,
            force_text=True,
            use_ocr=True,
            force_ocr=False,
            ocr_language="eng",
            ocr_function=None,
        )

    elif backend == "txt":
        return dict(
            filename="",
            header=True,
            footer=True,
            pages=None,
            ignore_code=False,
            show_progress=False,
            force_text=True,
            ocr_dpi=150,
            use_ocr=True,
            force_ocr=False,
            ocr_language="eng",
            ocr_function=None,
            table_format="grid",
            table_max_width=100,
            table_min_col_width=10,
            page_chunks=False,
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _validate_image_output_options(out_dir, options):
    if options.get("write_images") and not out_dir:
        raise ValueError("write_images=True requires out_dir")


# ---------------------------------------------------------------------------
# Main batch API
# ---------------------------------------------------------------------------


def convert_batch(
    inputs,
    *,
    workers=None,
    streaming=False,
    out_dir=None,
    backend="md",
    options=None,
    consumer=None,
    consumer_kwargs=None,
    progress_callback=None,
    persistent=False,
):
    """
    Convert a batch of documents using the selected strategy.
    """
    if pymupdf4llm._use_layout:
        import onnxruntime  # pylint: disable=unused-import
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("inputs must be a list or tuple")

    # Load default options if none provided
    if not options:
        options = copy.deepcopy(_default_options(backend))

    # Layout engine flag
    options["layout_enabled"] = pymupdf4llm._use_layout

    _validate_image_output_options(out_dir, options)

    if workers is None:
        workers = auto_workers(
            file_count=len(inputs),
            user_dpi=options.get("ocr_dpi", 150),
            ocr_mode=options.get("use_ocr", OCRMode.SELECT_KEEP_OLD),
            plugin=options.get("ocr_function"),
        )

    # Select strategy
    strategy = _select_strategy(
        workers=workers,
        streaming=streaming,
        persistent=persistent,
    )

    # Execute strategy
    results = strategy.run(
        inputs,
        backend,
        options,
        consumer,
        consumer_kwargs,
        out_dir,
        progress_callback,
    )

    # Streaming returns a generator
    if streaming:
        return results

    # Non-streaming returns a list
    return list(results)
