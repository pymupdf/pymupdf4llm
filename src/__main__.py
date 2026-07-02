import argparse
import importlib.util
from pathlib import Path

import pymupdf4llm
from pymupdf4llm import convert_batch
from pymupdf4llm.ocr import OCRMode
from pymupdf4llm.worker_sizing import auto_workers

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

OCR_MODE_MAP = {
    "never": OCRMode.NEVER,
    "select-drop": OCRMode.SELECT_DROP_OLD,
    "select-keep": OCRMode.SELECT_KEEP_OLD,
    "force-drop": OCRMode.FORCE_DROP_OLD,
    "force-keep": OCRMode.FORCE_KEEP_OLD,
}


def _collect_inputs(path, pattern):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(p.glob(pattern))


def _parse_key_value_pairs(pairs):
    """Parse --opt key=value into a dict."""
    opts = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Invalid option backend: {pair}. Expected key=value.")
        key, value = pair.split("=", 1)

        # Convert booleans and numbers
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        opts[key] = value
    return opts


def main():
    parser = argparse.ArgumentParser(
        description="Batch Convert Documents with PyMuPDF4LLM"
    )

    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes, default: automatically determined based on CPU cores and input file count.",
    )
    parser.add_argument(
        "--backend",
        default="md",
        choices=["md", "json", "txt"],
        help="Output format (default: md - Markdown)",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for input files (default: *.pdf)",
    )

    # Markdown / JSON / TXT options
    parser.add_argument(
        "--opt",
        action="append",
        help="Additional conversion options (key=value). "
        "Example: --opt page_separators=true --opt ocr_dpi=200 --opt image_dpi=200...",
    )

    # Common high-level flags (optional convenience)
    parser.add_argument(
        "--ocr-mode",
        choices=[
            "never",
            "select-drop",
            "select-keep",
            "force-drop",
            "force-keep",
        ],
        default="select-keep",
        help="OCR handling: *-keep will skip OCR if an old text layer is present, *-drop will discard old text layers, select-* will analyze pages and pass them to OCR (default: select-keep).",
    )
    parser.add_argument(
        "--ocr-func",
        default="auto",
        help="OCR function to use (default: auto). Strings 'tesseract', 'rapidocr', 'rapidtess' etc. invoke built-in engines. Other strings are used to import a Python file that must have a function called 'exec_ocr'.",
    )
    parser.add_argument(
        "--ocr-lang", default="eng", help="Language for OCR (default: eng)"
    )
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Write images to disk (default: False)",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed images in the output (default: False)",
    )
    parser.add_argument(
        "--layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Layout mode for conversion (default: True). Use --no-layout to disable.",
    )
    parser.add_argument(
        "--pro",
        default=False,
        help="Use PyMuPDF-Pro for extended Office support (default: False). Requires a valid license key for full functionality.",
    )
    parser.add_argument(
        "--pro-key",
        default=None,
        help="License key for PyMuPDF-Pro (default: None). Required for extended Office support.",
    )

    parser.add_argument(
        "--header",
        action="store_true",
        help="Include page headers in the output (default: False)",
    )
    parser.add_argument(
        "--footer",
        action="store_true",
        help="Include page footers in the output (default: False)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress during conversion (default: False). Ignored if 'tqdm' is unavailable.",
    )

    args = parser.parse_args()

    # Collect inputs
    inputs = _collect_inputs(args.input, args.pattern)
    if not inputs:
        if isinstance(args.input, str):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        else:
            raise ValueError(
                f"No input files found in {args.input} matching {args.pattern}"
            )
    input_count = len(inputs)
    # Build options dict
    options = _parse_key_value_pairs(args.opt)

    # Add convenience flags
    options["use_ocr"] = OCR_MODE_MAP[args.ocr_mode]
    options["write_images"] = True if args.write_images else False
    options["embed_images"] = True if args.embed_images else False
    options["header"] = True if args.header else False
    options["footer"] = True if args.footer else False
    options["ocr_lang"] = args.ocr_lang
    if options["write_images"] and options["embed_images"]:
        raise ValueError("Cannot do both: write and embed images.")
    if options["use_ocr"] == OCRMode.NEVER:
        ocr_function = None
        args.ocr_func = None
    if args.ocr_func == "tesseract":
        from pymupdf4llm.ocr import tesseract_api

        ocr_function = tesseract_api.exec_ocr
    elif args.ocr_func == "rapidocr":
        from pymupdf4llm.ocr import rapidocr_api

        ocr_function = rapidocr_api.exec_ocr
    elif args.ocr_func == "rapidtess":
        from pymupdf4llm.ocr import rapidtess_api

        ocr_function = rapidtess_api.exec_ocr
    elif args.ocr_func == "paddleocr":
        from pymupdf4llm.ocr import paddleocr_api

        ocr_function = paddleocr_api.exec_ocr
    elif args.ocr_func == "paddletess":
        from pymupdf4llm.ocr import paddletess_api

        ocr_function = paddletess_api.exec_ocr

    elif isinstance(args.ocr_func, str) and args.ocr_func.endswith(".py"):
        # Import a custom OCR function from a Python file
        spec = importlib.util.spec_from_file_location("custom_ocr", args.ocr_func)
        custom_ocr_module = importlib.util.module_from_spec(spec)
        # sys.modules["custom_ocr"] = custom_ocr_module
        spec.loader.exec_module(custom_ocr_module)
        if hasattr(custom_ocr_module, "exec_ocr"):
            ocr_function = custom_ocr_module.exec_ocr
        else:
            raise ValueError(
                f"Custom OCR module {args.ocr_func} does not have an 'exec_ocr' function."
            )
    else:
        ocr_function = None
    options["ocr_function"] = ocr_function

    pymupdf4llm.use_layout(args.layout)

    if args.pro:
        import pymupdf.pro

        if isinstance(args.pro_key, str):
            pymupdf.pro.unlock(args.pro_key)
        else:
            pymupdf.pro.unlock()

    workers = args.workers
    if workers is None:
        # Auto-determine number of workers based on CPU cores and input file count
        workers = auto_workers(
            file_count=input_count,
            user_dpi=options.get("ocr_dpi", 150),
            ocr_mode=options["use_ocr"],
            plugin=args.ocr_func,
        )
        worker_msg = f"Workers used: {workers} (auto)"
    else:
        workers = max(1, workers)
        worker_msg = f"Workers used: {workers} (user-specified)"

    # Run batch conversion
    from datetime import datetime

    # Start time
    start = datetime.now()

    # Progress bar
    if args.show_progress and tqdm is not None and len(inputs) > 1:
        pbar = tqdm(total=len(inputs), desc="Converting", unit="file")
    else:
        pbar = None

    def progress_callback(done, total):
        if pbar is not None:
            pbar.update(1)

    # Batch execution
    results = convert_batch(
        inputs,
        workers=workers,
        out_dir=args.out,
        backend=args.backend,
        options=options,
        consumer=None,
        consumer_kwargs=None,
        progress_callback=progress_callback,
    )

    if pbar is not None:
        pbar.close()

    # End time
    end = datetime.now()
    delta = end - start

    # Wall clock in minutes:seconds
    minutes, seconds = divmod(delta.total_seconds(), 60)
    wallclock = f"{int(minutes)}:{int(seconds):02d}"

    # Number of files
    count = len(results)

    print(f"\nBatch completed.")
    print(worker_msg)
    print(f"Processed files: {count}")
    print(f"Duration: {wallclock} (mm:ss)")


if __name__ == "__main__":
    main()
