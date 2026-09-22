import os

import psutil
from pymupdf4llm.ocr import OCRMode


def _is_tesseract_plugin(plugin):
    if isinstance(plugin, str):
        return plugin.lower().startswith("tesseract")

    module_name = getattr(plugin, "__module__", "")
    func_name = getattr(plugin, "__name__", "")
    text = f"{module_name}.{func_name}".lower()
    return "tesseract" in text


def auto_workers(file_count, user_dpi, ocr_mode, plugin=None):
    """Compute a safe worker-process count from CPU/RAM and OCR settings."""
    total_ram = psutil.virtual_memory().total
    total_ram_gb = total_ram / (1024**3)

    cpu_count = os.cpu_count() or 1
    if cpu_count == 1:
        return 1

    cpu_limit = max(1, cpu_count - 2)
    ram_limit = cpu_limit / cpu_count * total_ram_gb

    try:
        dpi = max(150, float(user_dpi))
    except (TypeError, ValueError):
        dpi = 150

    tesseract = _is_tesseract_plugin(plugin)
    ram_per_worker = 0.6 if (tesseract or ocr_mode == OCRMode.NEVER) else dpi / 150

    workers = min(file_count, cpu_limit, (ram_limit // ram_per_worker))
    return max(1, int(workers))
