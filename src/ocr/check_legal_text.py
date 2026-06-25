"""
Checks if a string contains only safe characters (Latin and CJK ranges).
We do not want to try OCR on text that contains characters outside of these
ranges, as it is likely to be garbage.
"""
ALLOWED_MATH = {
    0x2211,  # ∑
    0x222B,  # ∫
    0x2202,  # ∂
    0x221E,  # ∞
    0x221A,  # √
    0x2248,  # ≈
    0x2260,  # ≠
    0x2264,  # ≤
    0x2265,  # ≥
}

LATIN_RANGES = [
    (0x0000, 0x007F),
    (0x0080, 0x00FF),
    (0x0100, 0x017F),
    (0x0180, 0x024F),
    (0x1E00, 0x1EFF),
    (0x2C60, 0x2C7F),
    (0xA720, 0xA7FF),
    (0xAB30, 0xAB6F),
]

CJK_RANGES = [
    (0x3000, 0x303F),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xFF00, 0xFFEF),
]

def in_ranges(cp, ranges):
    return any(start <= cp <= end for start, end in ranges)

def is_safe_char(ch):
    cp = ord(ch)
    if cp == 0xFFFD:  # replacement char allowed
        return True
    if cp in ALLOWED_MATH:
        return True
    return in_ranges(cp, LATIN_RANGES) or in_ranges(cp, CJK_RANGES)

def contains_unsafe(text):
    return any(not is_safe_char(ch) for ch in text)

