import enum
class OCRMode(enum.IntEnum):
    """
    Enum representing different OCR modes.
    Each mode has an integer value and a description.
    """

    NEVER = (0, "Never run OCR")
    SELECT_DROP_OLD = (1, "Run OCR when needed, remove previous OCR text")
    SELECT_KEEP_OLD = (2, "Run OCR when needed, preserve previous OCR text")
    FORCE_DROP_OLD = (3, "Run OCR for all pages, remove previous OCR text")
    FORCE_KEEP_OLD = (4, "Run OCR for all pages, preserve previous OCR text")
    
    SELECT_REMOVING_OLD = SELECT_DROP_OLD
    SELECT_PRESERVING_OLD = SELECT_KEEP_OLD
    ALWAYS_REMOVING_OLD = FORCE_DROP_OLD
    ALWAYS_PRESERVING_OLD = FORCE_KEEP_OLD

    def __new__(cls, value, description):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    def __init__(self, value, description):
        self.__doc__ = description

