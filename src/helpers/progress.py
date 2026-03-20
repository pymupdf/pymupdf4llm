"""
This script defines a text-based progress bar to allow watching the advancement
of Markdown conversion of document pages.
 
Dependencies
-------------
None

Copyright and License
----------------------
Copyright 2024 Artifex Software, Inc.
License GNU Affero GPL 3.0
"""

import sys
from typing import Any, List


class _ProgressBar:
    def __init__(self, items: List[Any], progress_width: int = 40):
        self._len = len(items)
        self._iter = iter(items)
        self._len_digits = len(str(self._len))
        self._progress_width = progress_width
        self._progress_bar = 0
        self._current_index = 0

        # Calculate the increment for each item based on the list length and the progress width
        self._increment = self._progress_width / self._len if self._len else 1

        # Init progress bar
        sys.stdout.write("[%s] (0/%d)" % (" " * self._progress_width, self._len))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self._progress_width + len(str(self._len)) + 6))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = next(self._iter)
        except StopIteration as e:
            # End progress on StopIteration
            sys.stdout.write("]\n")
            raise e

        # Update the current index
        self._current_index += 1

        # Add the increment to the progress bar and calculate how many "=" to add
        self._progress_bar += self._increment
        while self._progress_bar >= 1:
            sys.stdout.write("=")
            sys.stdout.flush()
            self._progress_bar -= 1

        # Update the numerical progress
        padded_index = str(self._current_index).rjust(self._len_digits)
        progress_info = f" ({padded_index}/{self._len})"
        sys.stdout.write("\b" * (self._progress_width + len(progress_info) + 1))
        sys.stdout.write("[")
        sys.stdout.write(
            "=" * int(self._current_index * self._progress_width / self._len)
        )
        sys.stdout.write(
            " "
            * (
                self._progress_width
                - int(self._current_index * self._progress_width / self._len)
            )
        )
        sys.stdout.write("]" + progress_info)
        sys.stdout.flush()
        sys.stdout.write(
            "\b"
            * (
                self._progress_width
                - int(self._current_index * self._progress_width / self._len)
                + len(progress_info)
                + 1
            )
        )

        return result


def ProgressBar(list: List[Any], progress_width: int = 40):
    return iter(_ProgressBar(list, progress_width))
