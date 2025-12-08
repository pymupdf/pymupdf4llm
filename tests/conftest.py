import os
import platform
import subprocess
import sys

import pytest


# Install required packages. There doesn't seem to be any official way for
# us to programmatically specify required test packages in setup.py, or in
# pytest. Doing it here seems to be the least ugly approach.
#
def install_required_packages():
    packages = 'pymupdf4llm opencv-python'
    command = f'pip install {packages}'
    print(f'{__file__}:install_required_packages)(): Running: {command}', flush=1)
    subprocess.run(command, shell=1, check=1)

install_required_packages()
