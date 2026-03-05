import subprocess
import sys

import pymupdf
import pymupdf4llm


def test_layout_switch():
    '''
    Check that we can activate/deactivate use of layout.
    '''

    pymupdf4llm.use_layout(True)
    assert pymupdf._get_layout

    pymupdf4llm.use_layout(False)
    assert not pymupdf._get_layout

    pymupdf4llm.use_layout(True)
    assert pymupdf._get_layout


def test_layout_default():
    '''
    Check that we use layout by default.
    '''
    command = f'{sys.executable} -c "import pymupdf; import pymupdf4llm; assert pymupdf._get_layout"'
    subprocess.run(command, shell=1, check=1)
