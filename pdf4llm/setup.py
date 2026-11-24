import os
import sys

import pipcl

version = '2.2'

def build():
    return [
            ('pdf4llm/__init__.py', 'pdf4llm/'),
            ]

def sdist():
    root = os.path.abspath( f'{__file__}/..')
    return pipcl.git_items(root)
        
p = pipcl.Package(
        'pdf4llm',
        version,
        author = 'Artifex',
        author_email = 'support@artifex.com',
        classifier = [
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3',
            'Topic :: Utilities',
            ],
        description = 'README.md',
        description_content_type = 'text/markdown',
        license = 'Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License',
        project_url = [
            ('Documentation, https://pymupdf.readthedocs.io/'),
            ('Source, https://github.com/pymupdf/pymupdf'),
            ('Tracker, https://github.com/pymupdf/PyMuPDF/issues'),
            ('Changelog, https://pymupdf.readthedocs.io/en/latest/changes.html'),
            ],
        pure = True,
        requires_dist = f'pymupdf=={version}',
        requires_python = '>=3.11',
        summary = 'PyMuPDF Utilities for LLM/RAG',
        
        fn_build=build,
        fn_sdist=sdist,
        )

build_wheel = p.build_wheel
build_sdist = p.build_sdist

if __name__ == '__main__':
    p.handle_argv(sys.argv)
