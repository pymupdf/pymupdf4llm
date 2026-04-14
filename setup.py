import os
import sys
import textwrap

import pipcl


VERSION = '1.27.2.2'
VERSION_TUPLE = tuple(int(x) for x in VERSION.split("."))

# We build with, and run with, a particular PyMuPDF version usually, but not
# always, the same as our version.
#
pymupdf_version = VERSION

# We build with, and run with, a particular pymupdf_layout version usually, but
# not always, the same as our version.
#
pymupdf_layout_version = VERSION


PYMUPDF_SETUP_VERSION = os.environ.get('PYMUPDF_SETUP_VERSION')
if PYMUPDF_SETUP_VERSION:
    # Allow testing with non-matching pymupdf/layout versions.
    requires_dist = ["tabulate"]
else:
    requires_dist = [
            f"pymupdf=={pymupdf_version}",
            f"pymupdf_layout=={pymupdf_layout_version}",
            "tabulate",
            ]


def build():
    ret = list()
    
    version_info = textwrap.dedent(f'''
            # Generated file - do not edit.
            {VERSION=}
            {VERSION_TUPLE=}
            ''')
    ret.append((version_info.encode('utf-8'), 'pymupdf4llm/versions_file.py'))
    
    for p in pipcl.git_items('src'):
        ret.append((f'src/{p}', f'pymupdf4llm/{p}'))
    
    print(f'ret:')
    for i in ret:
        print(f'    {i}')
    return ret


def sdist():
    return pipcl.git_items('.')


p = pipcl.Package(
        'pymupdf4llm',
        VERSION,
        requires_dist=requires_dist,
        requires_python='>=3.10',
        pure=True,
        author="Artifex",
        author_email="support@artifex.com",
        summary='PyMuPDF Utilities for LLM/RAG',
        description='README.md',
        description_content_type='text/markdown',
        classifier = [
                'Development Status :: 5 - Production/Stable',
                'Environment :: Console',
                'Intended Audience :: Developers',
                'Programming Language :: Python :: 3',
                'Topic :: Utilities',
                ],
        license = 'Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License',
        project_url = [
                'Documentation, https://pymupdf.readthedocs.io/',
                'Source, https://github.com/pymupdf/pymupdf4llm',
                'Tracker, https://github.com/pymupdf/PyMuPDF/issues',
                'Changelog, https://pymupdf.readthedocs.io/en/latest/changes.html',
                ],
        fn_build = build,
        fn_sdist = sdist,
        )
        
build_wheel = p.build_wheel
        
if __name__ == '__main__':
    p.handle_argv(sys.argv)
