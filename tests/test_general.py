import difflib
import glob
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.abspath(f'{__file__}/../..'))
try:
    import pipcl
finally:
    del sys.path[0]

import pymupdf
import pymupdf.features
import pymupdf.layout


g_root = os.path.abspath(f'{__file__}/../..')

def test_simple():
    l = pymupdf.features.features_test('foo')
    assert l == 3

def test_activate():
    pymupdf.layout.activate()

def test_features():
    print()
    rect = pymupdf.mupdf.FzRect(pymupdf.mupdf.FzRect.Fixed_INFINITE)
    stext_page = pymupdf.mupdf.FzStextPage(rect)    # mediabox
    region = pymupdf.mupdf.FzRect(0, 0, 100, 100)
    features = pymupdf.features.fz_features_for_region(stext_page, region, 0)
    print(f'{features=}:')
    for name in dir(features):
        if not name.startswith('_') and name != 'this':
            print(f'    {name}: {getattr(features, name)!r}')


def test_competitor_examples():
    '''
    Check output for documents in repository CompetitorExamples.
    
    Our behaviour depends on environment variable
    PYMUPDF_LAYOUT_test_competitor_examples_dir:
    
    * If unset: we git clone
      git@github.com:pymupdf/CompetitorExamples.git. This will fail if we do
      not have appropriate credentials.
      
    * If empty string, we do nothing.
    
    * Otherwise it is assumed to be a local checkout.
    '''
    print(f'test_competitor_examples(): currently disabled.')
    return
    
    import pymupdf4llm

    PYMUPDF_LAYOUT_test_competitor_examples_dir = os.environ.get('PYMUPDF_LAYOUT_test_competitor_examples_dir')
    AUDITWHEEL_PLAT = os.environ.get('AUDITWHEEL_PLAT')
    if PYMUPDF_LAYOUT_test_competitor_examples_dir == '':
        print(f'test_competitor_examples(): not running because {PYMUPDF_LAYOUT_test_competitor_examples_dir=}.')
        return
    elif AUDITWHEEL_PLAT and AUDITWHEEL_PLAT.startswith('manylinux_'):
        # manylinux does not have ssh.
        print(f'test_competitor_examples(): not running because {AUDITWHEEL_PLAT=}.')
        return
    elif PYMUPDF_LAYOUT_test_competitor_examples_dir is None:
        ced = pipcl.git_get(
                'git-competitor_examples',
                remote='git@github.com:pymupdf/CompetitorExamples.git',
                branch='main',
                )
    else:
        ced = PYMUPDF_LAYOUT_test_competitor_examples_dir
    print()
    ced_abs = os.path.abspath(ced)
    pattern = f'{ced_abs}/_docs/*'
    print(f'{ced=}.')
    print(f'{ced_abs=}.')
    print(f'{pattern=}.')
    paths = list(sorted(glob.glob(pattern)))
    assert paths, f'No paths found in {pattern=}.'
    for path in paths:
        print(f'test_competitor_examples(): processing {path=}.', flush=1)
        with pymupdf.open(path) as document:
            expected_path = f'{g_root}/tests/competitor_examples_{os.path.basename(path)}.md'
            try:
                with open(expected_path) as f:
                    md_expected = f.read()
            except Exception:
                print(f'Cannot read {expected_path}.')
                md_expected = None
            t0 = time.time()
            md = pymupdf4llm.to_markdown(document, table_strategy='lines')
            t = time.time() - t0
            def print_unicode_safe(text, indent='', errors='backslashreplace'):
                '''
                Like print() but allows indentation and control of encoding
                errors.

                Useful on Windows to avoid exceptions if we attempt to print
                unicode characters that cannot be represented in Windows'
                non-utf8 encoding.
                '''
                if indent:
                    text = textwrap.indent(text, indent)
                if not text.endswith('\n'):
                    text += '\n'
                text_b = text.encode(sys.stdout.encoding, errors=errors)
                sys.stdout.flush()
                sys.stdout.buffer.write(text_b)
            print(f'test_competitor_examples(): {t=}', flush=1)
            if md_expected:
                print(f'md_expected:')
                print_unicode_safe(md_expected, indent='    ')
            print(f'md:')
            print_unicode_safe(md, indent='    ')
            if md_expected:
                if md != md_expected:
                    lines = difflib.unified_diff(
                            md_expected.split('\n'),
                            md.split('\n'),
                            fromfile='md_expected',
                            tofile='md',
                            lineterm='',
                            )
                    print_unicode_safe('\n'.join(lines), indent='    ')
                    
                assert md == md_expected
            else:
                # We need to specify utf8 on Windows, otherwise default
                # encoding can fail to handle some characters.
                with open(expected_path, 'w', errors='backslashreplace') as f:
                    f.write(md)
                print(f'test_competitor_examples(): have written to {expected_path=}.')
