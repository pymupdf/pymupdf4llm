import pymupdf
import pymupdf.features
import pymupdf.layout


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
