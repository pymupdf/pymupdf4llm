import pymupdf.layout

def test_simple():
    l = pymupdf.layout.features_test('foo')
    assert l == 3
