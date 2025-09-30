import pymupdf.features

def test_simple():
    l = pymupdf.features.features_test('foo')
    assert l == 3
