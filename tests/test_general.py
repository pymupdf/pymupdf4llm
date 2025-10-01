import pymupdf.features
import pymupdf.layout


def test_simple():
    l = pymupdf.features.features_test('foo')
    assert l == 3

def test_activate():
    pymupdf.layout.activate()
