#! /usr/bin/env python3

'''
Environment variables:

    PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF
        If set to none-empty string, we expect PyMuPDF to be already installed
        and we don't return it from get_requires_for_build_wheel().
    
    PYMUPDF_LAYOUT_SETUP_SWIG
        If set, we use this instead of `swig`.
    
    PYMUPDF_LAYOUT_SETUP_BUILD_TYPE
        We do a debug build if set to 'debug'.
    
    PYMUPDF_LAYOUT_SETUP_VSGRADE
        Specific Visual Studio, must be one of 'Community', 'Professional',
        'Enterprise'.
'''

import pipcl
import wdev

import glob
import inspect
import os
import platform
import re
import sys
import textwrap


run = pipcl.run
log = pipcl.log1

g_root = os.path.abspath(f'{__file__}/..')

if 1:
    # For debugging.
    log(f'### Starting.')
    
    log(f'{__file__=}')
    log(f'{__name__=}')
    pipcl.show_system()


# Define our package version number.
#
# The first three digits determine the PyMuPDF version that we build for, and
# which we list as a runtime requirement.
#
# Use a fourth digit if making multiple releases of PyMuPDFPro that are to be
# used with the same version of PyMuPDF, for example `1.24.9.1`.
#
g_version = '1.26.6'

# We build and run with PyMuPDF version equal to the first three numbers of our
# version.
#
# This allows us to make multiple releases for the same PyMuPDF if
# required. For example pymupdfsce-1.24.8 and pymupdfsce-1.24.8.1 would both
# use PyMuPDF-1.24.8.
#
g_pymupdf_version = '.'.join(g_version.split('.')[:3])

PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF = os.environ.get('PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF')
if PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF:
    # We are building with custom pymupdf, so don't list a pymupdf version as a
    # buildtime or runtime prerequisite.
    log(f'Setting g_pymupdf_version=None because {PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF=}.')
    g_pymupdf_version = None

PYMUPDF_LAYOUT_SETUP_SWIG = os.environ.get('PYMUPDF_LAYOUT_SETUP_SWIG')

def build():

    root = os.path.normpath(f'{__file__}/..')
    root = pipcl.relpath(root, allow_up=0)
    
    swig = PYMUPDF_LAYOUT_SETUP_SWIG or 'swig'
    run(f'{swig} -version')
    
    PYMUPDF_LAYOUT_SETUP_BUILD_TYPE = os.environ.get('PYMUPDF_LAYOUT_SETUP_BUILD_TYPE')
    
    PYMUPDF_LAYOUT_SETUP_VSGRADE = os.environ.get('PYMUPDF_LAYOUT_SETUP_VSGRADE')
    if PYMUPDF_LAYOUT_SETUP_VSGRADE is not None:
        assert PYMUPDF_LAYOUT_SETUP_VSGRADE in ('Community', 'Professional', 'Enterprise'), \
            f'{PYMUPDF_LAYOUT_SETUP_VSGRADE=} should undefined or one of Community, Professional, Enterprise.'
    
    if PYMUPDF_LAYOUT_SETUP_VSGRADE is None:
        GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS')
        if GITHUB_ACTIONS=='true':  
            # We are running as a Github action, which has VS Enterprise.
            PYMUPDF_LAYOUT_SETUP_VSGRADE = 'Enterprise'
            log(f'{GITHUB_ACTIONS=} so defaulting to {PYMUPDF_LAYOUT_SETUP_VSGRADE=}.')
        else:
            PYMUPDF_LAYOUT_SETUP_VSGRADE = 'Professional'
            log(f'Defaulting to {PYMUPDF_LAYOUT_SETUP_VSGRADE=}.')
    
    # We use the installed PyMuPDF's embedded MuPDF include and lib
    # directories.
    import pymupdf
    p = os.path.normpath(f'{pymupdf.__file__}/..')
    mupdf_include, mupdf_lib = pymupdf._mupdf_devel()

    log(f'Within installed PyMuPDF:')
    log(f'    {mupdf_include=}')
    log(f'    {mupdf_lib=}')
    
    assert os.path.isdir(mupdf_include), f'Not a directory: {mupdf_include=}.'
    assert os.path.isdir(mupdf_lib), f'Not a directory: {mupdf_lib=}.'
    
    build_dir = f'{root}/build'
    os.makedirs(build_dir, exist_ok=1)
    
    # Show what VS installations are available.
    #
    if pipcl.windows():
        vss = pipcl.wdev.windows_vs_multiple()
        log(f'Available Visual studio installations are ({len(vss)}):')
        for i, vs in enumerate(vss):
            log(f'{i}:')
            log(f'{vs.description_ml("    ")}')
    
        vs = pipcl.wdev.windows_vs_multiple(year=2022, grade=PYMUPDF_LAYOUT_SETUP_VSGRADE)
        if not vs:
            log(f'Warning, could not find Visual Studio 2022 matching {PYMUPDF_LAYOUT_SETUP_VSGRADE=}.')
            log(f'Consider setting PYMUPDF_LAYOUT_SETUP_VSGRADE to a match in the above list.')
    
    # Build sce module.
    #
    if pipcl.windows():
        libpaths = None
        libs = None
        linker_extra = f'{pipcl.relpath(mupdf_lib, allow_up=0)}/mupdfcpp64.lib'
    else:
        libpaths = [build_dir, mupdf_lib]
        libs=['mupdf', 'mupdfcpp']
        linker_extra = None
    
    sharedlibrary_leaf = pipcl.build_extension(
            'features',
            f'{root}/source/features.i',
            source_extra=f'{root}/source/features.c',
            outdir=build_dir,
            includes=[
                    f'{root}/source',
                    pipcl.relpath(mupdf_include, allow_up=0),
                    ],
            libpaths=libpaths,
            libs=libs,
            linker_extra=linker_extra,
            py_limited_api=1,
            debug=(PYMUPDF_LAYOUT_SETUP_BUILD_TYPE == 'debug'),
            optimise=(PYMUPDF_LAYOUT_SETUP_BUILD_TYPE != 'debug'),
            swig=PYMUPDF_LAYOUT_SETUP_SWIG,
            )
    
    # Create text for _layout_build.py with build-time information.
    build_py = ''
    sha, comment, diff, branch = pipcl.git_info(g_root)
    build_py += f'version = {g_version!r}\n'
    build_py += f'git_sha = {sha!r}\n'
    build_py += f'platform_python_implementation = {platform.python_implementation()!r}\n'
    # Don't show details.
    #build_py += f'git_branch = {branch!r}\n'
    #build_py += f'git_comment = {comment!r}\n'
    #build_py += f'git_diff = {diff!r}\n'
    
    # We returns source/destination files to install or put into a wheel.
    #
    to_dir = 'pymupdf/'
    ret = [
            (f'{build_dir}/features.py', to_dir),
            (f'{build_dir}/{sharedlibrary_leaf}', to_dir),
            (build_py.encode(), f'{to_dir}_features_build.py'),
            ]
    for p in pipcl.git_items(f'{g_root}/source/layout'):
        ret.append( (f'{g_root}/source/layout/{p}', f'{to_dir}layout/{p}'))
    
    for f, t in ret:
        log(f'    {f=} {t=}')
    return ret


# Define PyMuPDF-layout package.
#
p = pipcl.Package(
        'pymupdf-layout',
        g_version,
        requires_dist = [
                f'PyMuPDF=={g_pymupdf_version}' if g_pymupdf_version else None,
                'pyyaml',
                'numpy',
                'onnxruntime',
                'networkx',
                ],
        summary = 'PyMuPDF Layout turns PDFs into structured data 10× faster than vision-based tools using AI trained on PDF internals, not images. CPU-only. No GPU required.',
        description = 'README.md',
        description_content_type = 'text/markdown',
        license = 'Dual Licensed - Polyform Noncommercial or Artifex Commercial License',
        project_url = [
                ('Documentation, https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/')
                ],
        classifier = [
                'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Developers',
                'Intended Audience :: Information Technology',
                'License :: Dual Licensed - Polyform Noncommercial or Artifex Commercial License',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: MacOS',
                'Operating System :: POSIX :: Linux',
                'Programming Language :: C',
                'Programming Language :: C++',
                'Programming Language :: Python :: 3 :: Only',
                'Programming Language :: Python :: Implementation :: CPython',
                'Topic :: Utilities',
                'Topic :: Multimedia :: Graphics',
                'Topic :: Software Development :: Libraries',
                ],
        author = 'Artifex',
        author_email = 'support@artifex.com',
        requires_python = '>=3.10',
        fn_build = build,
        py_limited_api = True,
        )


build_wheel = p.build_wheel
build_sdist = p.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    ret = list()
    if PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF:
        log(f'Not requiring default pymupdf=={g_pymupdf_version} because {PYMUPDF_LAYOUT_SETUP_BUILD_PYMUPDF=}.')
    else:
        ret.append(f'pymupdf=={g_pymupdf_version}')
    if PYMUPDF_LAYOUT_SETUP_SWIG:
        pass
    elif pipcl.darwin():
        # 2025-10-27: new swig-4.4.0 fails badly at runtime.
        ret.append(f'swig==4.3.1')
    else:
        ret.append('swig')
    log(f'get_requires_for_build_wheel(): returning: {ret=}')
    return ret


if __name__ == '__main__':
    p.handle_argv(sys.argv)
