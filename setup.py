#! /usr/bin/env python3

'''
Environment variables:

    SCE_SETUP_BUILD_PYMUPDF
        If set to none-empty string, we expect PyMuPDF to be already installed
        and we don't return it from get_requires_for_build_wheel().
    
    SCE_SETUP_BUILD_TYPE
        We do a debug build if set to 'debug'.
    
    SCE_SETUP_VSGRADE
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
g_version = '1.26.4'

# We build and run with PyMuPDF version equal to the first three numbers of our
# version.
#
# This allows us to make multiple releases for the same PyMuPDF if
# required. For example pymupdfsce-1.24.8 and pymupdfsce-1.24.8.1 would both
# use PyMuPDF-1.24.8.
#
g_pymupdf_version = '.'.join(g_version.split('.')[:3])

SCE_SETUP_BUILD_PYMUPDF = os.environ.get('SCE_SETUP_BUILD_PYMUPDF')
if SCE_SETUP_BUILD_PYMUPDF:
    # We are building with custom pymupdf, so don't list a pymupdf version as a
    # buildtime or runtime prerequisite.
    log(f'Setting g_pymupdf_version=None because {SCE_SETUP_BUILD_PYMUPDF=}.')
    g_pymupdf_version = None


def build():

    root = os.path.normpath(f'{__file__}/..')
    root = pipcl.relpath(root, allow_up=0)
    
    SCE_SETUP_BUILD_TYPE = os.environ.get('SCE_SETUP_BUILD_TYPE')
    
    SCE_SETUP_VSGRADE = os.environ.get('SCE_SETUP_VSGRADE')
    if SCE_SETUP_VSGRADE is not None:
        assert SCE_SETUP_VSGRADE in ('Community', 'Professional', 'Enterprise'), \
            f'{SCE_SETUP_VSGRADE=} should undefined or one of Community, Professional, Enterprise.'
    
    if SCE_SETUP_VSGRADE is None:
        GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS')
        if GITHUB_ACTIONS=='true':  
            # We are running as a Github action, which has VS Enterprise.
            SCE_SETUP_VSGRADE = 'Enterprise'
            log(f'{GITHUB_ACTIONS=} so defaulting to {SCE_SETUP_VSGRADE=}.')
        else:
            SCE_SETUP_VSGRADE = 'Professional'
            log(f'Defaulting to {SCE_SETUP_VSGRADE=}.')
    
    # We use the installed PyMuPDF's embedded MuPDF include and lib
    # directories.
    import pymupdf
    p = os.path.normpath(f'{pymupdf.__file__}/..')
    if pymupdf.pymupdf_version_tuple >= (1, 26, 6):
        mupdf_include, mupdf_lib = pymupdf._mupdf_devel()
    else:
        # pymupdf._mupdf_devel() not available so we do things here.
        mupdf_include = f'{p}/mupdf-devel/include'
        if platform.system() == 'Windows':
            # Separate .lib files are used at build time.
            mupdf_lib = f'{p}/mupdf-devel/lib'
        else:
            # .so files are used for both buildtime and runtime linking.
            mupdf_lib = p
            # Make symbolic links within the installed pymupdf module so
            # that ld can find libmupdf.so etc. This is a bit of a hack, but
            # necessary because wheels cannot contain symbolic links.
            #
            # For example we create `libmupdf.so -> libmupdf.so.24.8`.
            #
            # We are careful to only create symlinks for the expected MuPDF
            # version, in case old .so files from a previous install are still
            # in place.
            #
            log(f'Creating symlinks in {mupdf_lib=} for MuPDF-{pymupdf.mupdf_version} .so files.')
            regex_suffix = pymupdf.mupdf_version.split('.')[1:3]
            regex_suffix = '[.]'.join(regex_suffix)
            mupdf_lib_regex = f'^(lib[^.]+[.]so)[.]{regex_suffix}$'
            log(f'{mupdf_lib_regex=}.')
            for leaf in os.listdir(mupdf_lib):
                m = re.match(mupdf_lib_regex, leaf)
                if m:
                    pfrom = f'{mupdf_lib}/{m.group(1)}'
                    # os.path.exists() can return false if softlink exists
                    # but points to non-existent file, so we also use
                    # `os.path.islink()`.
                    if os.path.islink(pfrom) or os.path.exists(pfrom):
                        log(f'Removing existing link {pfrom=}.')
                        os.remove(pfrom)
                    log(f'Creating symlink: {pfrom} -> {leaf}')
                    os.symlink(leaf, pfrom)

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
    
        vs = pipcl.wdev.windows_vs_multiple(year=2022, grade=SCE_SETUP_VSGRADE)
        if not vs:
            log(f'Warning, could not find Visual Studio 2022 matching {SCE_SETUP_VSGRADE=}.')
            log(f'Consider setting SCE_SETUP_VSGRADE to a match in the above list.')
    
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
            'layout',
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
            debug=(SCE_SETUP_BUILD_TYPE == 'debug'),
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
            (f'{build_dir}/layout.py', to_dir),
            (f'{build_dir}/{sharedlibrary_leaf}', to_dir),
            (build_py.encode(), f'{to_dir}/_layout_build.py'),
            ]
            
    return ret


def sdist():
    root = os.path.normpath(f'{__file__}/..')
    return pipcl.git_items(root)


# Define PyMuPDFPro package.
#
p = pipcl.Package(
        'pymupdflayout',
        g_version,
        requires_dist = [
                f'PyMuPDF=={g_pymupdf_version}' if g_pymupdf_version else None,
                ],
        summary = 'Commercial extension for PyMuPDF',
        description = 'README.md',
        description_content_type = 'text/markdown',
        license = 'Commercial license. See artifex.com for details.',
        classifier = [
                'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Developers',
                'Intended Audience :: Information Technology',
                'License :: Other/Proprietary License',
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
        requires_python = '>=3.9',
        fn_build = build,
        fn_sdist = sdist,
        py_limited_api = True,
        )


build_wheel = p.build_wheel
build_sdist = p.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    ret = list()
    ret.append('swig')
    if SCE_SETUP_BUILD_PYMUPDF:
        log(f'Not requiring default pymupdf=={g_pymupdf_version} because {SCE_SETUP_BUILD_PYMUPDF=}.')
    else:
        ret.append(f'pymupdf=={g_pymupdf_version}')
    ret.append(f'swig')
    log(f'get_requires_for_build_wheel(): returning: {ret=}')
    return ret


if __name__ == '__main__':
    p.handle_argv(sys.argv)
