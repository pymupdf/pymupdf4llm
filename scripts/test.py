#! /usr/bin/env python3

'''
Usage:

* Command line arguments are called parameters if they start with `-`,
  otherwise they are called commands.
* Parameters are evaluated first in the order that they were specified.
* Then commands are run in the order in which they were specified.
* Usually command `test` would be specified after a `build`, `install` or
  `wheel` command.
* Parameters and commands can be interleaved but it may be clearer to separate
  them on the command line.

Other:

* If we are not already running inside a Python venv, we automatically create a
  venv and re-run ourselves inside it (also see the -v option).
* Build/wheel/install commands always install into the venv.
* Tests use whatever PyMuPDF/MuPDF is currently installed in the venv.
* We run tests with pytest.

Example usage:

    scripts/test.py build test
        * Installs pymupdf from pypi.org and builds and installs pymupdf_layout.
        * Runs tests.
    
    scripts/test.py wheel test -p PyMuPDF -m mupdf
        * Builds pymupdf wheel in wheelhouse/ with specified mupdf, and
          installs.
        * Builds pymupdf_layout wheel in wheelhouse/ and installs.
        * Runs tests.

Parameters:

    -a <env_name>
        Read next space-separated argument(s) from environmental variable
        <env_name>.
        * Does nothing if <env_name> is unset.
        * Useful when running via Github action.
    
    -b <build>
        Set build type for `build` commands. `<build>` should be one of
        'release', 'debug', 'memento'.
    
    -i <package>
        Install miscellaneous package in venv before running any
        build/test/wheel/cibw commands.  We simply run `pip install
        <package>`. So for example:
        
            -i pymupdf4llm
                Install pymupdf4llm from pypi.org.
            
            -i ./pymupdf4llm
                Build and install pymupdf4llm from local checkout.
            
            -i pymupdf4llm-....whl
                Install local pymupdf4llm wheel.

    -m <mupdf>
        Specify mupdf location.
    
    -o <os_names>
        Control whether we do nothing on the current platform.
        * <os_names> is a comma-separated list of names.
        * If <os_names> is empty (the default), we always run normally.
        * Otherwise we only run if an item in <os_names> matches (case
          insensitive) platform.system().
        * For example `-o linux,darwin` will do nothing unless on Linux or
          MacOS.
    
    -p <pymupdf>
        Specify PyMuPDF location.
    
    --sync-paths <path>
        Do not run anything, instead write required files/directories/checkouts
        to <path>, one per line. This is to help with automated running on
        remote machines.
    
    -t <pytest_args>
        Extra pytest args.
    
    -T <prefix>
        Use specified prefix when running pytest, must be one of:
            gdb
            helgrind
            valgrind
    
    -v <venv>
        venv is:
        0 - do not use a venv.
        1 - Use venv. If it already exists, we assume the existing directory
            was created by us earlier and is a valid venv containing all
            necessary packages; this saves a little time.
        2 - Use venv.
        3 - Use venv but delete it first if it already exists.
        The default is 2.

Commands:
    build
        Build and install pymupdf_layout. If `-p` is specified we also build
        and install pymupdf.
    cibw
        Build/test using cibw.
    test
        Run pytest tests.
    wheel
        Build (in wheelhouse/) and install pymupdf_layout wheel. If `-p` is
        specified also build and install a pymupdf wheel.
'''

import os
import platform
import shlex
import sys


g_root_abs = os.path.abspath( f'{__file__}/../..')

try:
    sys.path.insert(0, g_root_abs)
    import pipcl
finally:
    del sys.path[0]

g_root = pipcl.relpath(g_root_abs)


def main():

    commands = list()
    env_extra = dict()
    
    build_type = None
    linux_aarch64 = False
    mupdf = None
    os_names = list()
    pymupdf = None
    pytest_args = None
    pytest_prefix = None
    sync_paths = False
    
    venv = 2
    
    args = iter(sys.argv[1:])
    while 1:
        try:
            arg = next(args)
        except StopIteration:
            break
        
        if arg in ('-h', '--help'):
            print(__doc__)
        
        elif arg == '-a':
            _name = next(args)
            _value = os.environ.get(_name, '')
            _args = shlex.split(_value) + list(args)
            args = iter(_args)
        
        elif arg == '-b':
            build_type = next(args)
            assert build_type in ('release', 'debug', 'memento')
            env_extra['PYMUPDF_SETUP_MUPDF_BUILD_TYPE'] = build_type
            env_extra['SCE_SETUP_BUILD_TYPE'] = build_type
        
        elif arg == '-e':
            _nv = next(args)
            assert '=' in _nv, f'-e <name>=<value> does not contain "=": {_nv!r}'
            _name, _value = _nv.split('=', 1)
            env_extra[_name] = _value
        
        elif arg == '-m':
            mupdf = next(args)
        
        elif arg == '-o':
            os_names += next(args).split(',')
        
        elif arg == '-p':
            pymupdf = next(args)
        
        elif arg == '-t':
            pytest_args = next(args)
        
        elif arg == '-T':
            pytest_prefix = next(args)
            assert pytest_prefix in ('gdb', 'helgrind', 'valgrind'), \
                    f'Unrecognised {pytest_prefix=}, should be one of: gdb valgrind helgrind.'
        
        elif arg == '-v':
            venv = int(next(args))
        
        elif arg == '--linux-aarch64':
             linux_aarch64 = int(next(args))
        
        elif arg == '--sync-paths':
            sync_paths = next(args)
            
        elif arg in ('build', 'cibw', 'test', 'wheel'):
            commands.append(arg)
        
        else:
            assert 0, f'Unrecognised {arg=}'
    
    # Handle special args --sync-paths, -h, -v, -o first.
    #
    if sync_paths:
        # Print required files, directories and checkouts, and exit without
        # running any commands.
        with open(sync_paths, 'w') as f:
            print(g_root, file=f)
            if mupdf and not mupdf.startswith('git:'):
                print(mupdf, file=f)
            if pymupdf and not pymupdf.startswith('git:'):
                print(pymupdf, file=f)
        return

    if os_names:
        if platform.system().lower() not in os_names:
            pipcl.log(f'Not running because {platform.system().lower()=} not in {os_names=}')
            return
    
    if commands:
        if venv:
            # Rerun ourselves inside a venv if not already in a venv.
            if not pipcl.venv_in():
                venv_name = f'venv-sce-{platform.python_version()}-{int.bit_length(sys.maxsize+1)}'
                e = pipcl.venv_run(
                        sys.argv,
                        venv_name,
                        recreate=(venv>=2),
                        clean=(venv>=3),
                        )
                sys.exit(e)
    else:
        pipcl.log(f'Warning, no commands specified so nothing to do.')
    
    for command in commands:
        no_build_isolation = ''
        if command in ('build', 'wheel'):
            # Install pymupdf in current venv and specify
            # `--no-build-isolation` so pip builds sce in the current venv.
            #
            if mupdf:
                assert pymupdf
                env_extra['PYMUPDF_SETUP_MUPDF_BUILD'] = pipcl.git_get(
                        local='mupdf-local',
                        remote='https://github.com/ArtifexSoftware/mupdf.git',
                        text=mupdf,
                        branch='master',
                        )
            if pymupdf:
                pymupdf = pipcl.git_get(
                        local='pymupdf-local',
                        remote='https://github.com/pymupdf/PyMuPDF.git',
                        branch='main',
                        text=pymupdf,
                        )
                pipcl.run(f'pip uninstall -y pymupdf', check=0, env_extra=env_extra)
                if command == 'wheel':
                    newfiles = pipcl.NewFiles(f'wheelhouse/*.whl')
                    pipcl.run(f'pip wheel -w wheelhouse -v {pymupdf}', env_extra=env_extra)
                    pymupdf_wheel = newfiles.get_one()
                    pipcl.run(f'pip install -v {pymupdf_wheel}', env_extra=env_extra)
                else:
                    pipcl.run(f'pip install -v {pymupdf}', env_extra=env_extra, prefix=f'## pip install pymupdf/: ')
                pipcl.run(f'pip install -v -U swig', env_extra=env_extra, prefix=f'## pip install swig: ')
                # Tell our setup.py not to return pymupdf from
                # sce/setup.py:get_requires_for_build_wheel(), or specify it as
                # runtime requirement in <requires_dist>.
                env_extra['SCE_SETUP_BUILD_PYMUPDF'] = '1'
                no_build_isolation = ' --no-build-isolation'
            
            pipcl.run(f'pip uninstall -y pymupdf_layout', check=0, env_extra=env_extra)
            if command == 'build':
                pipcl.run(f'pip install{no_build_isolation} -v {g_root_abs}', env_extra=env_extra, prefix=f'## pip install sce/: ')
            elif command == 'wheel':
                new_files = pipcl.NewFiles(f'wheelhouse/pymupdf_layout-*.whl')
                pipcl.run(f'pip wheel{no_build_isolation} -w wheelhouse -v {g_root_abs}', env_extra=env_extra, prefix=f'## pip wheel sce/: ')
                wheel = new_files.get_one()
                pipcl.run(f'pip install {wheel}', env_extra=env_extra)
                if pymupdf:
                    pipcl.log(f'## Have built wheel: {pymupdf_wheel}')
                pipcl.log(f'## Have built wheel: {wheel}')
            else:
                assert 0
        
        elif command == 'cibw':
            
            print(f'### Building wheels using cibuildwheel.')
            
            # Need to get sot checkout here because (on linux at least)
            # there is no `ssh` command available inside cibuildwheel's
            # docker. And we put the sot checkout within PyMuPDFPro/ so that it
            # is available inside docker.
            #
            print(f'Importing setup.py')
            sys.path.insert(0, g_root)
            try:
                import setup
            finally:
                del sys.path[0]
            
            # Specify python versions.
            CIBW_BUILD = env_extra.get('CIBW_BUILD')
            pipcl.log(f'{CIBW_BUILD=}')
            if CIBW_BUILD is None:
                if 1 or os.environ.get('GITHUB_ACTIONS') == 'true':
                    # Build/test all supported Python versions.
                    # 2025-10-14: we don't attempt to test with python-3.14
                    # because our prerequisite onnxruntime does not currently
                    # support it.
                    if linux_aarch64 and platform.system() == 'Linux':
                        # Build is slow, and testing all python versions
                        # overruns Github's 6h timeout.
                        CIBW_BUILD = 'cp39* cp313*'
                    else:
                        CIBW_BUILD = 'cp39* cp310* cp311* cp312* cp313*'
                else:
                    # Build/test current Python only.
                    v = platform.python_version_tuple()[:2]
                    v = ''.join(v)
                    pipcl.log(f'{v=}')
                    CIBW_BUILD = f'cp{v}*'
            
            pipcl.run(f'pip install --upgrade cibuildwheel')
            env_extra['CIBW_BUILD_VERBOSITY'] = '1'
            # `cp3??t-*` excludes free-threading, which currently fails tests.
            env_extra['CIBW_SKIP'] = '*i686 *musllinux* *-win32 *-aarch64 cp3??t-*'
            if linux_aarch64:
                env_extra['CIBW_ARCHS_LINUX'] = 'aarch64'
            else:
                env_extra['CIBW_ARCHS_LINUX'] = 'auto64'
            env_extra['CIBW_ARCHS_WINDOWS'] = 'auto64'
            env_extra['CIBW_ARCHS_MACOS'] = 'auto64'

            # Build will run inside a CentOS-7 container; it looks
            # like we need to install fontconfig-devel so `#include
            # <fontconfig/fonctconfig.h>` works. And for SO build we need ssh
            # to allow its git submodule commands.
            #
            env_extra['CIBW_BEFORE_BUILD_LINUX'] = (
                    'echo "installing fontconfig-devel and ssh"'
                    ' && yum -y install fontconfig-devel'
                    ' && yum groupinstall -y fonts'
                    ' && yum install -y openssh-clients'
                    )
            
            # Tell cibuildwheel not to use `auditwheel` on Linux and MacOS,
            # because it cannot cope with us deliberately having required
            # libraries in different wheel - specifically in the PyMuPDF or
            # PyMuPDFb wheel.
            #
            # We cannot use a subset of auditwheel's functionality
            # with `auditwheel addtag` because it says `No tags
            # to be added` and terminates with non-zero. See:
            # https://github.com/pypa/auditwheel/issues/439.
            #
            env_extra['CIBW_REPAIR_WHEEL_COMMAND_LINUX'] = ''
            env_extra['CIBW_REPAIR_WHEEL_COMMAND_MACOS'] = ''
            
            # Tell cibuildwheel how to test PyMuPDFPro.
            env_extra['CIBW_TEST_COMMAND'] = f'python {{project}}/scripts/test.py test'
            
            # Pass all the environment variables we have set, to Linux
            # docker. Note that this will miss any settings in the original
            # environment.
            env_extra['CIBW_ENVIRONMENT_PASS_LINUX'] = ' '.join(sorted(env_extra.keys()))

            env_extra['CIBW_BUILD'] = CIBW_BUILD
            pipcl.run(f'cd {g_root} && cibuildwheel', env_extra=env_extra, prefix='cibw: ')
            pipcl.run(f'ls -ld {g_root}/wheelhouse/*')
        
        elif command == 'test':
            pipcl.run(f'pip install -U pytest')
            
            command = ''
            
            if pytest_prefix is None:
                pass
            elif pytest_prefix == 'gdb':
                command += 'gdb --args '
            elif pytest_prefix == 'valgrind':
                env_extra['PYMUPDF_RUNNING_ON_VALGRIND'] = '1'
                env_extra['PYTHONMALLOC'] = 'malloc'
                command += (
                        f'valgrind'
                        f' --suppressions={pymupdf_dir_abs}/valgrind.supp'
                        f' --trace-children=no'
                        f' --num-callers=20'
                        f' --error-exitcode=100'
                        f' --errors-for-leak-kinds=none'
                        f' --fullpath-after='
                        f' '
                        )
            elif pytest_prefix == 'helgrind':
                env_extra['PYMUPDF_RUNNING_ON_VALGRIND'] = '1'
                env_extra['PYTHONMALLOC'] = 'malloc'
                command = (
                        f'valgrind'
                        f' --tool=helgrind'
                        f' --trace-children=no'
                        f' --num-callers=20'
                        f' --error-exitcode=100'
                        f' --fullpath-after='
                        f' '
                        )
            else:
                assert 0, f'Unrecognised {pytest_prefix=}'
            
            command += f'python -m pytest {g_root}/tests'
            
            if not pytest_args and pytest_prefix == 'valgrind':
                pytest_args = '-sv'
            if pytest_args:
                command += f' {pytest_args}'
            
            if pytest_prefix in ('valgrind', 'helgrind'):
                if 1:
                    pipcl.log('Installing valgrind.')
                    pipcl.run(f'sudo apt update')
                    pipcl.run(f'sudo apt install --upgrade valgrind')
                pipcl.run(f'valgrind --version')

            pipcl.run(command, env_extra=env_extra)
        
        else:
            assert 0, f'Unrecognised {command=}.'


if __name__ == '__main__':
    main()
