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

Parameters:

    -a <env_name>
        Read next space-separated argument(s) from environmental variable
        <env_name>.
        * Does nothing if <env_name> is unset.
        * Useful when running via Github action.
    
    -b <build>
        Set build type for `build` commands. `<build>` should be one of
        'release', 'debug', 'memento'.
    
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
        0
        1
        2
        3

Commands:
    build
    test
    wheel
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
        
        elif arg == '--sync-paths':
            sync_paths = next(args)
            
        elif arg in ('build', 'test', 'wheel'):
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
                pipcl.run(f'pip uninstall -y pymupdf', check=0)
                pipcl.run(f'pip install -v {pymupdf}', env_extra=env_extra, prefix=f'## pip install pymupdf/: ')
                pipcl.run(f'pip install -v -U swig', env_extra=env_extra, prefix=f'## pip install swig: ')
                # Tell our setup.py not to return pymupdf from
                # sce/setup.py:get_requires_for_build_wheel(), or specify it as
                # runtime requirement in <requires_dist>.
                env_extra['SCE_SETUP_BUILD_PYMUPDF'] = '1'
                no_build_isolation = ' --no-build-isolation'
            
            pipcl.run(f'pip uninstall -y pymupdfsce', check=0)
        
        if command == 'build':
            pipcl.run(f'pip install{no_build_isolation} -v {g_root_abs}', env_extra=env_extra, prefix=f'## pip install sce/: ')
        
        elif command == 'wheel':
            new_files = pipcl.NewFiles(f'wheelhouse/pymupdf_layout-*.whl')
            pipcl.run(f'pip wheel{no_build_isolation} -w wheelhouse -v {g_root_abs}', env_extra=env_extra, prefix=f'## pip wheel sce/: ')
            wheel = new_files.get_one()
            pipcl.run(f'pip install {wheel}')
        
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
                    run(f'sudo apt update')
                    run(f'sudo apt install --upgrade valgrind')
                run(f'valgrind --version')

            pipcl.run(command, env_extra=env_extra)
        
        else:
            assert 0, f'Unrecognised {command=}.'


if __name__ == '__main__':
    main()
