import os
import re
import platform
import shlex
import subprocess
import sys
import textwrap


def test_pylint():
    subprocess.run(f'pip install -U flake8 pylint codespell', shell=1, check=1)
    ignores = ''
    ignores += textwrap.dedent(
            '''
            C0103: Variable name "ARTIFEX_SOFTWARE_SSH_KEY" doesn't conform to snake_case naming style (invalid-name)
            C0115: Missing class docstring (missing-class-docstring)
            C0116: Missing function or method docstring (missing-function-docstring)
            C0301: Line too long (131/100) (line-too-long)
            C0302: Too many lines in module (1652/1000) (too-many-lines)
            C0303: Trailing whitespace (trailing-whitespace)
            C3001: Lambda expression assigned to a variable. Define a function using the "def" keyword instead. (unnecessary-lambda-assignment)
            R0902: Too many instance attributes (31/7) (too-many-instance-attributes)
            R0903: Too few public methods (0/2) (too-few-public-methods)
            R0913: Too many arguments (7/5) (too-many-arguments)
            R0914: Too many local variables (108/15) (too-many-locals)
            R1705: Unnecessary "else" after "return", remove the "else" and de-indent the code inside it (no-else-return)
            R1713: Consider using str.join(sequence) for concatenating strings from an iterable (consider-using-join)
            R1726: Boolean condition ... may be simplified to ... (simplifiable-condition)
            R1734: Consider using [] instead of list() (use-list-literal)
            R1735: Consider using '{}' instead of a call to 'dict'. (use-dict-literal)
            W0108: Lambda may not be necessary (unnecessary-lambda)
            W0125: Using a conditional statement with a constant value (using-constant-test)
            W0201: Attribute 'cibw_name' defined outside __init__ (attribute-defined-outside-init)
            W0511: fixme: be able to set to '' for system install? (fixme)
            W0718: Catching too general exception Exception (broad-exception-caught)
            W0719: Raising too general exception: Exception (broad-exception-raised)
            W1309: Using an f-string that does not have any interpolated variables (f-string-without-interpolation)
            C0325: Unnecessary parens after '=' keyword (superfluous-parens)
            W1514: Using open without explicitly specifying an encoding (unspecified-encoding)
            R0915: Too many statements (56/50) (too-many-statements)
            C0411: standard import "fnmatch" should be placed before first party import "pipcl"  (wrong-import-order)
            R1707: Disallow trailing comma tuple (trailing-comma-tuple)
            R1710: Either all return statements in a function should return an expression, or none of them should. (inconsistent-return-statements)
            R1723: Unnecessary "elif" after "break", remove the leading "el" from "elif" (no-else-break)
            R1714: Consider merging these comparisons with 'in' by using 'to_ in ('/', '')'. Use a set instead if elements are hashable. (consider-using-in)
            W0621: Redefining name 'verbose' from outer scope (line 3263) (redefined-outer-name)
            W0640: Cell variable write2 defined in loop (cell-var-from-loop)
            W0603: Using the global statement (global-statement)
            R0912: Too many branches (52/12) (too-many-branches)
            C0415: Import outside toplevel (pipcl) (import-outside-toplevel)
            C0114: Missing module docstring (missing-module-docstring)
            R0801: Similar lines in 2 files
            R1727: (condition-evals-to-constant)
            R0916: too-many-boolean-expressions
            '''
            )
    ignores_list = list()
    for line in ignores.split('\n'):
        if not line or line.startswith('#'):
            continue
        m = re.match('^(.....): ', line)
        assert m, f'Failed to parse {line=}'
        ignores_list.append(m.group(1))
    ignores = ','.join(ignores_list)
    
    root = os.path.normpath(f'{__file__}/../..')
    sys.path.insert(0, root)
    try:
        import pipcl
    finally:
        del sys.path[0]
        
    directory = root
    leafs = pipcl.git_items(directory)
    command = f'pylint -d {ignores}'
    for leaf in leafs:
        if leaf.endswith('.py'):
            command += f' {directory}/{leaf}'
    print(f'Running: {command}')
    subprocess.run(command, shell=1, check=1)


def test_codespell():
    '''
    Check rebased Python code with codespell.
    '''
    root = os.path.abspath(f'{__file__}/../..')
    
    # For now we ignore files that we would ideally still look at, because it
    # is difficult to exclude some text sections.
    skips = textwrap.dedent('''
            ''')
    skips = skips.strip().replace('\n', ',')
    
    command = textwrap.dedent(f'''
            cd {root} && codespell
                --skip {shlex.quote(skips)}
                --ignore-multiline-regex 'codespell:ignore-begin.*codespell:ignore-end'
            ''')
    
    sys.path.insert(0, root)
    try:
        import pipcl
    finally:
        del sys.path[0]
    
    git_files = pipcl.git_items(root)
    for p in git_files:
        if p in ['tests/test_370_expected.md']:
            continue
        _, ext = os.path.splitext(p)
        if ext in ['.pdf']:
            continue
        command += f'    {p}\n'
    
    if platform.system() != 'Windows':
        command = command.replace('\n', ' \\\n')
    # Don't print entire command because very long, and will be displayed
    # anyway if there is an error.
    print(f'test_codespell(): Running: {command}')
    #print(f'Running codespell.')
    subprocess.run(command, shell=1, check=1)
    print('test_codespell(): codespell succeeded.')


def test_flake8():
    ignores = textwrap.dedent('''
            E123 closing bracket does not match indentation of opening bracket\'s line
            E124 closing bracket does not match visual indentation
            E126 continuation line over-indented for hanging indent
            E127 continuation line over-indented for visual indent
            E128 continuation line under-indented for visual indent
            E131 continuation line unaligned for hanging indent
            E201 whitespace after \'(\'
            E203 whitespace before \':\'
            E225 missing whitespace around operator
            E226 missing whitespace around arithmetic operator
            E227 missing whitespace around bitwise or shift operator
            E241 multiple spaces after \':\'
            E251 unexpected spaces around keyword / parameter equals
            E252 missing whitespace around parameter equals
            E261 at least two spaces before inline comment
            E265 block comment should start with \'# \'
            E271 multiple spaces after keyword
            E272 multiple spaces before keyword
            E302 expected 2 blank lines, found 1
            E305 expected 2 blank lines after class or function definition, found 1
            E306 expected 1 blank line before a nested definition, found 0
            E402 module level import not at top of file
            E501 line too long
            E701 multiple statements on one line (colon)
            E731 do not assign a lambda expression, use a def
            E741 ambiguous variable name \'l\'
            F541 f-string is missing placeholders
            W293 blank line contains whitespace
            W503 line break before binary operator
            W504 line break after binary operator
            ''')
    ignores = ignores.strip().split('\n')
    ignores2 = list()
    for i in ignores:
        if not i.startswith('#'):
            ignores2.append(i.split(' ', 1)[0])
    ignores = ','.join(ignores2)
    root = os.path.normpath(f'{__file__}/../..')
    print(f'{root=}')
    sys.path.insert(0, root)
    try:
        import pipcl
    finally:
        del sys.path[0]
    
    command = f'flake8 --ignore={ignores} --statistics'
    
    git_files = pipcl.git_items(root)
    for p in git_files:
        if p.endswith('.py'):
            command += f' {root}/{p}'
    
    print(f'test_flake8(): Running: {command}')
    subprocess.run(command, shell=1, check=1)
    print(f'test_flake8(): flake8 succeeded.')
