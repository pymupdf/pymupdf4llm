import os
import subprocess
import sys
import textwrap


g_root = os.path.abspath(f'{__file__}/../..')

def test_376():

    code = textwrap.dedent('''
            import os
            
            from pymupdf4llm import LlamaMarkdownReader
            import pymupdf
            
            g_root = os.path.abspath(f'{__file__}/../..')
            path = f'{g_root}/_test_376_out.pdf'
            
            with pymupdf.open() as document:
                document.new_page()
                document.save(path)

            reader = LlamaMarkdownReader()
            documents = reader.load_data(path)

            print(f"Loaded {len(documents)} document(s)")
            ''')
    path_py = f'{g_root}/tests/_test_376.py'
    with open(path_py, 'w') as f:
        f.write(code)
    
    subprocess.run(f'{sys.executable} {path_py}', shell=1, check=1)
